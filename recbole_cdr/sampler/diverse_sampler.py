"""
recbole_cdr.sampler.diverse_sampler
########################
Diversity-aware negative sampling for Push mechanism.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from .sampler import AbstractSampler


class DiverseSampler(AbstractSampler):
    """DiverseSampler implements diversity-aware negative sampling for Push mechanism.
    
    For non-overlapping users, samples hard negatives that align with the escape
    direction (away from overlapping users' centroid). This encourages users to
    explore beyond their information cocoons.
    
    For overlapping users, uses standard sampling.
    
    Args:
        phases (str or list of str): All the phases of input.
        datasets (Dataset or list of Dataset): All the dataset for each phase.
        distribution (str, optional): Distribution of the negative items. Defaults to 'uniform'.
        model: The Diverse_CMF model for computing escape directions.
        hardness_weight (float, optional): Weight for hardness adjustment. Defaults to 1.0.
        candidate_pool_size (int, optional): Size of candidate pool for selection. Defaults to 10.
    """
    
    def __init__(self, phases, datasets, distribution='uniform', model=None, 
                 hardness_weight=1.0, candidate_pool_size=10):
        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError(f'Phases {phases} and datasets {datasets} should have the same length.')
        
        self.phases = phases
        self.datasets = datasets
        self.model = model
        self.hardness_weight = hardness_weight
        self.candidate_pool_size = candidate_pool_size
        
        # Handle CrossDomainDataset (different structure from standard Dataset)
        # CrossDomainDataset has source_domain_dataset and target_domain_dataset
        if hasattr(datasets[0], 'target_domain_dataset'):
            # CrossDomainDataset case
            target_dataset = datasets[0].target_domain_dataset
            self.uid_field = target_dataset.uid_field
            self.iid_field = target_dataset.iid_field
            self.user_num = datasets[0].num_total_user
            self.item_num = datasets[0].num_total_item
        else:
            # Standard Dataset case
            self.uid_field = datasets[0].uid_field
            self.iid_field = datasets[0].iid_field
            self.user_num = datasets[0].user_num
            self.item_num = datasets[0].item_num
        
        super().__init__(distribution=distribution)
    
    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()"""
        candidates_list = []
        for dataset in self.datasets:
            # Handle CrossDomainDataset
            if hasattr(dataset, 'target_domain_dataset'):
                target_dataset = dataset.target_domain_dataset
                candidates_list.extend(target_dataset.inter_feat[self.iid_field].numpy())
            else:
                candidates_list.extend(dataset.inter_feat[self.iid_field].numpy())
        return candidates_list
    
    def _uni_sampling(self, sample_num):
        """Uniform sampling as baseline"""
        return np.random.randint(1, self.item_num, sample_num)
    
    def _compute_diverse_hardness(self, user_ids_tensor, candidate_items_tensor):
        """Compute hardness for diversity-aware sampling.
        
        Hardness = alignment_with_escape_direction + hardness_weight * prediction_score
        
        Args:
            user_ids_tensor: Tensor of user IDs [batch_size]
            candidate_items_tensor: Tensor of candidate item IDs [batch_size, num_candidates]
        
        Returns:
            hardness: Tensor of hardness scores [batch_size, num_candidates]
        """
        batch_size = user_ids_tensor.size(0)
        num_candidates = candidate_items_tensor.size(1)
        
        with torch.no_grad():
            # Get user embeddings
            user_e = self.model.get_user_embedding(user_ids_tensor)  # [batch_size, emb_dim]
            
            # Compute escape direction (for non-overlapping users)
            # This points away from overlapping users' centroid
            escape_direction = self.model.compute_escape_direction(user_e)  # [batch_size, emb_dim]
            
            # Get candidate item embeddings
            # Flatten to [batch_size * num_candidates]
            candidate_items_flat = candidate_items_tensor.reshape(-1)
            item_e_flat = self.model.get_item_embedding(candidate_items_flat)
            item_e = item_e_flat.reshape(batch_size, num_candidates, -1)  # [batch_size, num_candidates, emb_dim]
            
            # Term 1: Alignment with escape direction
            # We want items that are in the direction of escaping the cocoon
            user_e_expanded = user_e.unsqueeze(1)  # [batch_size, 1, emb_dim]
            item_diff = item_e - user_e_expanded  # [batch_size, num_candidates, emb_dim]
            
            # Normalize for cosine similarity
            item_diff_norm = F.normalize(item_diff, p=2, dim=2)
            escape_direction_norm = F.normalize(escape_direction, p=2, dim=1).unsqueeze(1)
            
            # Cosine similarity: higher = better alignment with escape direction
            alignment = (item_diff_norm * escape_direction_norm).sum(dim=2)  # [batch_size, num_candidates]
            
            # Term 2: Prediction score (decision boundary distance)
            # We want items near the decision boundary (score ≈ 0.5), not high-confidence positives (score → 1.0)
            # Use score * (1 - score), which peaks at 0.5 and is 0 at both 0 and 1
            # This prevents treating high-confidence predictions as hard negatives (False Negatives)
            scores = torch.mul(user_e_expanded, item_e).sum(dim=2)  # [batch_size, num_candidates]
            scores = torch.sigmoid(scores)
            
            # Boundary distance: score * (1 - score), peaks at 0.5
            boundary_dist = scores * (1.0 - scores)  # [batch_size, num_candidates]
            
            # Combined hardness
            # Both terms encourage selecting items that:
            # 1. Align with escape direction (diversity)
            # 2. Are near decision boundary (hard negatives, not false negatives)
            hardness = alignment + self.hardness_weight * boundary_dist
            
            return hardness
    
    def _diverse_sampling(self, user_ids, sample_num):
        """Diversity-aware sampling for non-overlapping users.
        
        Args:
            user_ids: Array of user IDs
            sample_num: Number of samples per user (should be 1 in RecBole)
            
        Returns:
            sampled_items: Array of sampled item IDs [num_users]
        """
        # Convert user_ids to numpy array
        user_ids = np.array(user_ids)
        device = next(self.model.parameters()).device
        
        # Identify overlapping vs non-overlapping users
        if hasattr(self.model, 'overlapping_users') and self.model.overlapping_users is not None:
            is_overlapping = np.array([uid in self.model.overlapping_users for uid in user_ids])
        else:
            # If model doesn't have overlapping_users, treat all as non-overlapping
            is_overlapping = np.zeros(len(user_ids), dtype=bool)
        
        # Separate users
        overlap_indices = np.where(is_overlapping)[0]
        non_overlap_indices = np.where(~is_overlapping)[0]
        
        # Initialize result array
        sampled_items = np.zeros(len(user_ids), dtype=int)
        
        # For overlapping users: standard uniform sampling
        if len(overlap_indices) > 0:
            sampled_items[overlap_indices] = self._uni_sampling(len(overlap_indices))
        
        # For non-overlapping users: diversity-aware hard negative sampling
        if len(non_overlap_indices) > 0 and self.model.overlap_centroid is not None:
            non_overlap_user_ids = user_ids[non_overlap_indices]
            hard_negatives = self._batch_diverse_sampling(non_overlap_user_ids, device)
            sampled_items[non_overlap_indices] = hard_negatives
        elif len(non_overlap_indices) > 0:
            # Fallback: uniform sampling if no overlap centroid
            sampled_items[non_overlap_indices] = self._uni_sampling(len(non_overlap_indices))
        
        return sampled_items
    
    def _batch_diverse_sampling(self, user_ids, device):
        """Batch process diversity-aware sampling for multiple non-overlapping users.
        
        Args:
            user_ids: Numpy array of non-overlapping user IDs
            device: Device for tensor operations
            
        Returns:
            sampled_items: Numpy array of sampled item IDs
        """
        # Convert to tensors
        user_ids_tensor = torch.from_numpy(user_ids).long().to(device)
        
        # Stage 1: Sample candidate pool for all users
        num_candidates = min(self.candidate_pool_size, self.item_num - 1)
        candidate_items = np.arange(1, self.item_num)  # Exclude padding (0)
        
        # Vectorized candidate sampling
        if len(user_ids) > 1:
            # Ultra-efficient batch sampling
            random_vals = torch.rand(len(user_ids), num_candidates, device=device)
            item_indices = (random_vals * len(candidate_items)).long()
            item_indices = torch.clamp(item_indices, 0, len(candidate_items) - 1)
            candidate_tensor = torch.from_numpy(candidate_items).long().to(device)
            all_candidates = candidate_tensor[item_indices]  # [num_users, num_candidates]
        else:
            # Single user case
            indices = torch.randperm(len(candidate_items), device=device)[:num_candidates]
            candidate_tensor = torch.from_numpy(candidate_items).long().to(device)
            all_candidates = candidate_tensor[indices].unsqueeze(0)
        
        # Stage 2: Compute hardness scores for all user-candidate pairs
        hardness_scores = self._compute_diverse_hardness(user_ids_tensor, all_candidates)
        
        # Stage 3: Temperature-scaled selection
        # Compute temperature based on cocoon strength
        # Higher cocoon strength → lower temperature → sharper distribution → harder negatives
        user_e = self.model.get_user_embedding(user_ids_tensor)
        cocoon_strength = self.model.compute_cocoon_strength(user_e)  # [num_users]
        
        # tau = exp(-c * cocoon_strength)
        # Higher cocoon strength → more strongly trapped → need harder push
        tau = torch.exp(-self.hardness_weight * cocoon_strength).unsqueeze(1)  # [num_users, 1]
        
        # Temperature-scaled softmax
        selection_weights = torch.exp(hardness_scores / tau)
        selection_weights = selection_weights / selection_weights.sum(dim=1, keepdim=True)
        
        # Sample based on weights
        selected_indices = torch.multinomial(selection_weights, num_samples=1).squeeze(1)
        final_items = all_candidates[torch.arange(len(user_ids), device=device), selected_indices]
        
        return final_items.cpu().numpy()
    
    def sampling(self, sample_num, user_ids=None):
        """Sampling [sample_num] item_ids with diversity awareness.
        
        Args:
            sample_num (int): the number of samples.
            user_ids: Array of user IDs (required for diversity sampling)
        
        Returns:
            sample_list (np.array): a list of samples.
        """
        if user_ids is not None and self.model is not None:
            return self._diverse_sampling(user_ids, sample_num)
        elif self.distribution == 'uniform':
            return self._uni_sampling(sample_num)
        elif self.distribution == 'popularity':
            return self._pop_sampling(sample_num)
        else:
            raise NotImplementedError(f'The sampling distribution [{self.distribution}] is not implemented.')
    
    def get_used_ids(self):
        """
        Returns:
            dict: Used item_ids is the same as positive item_ids.
            Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        used_item_id = dict()
        last = [set() for _ in range(self.user_num)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            
            # Handle CrossDomainDataset
            if hasattr(dataset, 'target_domain_dataset'):
                target_dataset = dataset.target_domain_dataset
                uid_data = target_dataset.inter_feat[self.uid_field].numpy()
                iid_data = target_dataset.inter_feat[self.iid_field].numpy()
            else:
                uid_data = dataset.inter_feat[self.uid_field].numpy()
                iid_data = dataset.inter_feat[self.iid_field].numpy()
            
            for uid, iid in zip(uid_data, iid_data):
                cur[uid].add(iid)
            last = used_item_id[phase] = cur

        for used_item_set in used_item_id[self.phases[-1]]:
            if len(used_item_set) + 1 == self.item_num:  # [pad] is a item.
                raise ValueError(
                    'Some users have interacted with all items, '
                    'which we can not sample negative items for them. '
                    'Please set `user_inter_num_interval` to filter those users.'
                )
        return used_item_id
    
    def set_phase(self, phase):
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            DiverseSampler: the copy of this sampler, :attr:`phase` is set the same as input phase, and :attr:`used_ids`
            is set to the value of corresponding phase.
        """
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler
    
    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids with diversity awareness.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
        """
        try:
            # Convert user_ids to numpy array
            user_ids = np.array(user_ids)
            
            # Use diversity-aware sampling
            sampled_items = self._diverse_sampling(user_ids, num)
            
            return torch.tensor(sampled_items)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')
    
    def set_model(self, model):
        """Set the model for computing escape directions and hardness scores.
        
        Args:
            model: The Diverse_CMF model
        """
        self.model = model

