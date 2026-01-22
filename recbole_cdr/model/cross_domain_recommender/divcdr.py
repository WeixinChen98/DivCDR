import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import EmbLoss, BPRLoss

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class DivCDR(CrossDomainRecommender):
    r"""DivCDR extends CMF with diversity-aware mechanisms.
    
    It uses a push-pull approach:
    - Pull mechanism: Progressive diversity augmentation for overlapping users
    - Push mechanism: Hard negative sampling for non-overlapping users
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DivCDR, self).__init__(config, dataset)
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        self.phase = 'both'

        # Load parameters from CMF
        self.embedding_size = config['embedding_size']
        self.alpha_cmf = config['alpha']  # CMF's alpha for loss weighting
        self.lamda = config['lambda']  # Source regularization weight
        self.gamma = config['gamma']  # Target regularization weight (from CMF)

        # Diversity-specific parameters
        self.enable_pull = config['enable_pull']  # Whether to enable pull mechanism
        self.enable_push = config['enable_push']  # Whether to enable push mechanism
        
        # Check if DiverseSampler is used (replaces hard negatives in loss)
        self.use_diverse_sampler = config['use_diverse_sampler'] if 'use_diverse_sampler' in config else False
        
        # Pull mechanism parameters
        self.pull_weight = config['pull_weight'] if 'pull_weight' in config else 0.1  # Weight for diversity augmentation loss
        self.aug_strength = config['aug_strength'] if 'aug_strength' in config else 0.1
        self.info_loss_eta = config['info_loss_eta'] if 'info_loss_eta' in config else 0.01
        self.entropy_weight = config['entropy_weight'] if 'entropy_weight' in config else 1.0
        self.div_weight = config['div_weight'] if 'div_weight' in config else 1.0
        
        # Push mechanism parameters (used by DiverseSampler for negative sampling)
        self.hardness_weight = config['hardness_weight'] if 'hardness_weight' in config else 1.0
        self.candidate_pool_size = config['candidate_pool_size'] if 'candidate_pool_size' in config else 10
        
        # Define layers and loss (same as CMF)
        self.user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
        
        # Augmentation network for pull mechanism
        if self.enable_pull:
            self.aug_network = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size * 2),
                nn.ReLU(),
                nn.Linear(self.embedding_size * 2, self.embedding_size)
            )
            
            # Alpha predictor network (controls augmentation strength)
            self.alpha_network = nn.Sequential(
                nn.Linear(self.embedding_size * 2 + 1, 64),  # user, item, info_loss
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.source_reg_loss = EmbLoss()
        self.target_reg_loss = EmbLoss()

        # Parameters initialization
        self.apply(xavier_normal_initialization)
        
        # Store overlapping users (will be set during training)
        self.overlapping_users = None
        
        # Store DiverseSampler reference (if use_diverse_sampler=True)
        self.diverse_sampler = None
        
        # Register buffer for overlap centroid (Push mechanism)
        # This will be computed once per epoch for efficiency
        self.register_buffer('overlap_centroid', None)

    def set_phase(self, phase):
        self.phase = phase
    
    def set_overlapping_users(self, overlapping_users):
        """Set the overlapping users set."""
        if isinstance(overlapping_users, set):
            self.overlapping_users = overlapping_users
        elif isinstance(overlapping_users, (list, np.ndarray)):
            self.overlapping_users = set(overlapping_users)
        else:
            self.overlapping_users = set()
    
    def set_diverse_sampler(self, diverse_sampler):
        """Set the DiverseSampler (called by trainer if use_diverse_sampler=True)."""
        self.diverse_sampler = diverse_sampler
        # When DiverseSampler is set, we know hard negatives are handled at DataLoader level
        self.use_diverse_sampler = True
    
    def update_overlap_centroid(self):
        """Update the centroid of overlapping users' embeddings.
        
        This should be called at the start of each epoch to compute the mean
        embedding of all overlapping users. Using a static centroid instead of
        sampling random users provides:
        1. Efficiency: O(1) lookup vs O(K) sampling per batch
        2. Stability: Consistent gradient direction across batches
        3. Representativeness: Uses ALL overlapping users, not a sample
        """
        if self.overlapping_users is None or len(self.overlapping_users) == 0:
            # No overlapping users, set centroid to None
            self.overlap_centroid = None
            return
        
        # Convert overlapping users to tensor
        overlap_user_ids = torch.tensor(
            list(self.overlapping_users),
            dtype=torch.long,
            device=self.user_embedding.weight.device
        )
        
        # Get embeddings of all overlapping users
        with torch.no_grad():  # No gradient needed for centroid computation
            overlap_embeddings = self.get_user_embedding(overlap_user_ids)  # [num_overlap, emb_dim]
            
            # Compute mean (centroid)
            centroid = overlap_embeddings.mean(dim=0)  # [emb_dim]
            
            # Store as buffer (will be moved to correct device automatically)
            self.overlap_centroid = centroid

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id."""
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id."""
        return self.item_embedding(item)

    def forward(self, user, item):
        """Standard forward pass (same as CMF)."""
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))
    
    # ==================== Pull Mechanism ====================
    
    def estimate_information_loss(self, user_e, item_e):
        """Estimate information loss by adding small Gaussian noise.
        
        This measures how sensitive the model is to perturbations.
        High IL means the model is fragile -> use smaller alpha.
        Low IL means the model is robust -> use larger alpha.
        
        Args:
            user_e: user embeddings [batch_size, emb_dim]
            item_e: item embeddings [batch_size, emb_dim]
        
        Returns:
            info_loss: estimated information loss [batch_size]
        """
        batch_size = user_e.size(0)
        device = user_e.device
        
        # Original prediction
        orig_score = torch.mul(user_e, item_e).sum(dim=1)
        
        # Add small Gaussian noise
        noise = torch.randn_like(item_e) * self.info_loss_eta
        noisy_item_e = item_e + noise
        
        # Noisy prediction
        noisy_score = torch.mul(user_e, noisy_item_e).sum(dim=1)
        
        # Information loss (L2 distance)
        info_loss = (orig_score - noisy_score).pow(2)
        
        return info_loss
    
    def compute_adaptive_alpha(self, user_e, item_e):
        """Compute adaptive augmentation strength alpha.
        
        Args:
            user_e: user embeddings [batch_size, emb_dim]
            item_e: item embeddings [batch_size, emb_dim]
        
        Returns:
            alpha: augmentation strength [batch_size, 1]
        """
        # Estimate information loss
        info_loss = self.estimate_information_loss(user_e, item_e)  # [batch_size]
        
        # Concatenate features: [user_e, item_e, info_loss]
        features = torch.cat([
            user_e, 
            item_e, 
            info_loss.unsqueeze(1)
        ], dim=1)  # [batch_size, emb_dim*2 + 1]
        
        # Predict alpha using MLP
        alpha = self.alpha_network(features)  # [batch_size, 1]
        
        return alpha
    
    def generate_augmentation(self, user_e):
        """Generate augmentation vector epsilon.
        
        Args:
            user_e: user embeddings [batch_size, emb_dim]
        
        Returns:
            epsilon: augmentation vector [batch_size, emb_dim]
        """
        epsilon = self.aug_network(user_e)
        return epsilon
    
    def augment_positive_samples(self, user, item):
        """Apply progressive diversity augmentation to positive samples.
        
        This is the PULL mechanism for overlapping users.
        
        Args:
            user: user IDs [batch_size]
            item: item IDs [batch_size]
        
        Returns:
            augmented_item_e: augmented item embeddings [batch_size, emb_dim]
            aug_loss: augmentation loss to maximize diversity
        """
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        
        # Generate augmentation
        epsilon = self.generate_augmentation(user_e)  # [batch_size, emb_dim]
        
        # Compute adaptive alpha (network output in [0, 1])
        alpha_raw = self.compute_adaptive_alpha(user_e, item_e)  # [batch_size, 1]
        
        # Scale alpha by aug_strength to keep augmentations bounded
        # This ensures augmentations stay within semantic neighborhood
        alpha = alpha_raw * self.aug_strength  # [batch_size, 1]
        
        # Augment item embeddings: v_i' = v_i + alpha * epsilon
        augmented_item_e = item_e + alpha * epsilon
        
        # Compute augmentation loss (to be minimized, which maximizes diversity)
        # This will be implemented in calculate_aug_loss
        aug_loss = self.calculate_aug_loss(user_e, item_e, augmented_item_e, alpha, epsilon)
        
        return augmented_item_e, aug_loss
    
    def calculate_aug_loss(self, user_e, item_e, augmented_item_e, alpha, epsilon):
        """Calculate augmentation loss to maximize diversity for individual users.
        
        Loss = -(ΔH + ΔDiv)
        where:
        - ΔH: change in entropy (uncertainty) for individual predictions
        - ΔDiv: per-user distance between augmented and original item embeddings
        
        This focuses on expanding INDIVIDUAL users' cocoons, not enforcing diversity
        across different users' items.
        
        IMPORTANT: This loss should ONLY update the generator networks (aug_network, alpha_network),
        NOT the recommendation embeddings (user_embedding, item_embedding).
        Therefore, we detach user_e and item_e to prevent gradient flow to embeddings.
        
        Args:
            user_e: original user embeddings [batch_size, emb_dim]
            item_e: original item embeddings [batch_size, emb_dim]
            augmented_item_e: augmented item embeddings [batch_size, emb_dim]
            alpha: augmentation strength [batch_size, 1]
            epsilon: augmentation vector [batch_size, emb_dim]
        
        Returns:
            aug_loss: scalar loss (gradients only flow to generator)
        """
        # Detach embeddings to prevent gradient flow to recommendation embeddings
        # Only augmented_item_e should have gradients (to train the generator)
        user_e_detached = user_e.detach()
        item_e_detached = item_e.detach()
        
        # ========== 1. Entropy Maximization (ΔH) ==========
        # Measure uncertainty/exploration for individual users
        # Higher entropy = more uncertain predictions = exploring beyond cocoon
        
        # Compute prediction scores
        orig_scores = torch.mul(user_e_detached, item_e_detached).sum(dim=1)  # [batch_size]
        aug_scores = torch.mul(user_e_detached, augmented_item_e).sum(dim=1)  # [batch_size]
        
        # Convert to probabilities
        orig_probs = torch.sigmoid(orig_scores)
        aug_probs = torch.sigmoid(aug_scores)
        
        # Shannon entropy H(p) = -p*log(p) - (1-p)*log(1-p)
        def entropy(p):
            p = torch.clamp(p, 1e-7, 1-1e-7)  # Avoid log(0)
            return -(p * torch.log(p) + (1-p) * torch.log(1-p))
        
        orig_entropy = entropy(orig_probs)
        aug_entropy = entropy(aug_probs)
        delta_H = (aug_entropy - orig_entropy).mean()  # Average over batch
        
        # ========== 2. Per-User Diversity (ΔDiv) ==========
        # Maximize distance between augmented_item_e and original item_e
        # This pushes the augmented item AWAY from the original, expanding the cocoon
        
        # Normalize embeddings for cosine similarity
        item_e_norm = F.normalize(item_e_detached, p=2, dim=1)  # [batch_size, emb_dim]
        aug_item_e_norm = F.normalize(augmented_item_e, p=2, dim=1)  # [batch_size, emb_dim]
        
        # Cosine similarity between original and augmented items (per-user)
        # Shape: [batch_size] (element-wise dot product)
        cosine_sim = (item_e_norm * aug_item_e_norm).sum(dim=1)
        
        # Cosine distance = 1 - cosine_similarity
        # Higher distance = more diverse from original
        cosine_dist = 1 - cosine_sim  # [batch_size]
        
        # Weight by alpha: stronger augmentation should lead to larger distance
        # alpha: [batch_size, 1], squeeze to [batch_size]
        alpha_squeezed = alpha.squeeze(1) if alpha.dim() > 1 else alpha
        weighted_dist = cosine_dist * alpha_squeezed  # [batch_size]
        
        # Average distance as diversity metric
        delta_Div = weighted_dist.mean()
        
        # ========== 3. Epsilon Regularization ==========
        # Add L2 regularization on epsilon to prevent generator explosion
        # This keeps the noise magnitude bounded
        epsilon_norm = torch.norm(epsilon, p=2, dim=1).mean()  # Average L2 norm
        
        # ========== 4. Combined Loss ==========
        # Minimize -(ΔH + ΔDiv) + λ * ||ε||_2
        # The first two terms encourage diversity, the last term prevents explosion
        aug_loss = -(self.entropy_weight * delta_H + self.div_weight * delta_Div) + 1e-3 * epsilon_norm
        
        return aug_loss
    
    # ==================== Push Mechanism ====================
    
    def compute_cocoon_strength(self, user_e):
        """Compute cocoon strength for non-overlapping users.
        
        Simplified version using the overlap centroid:
        C(u) = exp(-||h_u - c_overlap||^2)
        
        where c_overlap is the mean embedding of all overlapping users.
        
        This is more efficient (O(N) vs O(N*K)) and more stable than
        sampling K random overlapping users each time.
        
        Args:
            user_e: embeddings of non-overlapping users [batch_size, emb_dim]
        
        Returns:
            cocoon_strength: [batch_size]
        """
        if self.overlap_centroid is None:
            # No overlapping users, return zero strength
            return torch.zeros(user_e.size(0), device=user_e.device)
        
        # Compute squared distance to centroid
        # user_e: [batch_size, emb_dim]
        # overlap_centroid: [emb_dim]
        dist_sq = ((user_e - self.overlap_centroid) ** 2).sum(dim=1)  # [batch_size]
        
        # Compute cocoon strength
        strength = torch.exp(-dist_sq)  # [batch_size]
        
        return strength
    
    def compute_escape_direction(self, user_e):
        """Compute the gradient direction to escape the cocoon.
        
        d_u = -∇_{h_u} C(u)
        
        Using the overlap centroid, this simplifies to (analytical form):
        d_u = 2 * exp(-||h_u - c||^2) * (h_u - c)
        
        which points away from the overlap centroid.
        
        Args:
            user_e: embeddings of non-overlapping users [batch_size, emb_dim]
        
        Returns:
            escape_direction: normalized direction vectors [batch_size, emb_dim]
        """
        if self.overlap_centroid is None:
            # No overlapping users, return random normalized directions
            escape_direction = torch.randn_like(user_e)
            escape_direction = F.normalize(escape_direction, p=2, dim=1)
            return escape_direction
        
        # Compute analytical gradient (no autograd needed)
        # C(u) = exp(-||u - c||^2)
        # ∇_u C(u) = -2 * exp(-||u - c||^2) * (u - c)
        # d_u = -∇_u C(u) = 2 * exp(-||u - c||^2) * (u - c)
        
        diff = user_e - self.overlap_centroid  # [batch_size, emb_dim]
        dist_sq = (diff ** 2).sum(dim=1, keepdim=True)  # [batch_size, 1]
        exp_term = torch.exp(-dist_sq)  # [batch_size, 1]
        
        # Escape direction (analytical gradient)
        escape_direction = 2 * exp_term * diff  # [batch_size, emb_dim]
        
        # Normalize
        escape_direction = F.normalize(escape_direction, p=2, dim=1)
        
        return escape_direction.detach()  # Detach to prevent gradient flow
    
    def compute_negative_hardness(self, user_e, neg_item_e, escape_direction):
        """Compute hardness score for negative samples.
        
        Hardness(i,u) = cos(h_i - h_u, d_u) + A * cos(h_u^T h_i)
        
        Args:
            user_e: user embeddings [batch_size, emb_dim]
            neg_item_e: negative item embeddings [batch_size, num_neg, emb_dim]
            escape_direction: escape direction for users [batch_size, emb_dim]
        
        Returns:
            hardness: hardness scores [batch_size, num_neg]
        """
        batch_size, num_neg, emb_dim = neg_item_e.shape
        
        # Term 1: cos(h_i - h_u, d_u) - alignment with escape direction
        user_e_expanded = user_e.unsqueeze(1)  # [batch_size, 1, emb_dim]
        diff = neg_item_e - user_e_expanded  # [batch_size, num_neg, emb_dim]
        diff_norm = F.normalize(diff, p=2, dim=2)  # [batch_size, num_neg, emb_dim]
        escape_direction_expanded = escape_direction.unsqueeze(1)  # [batch_size, 1, emb_dim]
        
        term1 = (diff_norm * escape_direction_expanded).sum(dim=2)  # [batch_size, num_neg]
        
        # Term 2: cos(h_u^T h_i) - decision boundary distance
        user_e_norm = F.normalize(user_e, p=2, dim=1).unsqueeze(1)  # [batch_size, 1, emb_dim]
        neg_item_e_norm = F.normalize(neg_item_e, p=2, dim=2)  # [batch_size, num_neg, emb_dim]
        
        term2 = (user_e_norm * neg_item_e_norm).sum(dim=2)  # [batch_size, num_neg]
        
        # Combined hardness
        hardness = term1 + self.hardness_weight * term2
        
        return hardness

    # ==================== Loss Calculation ====================

    def calculate_loss(self, interaction):
        if self.phase == 'BOTH':
            return self.calculate_map_loss(interaction)
        else:
            return self.calculate_target_loss(interaction)

    def calculate_target_loss(self, interaction):
        """Calculate loss for target domain with diversity mechanisms.
        
        Gradient Flow Design:
        ---------------------
        For overlapping users (Pull mechanism):
        - loss_accuracy: BCE + reg → updates user_embedding & item_embedding
        - aug_loss: diversity objective → updates aug_network & alpha_network ONLY
                   (embeddings are detached in calculate_aug_loss)
        
        For non-overlapping users (Push mechanism):
        - loss with hard negatives → updates user_embedding & item_embedding
        
        This ensures that the diversity objective (aug_loss) trains the generator
        without degrading the recommendation embeddings.
        """
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]
        
        batch_size = target_user.size(0)
        
        # Identify overlapping vs non-overlapping users in this batch
        if self.overlapping_users is not None:
            user_ids_cpu = target_user.cpu().numpy()
            is_overlapping = np.array([uid in self.overlapping_users for uid in user_ids_cpu])
            overlap_mask = torch.tensor(is_overlapping, device=target_user.device)
        else:
            # Default: treat all as non-overlapping
            overlap_mask = torch.zeros(batch_size, dtype=torch.bool, device=target_user.device)
        
        # ========== Pull Mechanism (Overlapping Users) ==========
        total_loss = torch.tensor(0.0, device=target_user.device)
        aug_loss_total = 0.0
        
        if self.enable_pull and overlap_mask.sum() > 0:
            overlap_indices = overlap_mask.nonzero(as_tuple=True)[0]
            overlap_users = target_user[overlap_indices]
            overlap_items = target_item[overlap_indices]
            overlap_labels = target_label[overlap_indices]
            
            # Apply augmentation (only if we have more than 1 user for pairwise diversity)
            if len(overlap_users) >= 2:
                # Generate augmented samples and diversity loss
                augmented_item_e, aug_loss = self.augment_positive_samples(overlap_users, overlap_items)
                user_e = self.get_user_embedding(overlap_users)
                item_e = self.get_item_embedding(overlap_items)
                
                # ===== Accuracy Loss (trains embeddings) =====
                # Prediction with augmented items
                p = self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))
                p_aug = self.sigmoid(torch.mul(user_e, augmented_item_e).sum(dim=1))
                
                # BCE loss for accuracy
                loss_accuracy = self.loss(p_aug, overlap_labels) + self.loss(p, overlap_labels)
                
                # Regularization
                reg_loss = self.target_reg_loss(user_e, augmented_item_e)
                loss_accuracy = loss_accuracy + self.gamma * reg_loss
                
                # ===== Diversity Loss (trains generator only) =====
                # aug_loss already detaches embeddings in calculate_aug_loss
                # so it only updates aug_network and alpha_network
                if isinstance(aug_loss, torch.Tensor):
                    aug_loss_weighted = aug_loss * self.pull_weight
                    aug_loss_total = aug_loss.item()
                else:
                    aug_loss_weighted = aug_loss * self.pull_weight
                    aug_loss_total = aug_loss
                
                # ===== Combined Loss =====
                # Both losses contribute to total_loss:
                # - loss_accuracy updates user_embedding & item_embedding
                # - aug_loss_weighted updates aug_network & alpha_network
                loss_overlap = loss_accuracy + aug_loss_weighted
                
                total_loss = total_loss + loss_overlap * (overlap_indices.size(0) / batch_size)
            else:
                # Fallback: standard forward for single user
                p_overlap = self.forward(overlap_users, overlap_items)
                loss_overlap = self.loss(p_overlap, overlap_labels)
                loss_overlap = loss_overlap + self.gamma * self.target_reg_loss(
                    self.get_user_embedding(overlap_users),
                    self.get_item_embedding(overlap_items)
                )
                total_loss = total_loss + loss_overlap * (overlap_indices.size(0) / batch_size)
        
        # ========== Push Mechanism (Non-overlapping Users) ==========
        if self.enable_push and (~overlap_mask).sum() > 0:
            non_overlap_indices = (~overlap_mask).nonzero(as_tuple=True)[0]
            non_overlap_users = target_user[non_overlap_indices]
            non_overlap_items = target_item[non_overlap_indices]
            non_overlap_labels = target_label[non_overlap_indices]
            
            # Standard prediction for positive items
            p_non_overlap = self.forward(non_overlap_users, non_overlap_items)
            loss_non_overlap = self.loss(p_non_overlap, non_overlap_labels)
            
            # ===== Push Mechanism via DiverseSampler =====
            # Hard negatives are provided by DiverseSampler at DataLoader level
            # No extra sampling or loss computation needed here
            # DiverseSampler maintains 1:1 positive/negative ratio
            
            # Regularization
            reg_loss = self.target_reg_loss(
                self.get_user_embedding(non_overlap_users),
                self.get_item_embedding(non_overlap_items)
            )
            loss_non_overlap = loss_non_overlap + self.gamma * reg_loss
            
            total_loss = total_loss + loss_non_overlap * (non_overlap_indices.size(0) / batch_size)
        
        # If no mechanism is enabled or no users in categories, use standard CMF loss
        if isinstance(total_loss, torch.Tensor) and total_loss.item() == 0.0:
            p_target = self.forward(target_user, target_item)
            loss_t = self.loss(p_target, target_label)
            reg_loss = self.target_reg_loss(
                self.get_user_embedding(target_user),
                self.get_item_embedding(target_item)
            )
            loss_t = loss_t + self.gamma * reg_loss
            return loss_t
        elif not isinstance(total_loss, torch.Tensor) and total_loss == 0.0:
            p_target = self.forward(target_user, target_item)
            loss_t = self.loss(p_target, target_label)
            reg_loss = self.target_reg_loss(
                self.get_user_embedding(target_user),
                self.get_item_embedding(target_item)
            )
            loss_t = loss_t + self.gamma * reg_loss
            return loss_t
        
        return total_loss

    def calculate_map_loss(self, interaction):
        """Calculate loss for both domains (mapping phase)."""
        # Source domain loss (standard)
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        p_source = self.forward(source_user, source_item)
        loss_s = self.loss(p_source, source_label)
        reg_loss_s = self.source_reg_loss(
            self.get_user_embedding(source_user),
            self.get_item_embedding(source_item)
        )
        loss_s = loss_s + self.lamda * reg_loss_s
        
        # Target domain loss (with diversity mechanisms)
        loss_t = self.calculate_target_loss(interaction)
        
        return loss_s * self.alpha_cmf + loss_t * (1 - self.alpha_cmf)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        p = self.forward(user, item)
        return p

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight[:self.target_num_items]
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

