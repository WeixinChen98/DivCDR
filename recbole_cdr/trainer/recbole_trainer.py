r"""
recbole.trainer.trainer
################################
"""

import os
import sys
from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

from recbole.data.interaction import Interaction
from recbole_cdr.data.dataloader import FullSortEvalDataLoader
from recbole_cdr.evaluator.evaluator import Evaluator
from recbole_cdr.evaluator.collector import Collector
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage, WandbLogger


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        


        self.eval_type = config['eval_type']
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_tensor = None
        self.tot_item_num = None

        # Step 0: separate parameter groups
        self.generator_params = []
        self.baseline_params = []
        for name, param in model.named_parameters():
            if 'source_generator' in name:  # or 'W_q', 'W_k', 'W_v', etc.
                self.generator_params.append(param)
                # self.baseline_params.append(param)
            else:
                self.baseline_params.append(param)
        

        # self.optimizer = self._build_optimizer()
        self.optimizer = self._build_optimizer(params=self.baseline_params)
        if self.generator_params:
            self.gen_optimizer = self._build_optimizer(params=self.generator_params, learning_rate=0.001)
            # for CMF with FCDR on Amazons

        self.user_groups = {}  # Will store user_id -> sensitive_attribute mapping for evaluation analysis
        # User attributes will be loaded after dataset is available
        self._user_groups_ready_callbacks = []
        
        # Set trainer reference in model if it has the method
        if hasattr(self.model, 'set_trainer'):
            self.model.set_trainer(self)
        
        # Initialize FairSampler reference
        self.fair_sampler = None
        # Initialize DiverseSampler reference
        self.diverse_sampler = None

    def set_fair_sampler(self, fair_sampler):
        """Set FairSampler for fairness-aware negative sampling"""
        self.fair_sampler = fair_sampler
        if hasattr(self.model, 'set_fair_sampler'):
            self.model.set_fair_sampler(fair_sampler)
    
    def set_diverse_sampler(self, diverse_sampler):
        """Set DiverseSampler for diversity-aware negative sampling (Push mechanism)"""
        self.diverse_sampler = diverse_sampler
        if hasattr(self.model, 'set_diverse_sampler'):
            self.model.set_diverse_sampler(diverse_sampler)

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config['reg_weight'] and weight_decay and weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        
        # Update overlap centroid at the start of each epoch (for Push mechanism)
        if hasattr(self.model, 'update_overlap_centroid'):
            self.model.update_overlap_centroid()
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
                disable=not sys.stdout.isatty(),  # 在重定向到文件时自动禁用进度条
                mininterval=1.0,  # 最小更新间隔1秒，减少输出频率
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)

            # -------------------------
            # 1) TRAIN BASELINE PARAMS
            # -------------------------
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            # print("loss: ", loss)
            self._check_nan(loss)
            # loss.backward(retain_graph=True)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.baseline_params, **self.clip_grad_norm)
            self.optimizer.step()



            # --------------------------------
            # 2) TRAIN GENERATOR
            # --------------------------------
            if self.model.phase in ['BOTH', 'OVERLAP'] and self.generator_params:
                self.gen_optimizer.zero_grad()
                generator_loss = self.model.calculate_gen_loss(interaction)  # separate forward pass 
                self._check_nan(generator_loss)
                generator_loss.backward()
                if self.clip_grad_norm:
                    clip_grad_norm_(self.generator_params, **self.clip_grad_norm)
                self.gen_optimizer.step()


            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        # valid_score = {}
        # valid_score['overall'] = calculate_valid_score(valid_result['overall'], self.valid_metric)
        # valid_score['group1'] = calculate_valid_score(valid_result['group1'], self.valid_metric)
        # valid_score['group2'] = calculate_valid_score(valid_result['group2'], self.valid_metric)
        valid_score = calculate_valid_score(valid_result['overall'], self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        saved_model_file = kwargs.pop('saved_model_file', self.saved_model_file)
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)
        if verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {saved_model_file}')

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.load_other_parameter(checkpoint.get('other_parameter'))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learner': self.config['learner'],
            'learning_rate': self.config['learning_rate'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        
        # Load user attributes after data collection
        if hasattr(train_data, 'dataset'):
            self.load_user_attributes(self.config, train_data.dataset)
        else:
            self.load_user_attributes(self.config)
        
        # Set overlapping users for diversity-aware models (e.g., Diverse_CMF)
        if hasattr(train_data, 'overlap_users') and train_data.overlap_users is not None:
            if hasattr(self.model, 'set_overlapping_users'):
                self.model.set_overlapping_users(train_data.overlap_users)
                self.logger.info(f"Set {len(train_data.overlap_users)} overlapping users for model")
            
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step':epoch_idx}, head='train')

            # Call model's on_epoch_end if available
            if hasattr(self.model, 'on_epoch_end'):
                self.model.on_epoch_end(epoch_idx)
            
            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result (overall)', 'blue') + ': \n' + dict2str(valid_result['overall'])
                valid_result_output += '\n' + set_color('valid result (group1)', 'blue') + ': \n' + dict2str(valid_result['group1'])
                valid_result_output += '\n' + set_color('valid result (group2)', 'blue') + ': \n' + dict2str(valid_result['group2'])
                valid_result_output += '\n' + set_color('valid result (ugf)', 'blue') + ': \n' + dict2str(valid_result['ugf'])
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step+=1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def load_user_attributes(self, config, dataset=None):
        """Load user attributes - identify overlapping vs non-overlapping users from CrossDomainDataset."""
        
        try:
            if dataset is None:
                self.logger.warning("Dataset not provided, cannot identify overlapping users")
                self.user_groups = {}
                return
            
            # CrossDomainDataset已经区分了overlapping和non-overlapping用户
            # overlap_users: ID范围是 [1, num_overlap_user)
            # non_overlap_users: ID范围是 [num_overlap_user, num_total_user)
            # Group 1: overlapping users (exist in both source and target domains)
            # Group 2: non-overlapping users (only in target domain)
            
            if hasattr(dataset, 'overlap_users') and hasattr(dataset, 'non_overlap_users'):
                # 直接使用dataset中已经识别好的用户集合
                overlapping_users = dataset.overlap_users
                non_overlapping_users = dataset.non_overlap_users
                
                # 分配用户组
                for user_id in overlapping_users:
                    self.user_groups[int(user_id)] = 1  # Group 1: overlapping
                
                for user_id in non_overlapping_users:
                    self.user_groups[int(user_id)] = 2  # Group 2: non-overlapping
                
                # 验证分组结果
                self.logger.info(f"User groups identified from CrossDomainDataset:")
                self.logger.info(f"  Group 1 (overlapping users): {len(overlapping_users)} users")
                self.logger.info(f"  Group 2 (non-overlapping users): {len(non_overlapping_users)} users")
                
                # Notify listeners that user_groups is now ready
                for cb in list(self._user_groups_ready_callbacks):
                    try:
                        cb()
                    except Exception:
                        pass
            else:
                self.logger.warning("overlap_users or non_overlap_users not found in dataset")
                self.user_groups = {}
                
        except Exception as e:
            self.logger.warning(f"Failed to identify overlapping users: {str(e)}")
            self.user_groups = {}

    def register_user_groups_ready_callback(self, callback):
        """Register a callback to be invoked once user_groups have been loaded."""
        if callable(callback):
            self._user_groups_ready_callbacks.append(callback)

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf

        # Extract user IDs from interaction
        user_ids = list(interaction.numpy().values())[0]

        # 添加调试信息（只在第一次时打印）
        if not hasattr(self, '_debug_logged'):
            self.logger.info(f"Total users in batch: {len(user_ids)}")
            self.logger.info(f"User groups loaded: {len(self.user_groups)}")
            self.logger.info(f"Sample user groups: {dict(list(self.user_groups.items())[:5])}")
            self.logger.info(f"Sample user_ids from interaction: {user_ids[:5]}")
            self.logger.info(f"User_ids type: {type(user_ids[0]) if len(user_ids) > 0 else 'empty'}")
            self.logger.info(f"User_groups keys type: {type(list(self.user_groups.keys())[0]) if self.user_groups else 'empty'}")
            self._debug_logged = True

        # Separate indices for different demographic groups
        group1_indices = [idx for idx, uid in enumerate(user_ids) if self.user_groups.get(int(uid), 0) == 1]
        group2_indices = [idx for idx, uid in enumerate(user_ids) if self.user_groups.get(int(uid), 0) == 2]
        
        # 添加分组统计（只在第一次时打印）
        if not hasattr(self, '_group_stats_logged'):
            self.logger.info(f"Group 1 users: {len(group1_indices)}, Group 2 users: {len(group2_indices)}")
            # 检查前几个用户ID的匹配情况
            for i, uid in enumerate(user_ids[:5]):
                group_value = self.user_groups.get(int(uid), 0)
                self.logger.info(f"User {uid} (type: {type(uid)}) -> group: {group_value} (type: {type(group_value)})")
            
            # 检查用户ID范围
            self.logger.info(f"User IDs range: min={min(user_ids)}, max={max(user_ids)}")
            self.logger.info(f"User groups range: min={min(self.user_groups.keys())}, max={max(self.user_groups.keys())}")
            
            # 检查是否有重叠
            user_ids_set = set(int(uid) for uid in user_ids)
            groups_keys_set = set(self.user_groups.keys())
            overlap = user_ids_set.intersection(groups_keys_set)
            self.logger.info(f"Overlap between user_ids and user_groups: {len(overlap)} users")
            if overlap:
                self.logger.info(f"Sample overlapping users: {list(overlap)[:5]}")
            self._group_stats_logged = True

        # Extract scores for each group
        group1_scores = scores[group1_indices] if group1_indices else torch.empty((0, scores.shape[1]), device=scores.device)
        group2_scores = scores[group2_indices] if group2_indices else torch.empty((0, scores.shape[1]), device=scores.device)

        # Extract corresponding user IDs for each group
        group1_user_ids = [user_id for user_id in user_ids if self.user_groups.get(int(user_id), 0) == 1]
        group2_user_ids = [user_id for user_id in user_ids if self.user_groups.get(int(user_id), 0) == 2]

        # Get positive items for each group
        group1_pos_items = self.uid2positive_item[group1_user_ids]
        group2_pos_items = self.uid2positive_item[group2_user_ids]

        # Handle positive items for group 1
        if len(group1_pos_items) > 0:
            group1_pos_items_non_none = [p if p is not None else torch.tensor([], dtype=torch.long) for p in group1_pos_items]
            group1_positive_u = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(group1_pos_items_non_none)])
            group1_positive_i = torch.cat(group1_pos_items_non_none)
        else:
            group1_positive_u = torch.empty((0,), dtype=torch.long, device=scores.device)
            group1_positive_i = torch.empty((0,), dtype=torch.long, device=scores.device)

        # Handle positive items for group 2
        if len(group2_pos_items) > 0:
            group2_pos_items_non_none = [p if p is not None else torch.tensor([], dtype=torch.long) for p in group2_pos_items]
            group2_positive_u = torch.cat([torch.full_like(pos_iid, i) for i, pos_iid in enumerate(group2_pos_items_non_none)])
            group2_positive_i = torch.cat(group2_pos_items_non_none)
        else:
            group2_positive_u = torch.empty((0,), dtype=torch.long, device=scores.device)
            group2_positive_i = torch.empty((0,), dtype=torch.long, device=scores.device)

        # Ensure all tensors are on the correct device
        group1_positive_u = group1_positive_u.to(scores.device)
        group1_positive_i = group1_positive_i.to(scores.device)
        group2_positive_u = group2_positive_u.to(scores.device)
        group2_positive_i = group2_positive_i.to(scores.device)

        return (interaction, scores, positive_u, positive_i,
                group1_scores, group1_positive_u, group1_positive_i,
                group2_scores, group2_positive_u, group2_positive_i)

    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config['eval_type'] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config['eval_type'] == EvaluatorType.RANKING:
            col_idx = interaction[self.config['ITEM_ID_FIELD']]
            batch_user_num = positive_u[-1] + 1
            scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
            scores[row_idx, col_idx] = origin_scores
            return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        """Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether to load the best model in the training process. Defaults to True.
            model_file (str, optional): The saved model file. If provided, it loads the trained model.
            show_progress (bool): Show the progress of the evaluation epoch. Defaults to False.

        Returns:
            collections.OrderedDict: Evaluation results containing metrics for all users, group1 users, and group2 users.
        """
        if not eval_data:
            return

        self.overlap_users = set(eval_data.overlap_users)
        self.uid2positive_item = eval_data.uid2positive_item
        

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
                disable=not sys.stdout.isatty(),  # 在重定向到文件时自动禁用进度条
                mininterval=1.0,  # 最小更新间隔1秒，减少输出频率
            ) if show_progress else eval_data
        )


        for batch_idx, batched_data in enumerate(iter_data):
            (interaction, scores, positive_u, positive_i,
             group1_scores, group1_positive_u, group1_positive_i,
             group2_scores, group2_positive_u, group2_positive_i) = eval_func(batched_data)

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i, group=None)
            if group1_scores.numel() > 0:
                self.eval_collector.eval_batch_collect(group1_scores, interaction, group1_positive_u, group1_positive_i, group="group1")
            # else: 
            #     self.logger.debug("group1_scores is empty")  # 正常现象，某些batch没有该组用户

            if group2_scores.numel() > 0:
                self.eval_collector.eval_batch_collect(group2_scores, interaction, group2_positive_u, group2_positive_i, group="group2")
            # else:
            #     self.logger.debug("group2_scores is empty")  # 正常现象，某些batch没有该组用户
                
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        # 添加user_groups信息到struct，用于metrics的分组计算
        if hasattr(self, 'user_groups') and self.user_groups:
            struct.set('data.user_groups', self.user_groups)
        result = self.evaluator.evaluate(struct)

        self.wandblogger.log_eval_metrics(result['overall'], head='eval overall')
        self.wandblogger.log_eval_metrics(result['group1'], head='eval group1')
        self.wandblogger.log_eval_metrics(result['group2'], head='eval group2')
        self.wandblogger.log_eval_metrics(result['ugf'], head='eval ugf')

        return result

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(Interaction(current_interaction).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)
