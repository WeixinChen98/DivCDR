# @Time   : 2022/3/12
# @Author : zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole_cdr.quick_start
########################
"""
import logging
from logging import getLogger
import torch
import os

from recbole.utils import init_logger, init_seed, set_color, dict2str

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.utils import get_model, get_trainer
import warnings

def run_recbole_cdr(model=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    warnings.filterwarnings("ignore")

    config = CDRConfig(model=model, config_file_list=config_file_list, config_dict=config_dict)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # model loading and initialization (needed for FairSampler)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, dataset).to(config['device'])
    logger.info(model)
    
    # dataset splitting (with model for FairSampler if needed)
    train_data, valid_data, test_data = data_preparation(config, dataset, model)
    

    
    # set in fit function 
    # if valid_data.overlap_users is None:
    #     valid_data.overlap_users = train_data.overlap_users
    if test_data.overlap_users is None:
        test_data.overlap_users = train_data.overlap_users
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # Set FairSampler if available
    if hasattr(dataset, 'fair_sampler') and dataset.fair_sampler is not None:
        trainer.set_fair_sampler(dataset.fair_sampler)
    
    # Set up DiverseSampler if configured (DiverseSampler is now created in data_preparation)
    use_diverse_sampler = config['use_diverse_sampler'] if 'use_diverse_sampler' in config else False
    if use_diverse_sampler and config['model'].startswith('Diverse_'):
        # DiverseSampler is already created and attached to DataLoader in data_preparation
        # We just need to set overlapping users and update centroid
        if hasattr(model, 'set_overlapping_users') and hasattr(train_data, 'overlap_users'):
            model.set_overlapping_users(train_data.overlap_users)
            model.update_overlap_centroid()
            logger.info(f"DiverseSampler configured for {config['model']} with {len(train_data.overlap_users)} overlapping users")

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    if config['delete_checkpoint_after_eval']:
        os.remove(trainer.saved_model_file)

    for key, value in best_valid_result.items():
        logger.info(set_color(f'best valid {key}:', 'yellow') + f'\n{dict2str(value)}')
    
    for key, value in test_result.items():
        logger.info(set_color(f'test result {key}:', 'yellow') + f'\n{dict2str(value)}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    warnings.filterwarnings("ignore")

    config = CDRConfig(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    
    # Check if we need FairSampler (only initialize model early if needed)
    use_fair_sampler = config['use_fair_sampler'] if 'use_fair_sampler' in config else False
    model = None
    
    if use_fair_sampler:
        # model loading and initialization (needed for FairSampler)
        init_seed(config['seed'], config['reproducibility'])
        model = get_model(config['model'])(config, dataset).to(config['device'])
    else:
        # Initialize model before data preparation to ensure consistent seed
        init_seed(config['seed'], config['reproducibility'])
        model = get_model(config['model'])(config, dataset).to(config['device'])
    
    train_data, valid_data, test_data = data_preparation(config, dataset, model)
    
    if test_data.overlap_users is None:
        test_data.overlap_users = train_data.overlap_users
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # Set up DiverseSampler if configured (same as run_recbole_cdr)
    use_diverse_sampler = config['use_diverse_sampler'] if 'use_diverse_sampler' in config else False
    if use_diverse_sampler and config['model'].startswith('Diverse_'):
        if hasattr(model, 'set_overlapping_users') and hasattr(train_data, 'overlap_users'):
            model.set_overlapping_users(train_data.overlap_users)
            model.update_overlap_centroid()
    
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    
    if config['delete_checkpoint_after_eval']:
        os.remove(trainer.saved_model_file)


    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    
    # Check if we need FairSampler (only initialize model early if needed)
    use_fair_sampler = config['use_fair_sampler'] if 'use_fair_sampler' in config else False
    model = None
    
    if use_fair_sampler:
        # model loading and initialization (needed for FairSampler)
        init_seed(config['seed'], config['reproducibility'])
        model = get_model(config['model'])(config, dataset).to(config['device'])
    else:
        # Initialize model before data preparation to ensure consistent seed
        init_seed(config['seed'], config['reproducibility'])
        model = get_model(config['model'])(config, dataset).to(config['device'])
    
    train_data, valid_data, test_data = data_preparation(config, dataset, model)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
