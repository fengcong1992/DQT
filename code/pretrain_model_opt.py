"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import os
import sys
import time
import pickle
import json
import nni

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from running import  pipeline_factory, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer
import numpy as np

def validate(val_evaluator, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""
 
    # eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    # eval_runtime = time.time() - eval_start_time
 
    # global val_times
    # val_times["total_time"] += eval_runtime
    # val_times["count"] += 1
    # avg_val_time = val_times["total_time"] / val_times["count"]
    # avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    # avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
     
    # print_str = 'Epoch {} Validation Summary: '.format(epoch)
    # for k, v in aggr_metrics.items(): 
    #     print_str += '{}: {:8f} | '.format(k, v)
   
    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']] 
        best_metrics = aggr_metrics.copy()

        # pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        # np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value

def main(config,arg_t,fold):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
   

    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features
    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    # Split dataset
    test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    if config['test_pattern']:  # used if test data come from different files / file patterns
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
        print('test_pattern')
    if config['test_from']:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
        test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
        print('test_from')
        try:
            test_indices = [int(ind) for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
    
    # split dataset
    N_train = len(my_data.all_IDs)
    N_valid = int(N_train/5)
    train_indices = np.array(my_data.all_IDs)
    
    train_indices = np.roll(my_data.all_IDs,-fold*N_valid,axis=0)
    val_indices = train_indices[0:N_valid]
    train_indices = train_indices[N_valid:-1]
    test_indices = [] 
    
    # if config['val_pattern']:  # used if val data come from different files / file patterns
    #     print('val_pattern')
    #     val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
    #     val_indices = val_data.all_IDs 

    # # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    # if config['val_ratio'] > 0: 
    #     train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
    #                                                               validation_method=validation_method,
    #                                                               n_splits=1,
    #                                                               validation_ratio=config['val_ratio'],
    #                                                               test_set_ratio=config['test_ratio'],  # used only if test_indices not explicitly specified
    #                                                               test_indices=test_indices,
    #                                                               random_seed=1337,
    #                                                               labels=labels) 
    #     train_indices = train_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
    #     val_indices = val_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
    # else:
    #     train_indices = my_data.all_IDs
    #     if test_indices is None:
    #         test_indices = []
            
    print('test_indices:\n',test_indices)
    # with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
    #     try:
    #         json.dump({'train_indices': list(map(int, train_indices)),
    #                    'val_indices': list(map(int, val_indices)),
    #                    'test_indices': list(map(int, test_indices))}, f, indent=4)
    #     except ValueError:  # in case indices are non-integers
    #         json.dump({'train_indices': list(train_indices),
    #                    'val_indices': list(val_indices),
    #                    'test_indices': list(test_indices)}, f, indent=4)

    # Create model
    model = model_factory(config, my_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Initialize optimizer 
    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if arg_t.load_model:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    print('model max length:',model.max_len)
    model.to(device) 
    
    loss_module = get_loss_module(config)
  
    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, val_indices)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    train_dataset = dataset_class(my_data, train_indices)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

    # tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
     
    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    val_loss = []
    stop_flag = 0
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last' 
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics 
         
        # for k, v in aggr_metrics_train.items():
        #     tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch) 
       

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values)) 
            
            # report validation accuracy to BOHB web portal
            nni.report_intermediate_result(aggr_metrics_val['loss'])
            
            # converge measurement 
            if epoch == 1:
                val_loss.append(aggr_metrics_val['loss'])
            else: 
                if aggr_metrics_val['loss'] >= val_loss[-1]:
                    stop_flag+=1
                else:
                    val_loss.append(aggr_metrics_val['loss'])
                    stop_flag = 0
                if stop_flag >=10:
                    break  
                 
        # utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                lr_step += 1
           
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update() 
      
    return best_value
 
if __name__ == '__main__':  
    # hyperparameters to be optimized
    params = {
        # 'optimizer': 'Adam',
        'lr': 0.1,
        'batch_size':64,
        'l2_reg':0,
        'global_reg':True,
        'd_model':128,
        'dim_feedforward':256,
        'num_heads':8,
        'num_layers':3,
        'dropout':0.1,
        # 'pos_encoding':'learnable',
        # 'activation':'gelu',
        # 'normalization_layer':'BatchNorm'
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
  
    # common configurations 
# =============================================================================
#     # task = pretrain
# ============================================================================= 
    args = Options().parse()  # `argsparse` object}
    args.comment='optimization of pretraining model'
    args.experiment_name = 'opt_MrktPrice_train4test1_LA_repeated_sincos_enc_uniformLog'
    args.records_file="Imputation_records.xls"   
    args.data_dir='data/MrktPrice_train4test1_LA_repeated_sincos_enc_uniformLog/'
    args.output_dir='experiments'
    args.print_interval = 0
    args.data_class = 'tsra'
    args.pattern = 'TRAIN'
    args.val_ratio = 0
    args.epochs = 600
    args.task="imputation"
    args.val_pattern = ""
    
    args.optimizer = 'Adam'
    args.lr = params['lr']
    args.batch_size = params['batch_size']
    args.l2_reg = params['l2_reg']
    args.global_reg = params['global_reg']
    args.d_model = params['d_model']
    args.dim_feedforward = params['dim_feedforward']
    args.num_heads = params['num_heads']
    args.num_layers = params['num_layers']
    args.dropout = params['dropout']
    args.pos_encoding = 'fixed'
    args.activation = 'relu'
    args.normalization_layer = 'LayerNorm'
    
    config = args.__dict__
    # config = setup(args)  
    print('pretrain:\n',config)
    validation_loss = []
    for fold in range(5):
        best_value = main(config,args,fold)
        validation_loss.append(best_value) 
    # report final testing loss to BOHB web portal
    nni.report_final_result(np.mean(validation_loss)) 
