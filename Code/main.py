import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import TSRegressionArchive
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer
from load_pred import predict
# import numpy as np
import pandas as pd 
import numpy as np
import scipy.io as sio
import json

def main(config):

    total_epoch_time = 0
    total_eval_time = 0

    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...") 
    
    # load data 
    my_data = TSRegressionArchive(config['data_dir'], pattern=config['pattern'], 
                                  n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
   
    validation_method = 'ShuffleSplit'
    labels = None

    val_data = my_data
    val_indices = []
 
    # option 1 select validation file ending with config['val_pattern']
    if config['val_pattern']:  # used if val data come from different files / file patterns
        val_data = TSRegressionArchive(config['data_dir'], pattern = config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs

    # option 2 select valication data from training data
    if config['val_ratio'] > 0:
        train_indices, val_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                  validation_method=validation_method,
                                                                  n_splits=1,
                                                                  validation_ratio=config['val_ratio'],
                                                                  random_seed=1337,
                                                                  labels=labels)
        train_indices = train_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
        val_indices = val_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
    else:
        train_indices = my_data.all_IDs

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    # logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices)}, f, indent=4)

    # Create model
    logger.info("Creating model ...")
    model = model_factory(config, my_data)
    
    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False
    

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))


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
    if config['load_model']:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])

    model.to(device) 
    
    loss_module = get_loss_module(config)
  
    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config) 
    train_dataset = dataset_class(my_data, train_indices) 
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len)) 
    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                                 print_interval=config['print_interval'], console=config['console']) 
    if len(val_indices)>0:
        val_dataset = dataset_class(my_data, val_indices)
        val_loader = DataLoader(dataset = val_dataset,
                                batch_size=config['batch_size'],
                                shuffle=False,
                                num_workers = config['num_workers'],
                                pin_memory=True,
                                collate_fn = lambda x: collate_fn(x, max_len=model.max_len))
        val_evaluator = runner_class(model, val_loader, device, loss_module, print_interval=config['print_interval'],
                                     console = config['console'])
        
    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}
     
    if config['task'] == "imputation" :
        if len(val_indices)>0:
            Loss = {'epoch':[],'train_loss':[],'val_loss':[]}
        else:
            Loss = {'epoch':[],'train_loss':[]}
    else:
        if len(val_indices)>0:
            Loss = {'epoch':[],'train_pinball_regloss':[],'train_pinball':[],'train_CRPS':[],
                     'val_pinball_regloss':[],'val_pinball':[],'val_CRPS':[]}
        else:
            Loss = {'epoch':[],'train_pinball_regloss':[],'train_pinball':[],'train_CRPS':[]} 
        
    logger.info('Starting training...')
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False): 
        epoch_start_time = time.time() 
        aggr_metrics_train = trainer.train_epoch(config,epoch)  # dictionary of aggregate epoch metric 
        if config['task'] == "imputation" :
            Loss['epoch'].append(epoch)
            Loss['train_loss'].append(aggr_metrics_train['loss'])
            print('training loss:',Loss['train_loss'][-1])
        else:
            print('task is finetune-------------------------------------------------------')
            Loss['epoch'].append(epoch)
            # Loss['train_pinball_regloss_CityDayTime'].append(aggr_metrics_train['pinball_regloss_CityDayTime'])
            Loss['train_pinball_regloss'].append(aggr_metrics_train['pinball_regloss'])
            Loss['train_pinball'].append(aggr_metrics_train['pinball_loss'])
            # Loss['train_CRPS_cityloss'].append(aggr_metrics_train['CRPS_cityloss'])
            Loss['train_CRPS'].append(aggr_metrics_train['CRPS_loss'])
             
        epoch_runtime = time.time() - epoch_start_time  
            
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        
        logger.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time))) 
        
        if len(val_indices)>0: 
            # evaluate if first or last epoch or at specified interval
            if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
                aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                      best_metrics, best_value, epoch)
                metrics_names, metrics_values = zip(*aggr_metrics_val.items())
                metrics.append(list(metrics_values))
                if config['task'] == 'imputation': 
                    Loss['val_loss'].append(aggr_metrics_val['loss'])  
                    print('validation loss:',Loss['val_loss'][-1])
                else: 
                    # Loss['val_pinball_regloss_CityDayTime'].append(aggr_metrics_val['pinball_regloss_CityDayTime'])
                    Loss['val_pinball_regloss'].append(aggr_metrics_val['pinball_regloss'])
                    Loss['val_pinball'].append(aggr_metrics_val['pinball_loss'])
                    # Loss['val_CRPS_cityloss'].append(aggr_metrics_val['CRPS_cityloss'])
                    Loss['val_CRPS'].append(aggr_metrics_val['CRPS_loss']) 
        if config['save_all'] and epoch%10==0: 
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
        if epoch==(config["epochs"]):
            utils.save_model(os.path.join(config['save_dir'], 'model_last.pth'), epoch, model, optimizer)
        
        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
    
    with pd.ExcelWriter(os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xlsx")) as ew: 
        df = pd.DataFrame(Loss)
        df.to_excel(ew)
   
    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!') 
    
    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime))) 
    return best_value

def config_settings(comment,epochs,experiment_name,data_dir,output_path,pattern,
                    val_ratio,task,val_pattern,finetuneloss='pinball_regloss', 
                    strategy = '', reg_weight=7, model_path = ''):
    
    # pretrain_model_path = config['save_dir']+"\\model_last.pth"
    args = Options().parse()  # `argsparse` object}
    args.comment= comment
    args.epochs = epochs 
    args.experiment_name = experiment_name 
    args.data_dir = data_dir
    args.output_dir= output_path 
    args.pattern = pattern
    args.val_ratio = val_ratio  
    args.val_pattern = val_pattern
    
    args.optimizer = 'RAdam'
    args.lr = 0.001
    args.batch_size = 32
    args.l2_reg = 0
    args.global_reg = False
    args.d_model = 128
    args.dim_feedforward = 128
    args.num_heads = 4
    args.num_layers = 4 
    args.pos_encoding = 'fixed'
    args.activation = 'relu'
    args.normalization_layer = 'LayerNorm' 
    args.save_all = True
    
    if task=="imputation":
        args.records_file="Imputation_records.xls"
        args.task=task
        args.dropout = 0.5
    if task=="regression":
        args.records_file="Regression_records.xls"
        args.task=task
        args.dropout = 0.3
        args.finetuneloss = finetuneloss
        args.load_model = model_path
        args.reg_weight = reg_weight
        args.change_output = True
        args.freeze = True
        args.strategy = strategy
        
    config = setup(args)  # configuration dictionary
       
    return config
    
    
if __name__ == '__main__':   
    # =============================================================================
    #      training mode and paths
    # ============================================================================= 
    finetune_only = False 
    val_ratio = 0
    output_path = 'competition_experiments' ## folder to save experiments 
    # experiment_name = ["NL_lag_6d_to_1d_lt24","PVLoadNL_lag_6d_to_1d_lt24","HECO_and_NSRDB_lag_6d_to_1d_lt24",
                       #  "PVLoadNL_lag_1d_lt24","HECO_and_NSRDB_lag_1d_lt24",
                       # "PVLoadNL_lag_6d_1d_lt24","HECO_and_NSRDB_lag_6d_1d_lt24",
                       # "PVLoadNL_lag_6d_to_1d_lt24","HECO_and_NSRDB_lag_6d_to_1d_lt24"]
    experiment_name = ["Amity_competition_lag0d_lt0d_pred1d",
                       "Donalsonville_competition_lag0d_lt0d_pred1d",
                       "San_Antonio_competition_lag0d_lt0d_pred1d",
                       "Waianae_competition_lag0d_lt0d_pred1d"]
   
    if finetune_only:
        model_folder = ["Donalsonville_competition_lag0d_lt0d_pred1d_daytime_pretrain_2023-08-17_01-31-18_mNS"]
    
    # =============================================================================
    #     loop over datasets
    # ============================================================================= 
    strategy = ['','narrow','wide'] 
    criteria = ["pinball_loss","pinball_regloss"]
    
    for i in range(4):
        idata_dir = "data/"+experiment_name[i]
        pretrain_model_path = ''
        finetune_path = ''
        print("read data folder:",idata_dir)
        if (not finetune_only):
            print('Satrt pretrain------------------')
        # # # # =============================================================================
        # # # #     # task = pretrain
        # # # # # =============================================================================   
            config1 = config_settings(comment='pretraining through imputation', 
                                     epochs=200, 
                                     experiment_name=experiment_name[i] +'_pretrain',  
                                     data_dir=idata_dir, 
                                     output_path=output_path, 
                                     pattern='TRAIN', 
                                     val_ratio=val_ratio, 
                                     task="imputation", 
                                     val_pattern='')
            pretrain_model_path = config1['save_dir']+"/model_last.pth" 
            main(config1)
            
        # =============================================================================
        #   # task = finetune
        # ============================================================================= 
        if finetune_only:  
            pretrain_model_path = output_path+'/'+model_folder[0]+"/checkpoints/model_last.pth"  
        # print("pretrain_model path: \n", pretrain_model_path)
        print('Satrt finetune------------------')
        for iloss in range(2):
            for istrategy in strategy:
                config2 = config_settings(comment='finetune for regression', 
                                         epochs=50, 
                                         experiment_name=experiment_name[i] +'_finetune_'+criteria[iloss]+'_'+istrategy,  
                                         data_dir=idata_dir, 
                                         output_path=output_path, 
                                         pattern='TRAIN', 
                                         val_ratio=val_ratio, 
                                         task="regression", 
                                         val_pattern='',
                                          # finetuneloss = 'pinball_regloss',
                                         finetuneloss = criteria[iloss],
                                         strategy = istrategy,
                                         reg_weight = 7,
                                         model_path = pretrain_model_path)
                finetune_path = config2['save_dir']+"/model_last.pth" 
                main(config2)
                
                # =============================================================================
                #   # task = prediction
                # ============================================================================= 
                config3 = config_settings(comment='predict test y', 
                                         epochs=0, 
                                         experiment_name= 'prediction_'+experiment_name[i]+'_'+criteria[iloss]+'_'+istrategy,  
                                         data_dir=idata_dir, 
                                         output_path=output_path, 
                                         pattern='TEST', 
                                         val_ratio= val_ratio, 
                                         task="regression", 
                                         val_pattern='TEST',
                                          # finetuneloss = 'pinball_regloss',
                                         finetuneloss = criteria[iloss],
                                         strategy = istrategy,
                                         reg_weight = 7,
                                         model_path = finetune_path)
                  
                config3['test_only'] = 'True'
                predict(config3)
            
