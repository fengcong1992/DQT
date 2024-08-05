"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

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
# from models.ts_transformer_v2 import model_factory # Jie's original line, there is no ts_transformer_v2
from models.ts_transformer import model_factory # I change it to this line, there is ts_transformer and same with content in PPT
from models.loss import get_loss_module
from optimizers import get_optimizer
import numpy as np
import scipy.io as sio
from models.measurements import measurements 
import pandas as pd
import csv

def produce_testy(dataloader,config, model, device): 
    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    t = []
    p = []
    testLoss = {'pinball_regloss':[],'pinball_loss':[],'CRPS_loss':[]}
    with torch.no_grad():
        model = model.eval()  
        per_batch = {'target_masks': [], 'targets': [], 'predictions': []}
        for i, batch in enumerate(dataloader): 
            X, targets, padding_masks, IDs = batch 
            padding_masks = padding_masks.to(device)  # 0s: ignore
            # regression: (batch_size, prediction time steps)
            predictions = model(X.to(device), padding_masks) 
            per_batch['targets'].append(targets.cpu().numpy())
            per_batch['predictions'].append(predictions.cpu().numpy()) 
            
            if i==0:
                t = targets.detach().cpu().numpy()
                p = predictions.detach().cpu().numpy()
            else:
                t = np.concatenate((t,targets.detach().cpu().numpy()),axis = 0)
                p = np.concatenate((p,predictions.detach().cpu().numpy()),axis = 0)
    pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
    # np.savez(pred_filepath, **per_batch)
    # if config['test_only']:
    loss_functions = measurements(p,t,config['reg_weight'])
    # testLoss['pinball_regloss_CityDayTime'].append(loss_functions.pinball_regloss_CityDayTime())
    testLoss['pinball_regloss'].append(loss_functions.pinball_regloss())
    testLoss['pinball_loss'].append(loss_functions.pinball_loss())
    # testLoss['CRPS_cityloss'].append(loss_functions.CRPS_cityloss()) 
    testLoss['CRPS_loss'].append(loss_functions.CRPS_loss())
    df = pd.DataFrame(testLoss)
    df.to_csv(f"{config['output_dir']}/metrics_{config['experiment_name']}.csv")
    # with pd.ExcelWriter(os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xlsx")) as ew: 
    #     df = pd.DataFrame(testLoss)
    #     df.to_excel(ew)
    
    # sio.savemat(pred_filepath+".mat",{"targets":t, "predictions":np.reshape(p,(-1,11))}) 
    np.savez(pred_filepath+".npz",targets=t, predictions=p) 
    
def predict(config):

    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu') 
    
    # initialize data class for regression task 
    my_data = TSRegressionArchive(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    
    # load model 
    model = model_factory(config, my_data)
 
    if config['load_model']:
        model = utils.load_model(model, config['load_model'])
    model.to(device) 
     
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(my_data, my_data.all_IDs)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
    
    produce_testy(val_loader,config, model, device) 


if __name__ == '__main__':  
    output_path = 'experiments'
    # city = ['Amity', 'Donalsonville','SanAntonio','Waianae']  
    # path_keywords = [icity+'_3day_to_1day_finetune_May_finetune_400epochs__pinball_regloss_'
    #                   for icity in city] ## experiment folder names of models to be loaded 
    folders = os.listdir(os.getcwd()+'/experiments')
    # n = len(folders)
    # paths = []
    # for i in range(n): 
    #     for j in range(4):
    #         if path_keywords[j] in folders[i]:
    #             paths.append(folders[i])
    
    data_folder = ["NL_lag_7d_to_1d_lt24","PVLoadNL_lag_7d_to_1d_lt24","HECO_and_NSRDB_lag_7d_to_1d_lt24"]
    paths = ["NL_lag_7d_to_1d_lt24_finetune_regloss_001_099_2023-07-25_02-37-33_4JY",\
             "PVLoadNL_lag_7d_to_1d_lt24_finetune_regloss_001_099_2023-07-25_02-38-08_lXy",\
             "HECO_and_NSRDB_lag_7d_to_1d_lt24_finetune_regloss_001_099_2023-07-25_02-38-44_nDj"]
    for i in range(3):
        # path_kwd = 'HECO_nlag24_lt24_PVLoadNL_finetune_50epochs_pinball_regloss_regweight_'+str(iweight)
        # path = [ifolder for ifolder in folders if path_kwd in ifolder]
        path=paths
        lossfunc = 'pinball_regloss'
        # paths = ['Amity_finetune_May_finetune__pinballregloss_2023-06-21_01-00-22_FLi',
        #         'Donalsonville_finetune_May_finetune__pinballregloss_2023-06-21_00-58-08_jWK',
        #         'SanAntonio_finetune_May_finetune__pinballregloss_2023-06-21_00-58-49_cVK',
        #         'Waianae_finetune_May_finetune__pinballregloss_2023-06-21_00-59-18_qt4']
        # for i in range(3,4):
        data_dir = 'data/'+ data_folder[i]
        print("data folder: ",data_dir)
        model_path = output_path+"\\"+path[i]
        model_path+="\\checkpoints\\model_last.pth" 
        
        args = Options().parse()
        args.data_dir = data_dir
        args.output_dir= output_path
        args.comment = "predict test y"
        args.experiment_name = 'prediction_'+lossfunc+'_'+data_folder[i]+'_perc_001_099'
        args.records_file = "Regression_records.xls"
        # to load data ending with TEST
        args.val_pattern = "TEST"
        args.load_model = model_path
        args.pattern = "TEST" 
        args.task = "regression" 
        args.change_output = False
        
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.batch_size = 32
        args.l2_reg = 0
        args.global_reg = False
        args.d_model = 128
        args.dim_feedforward = 128
        args.num_heads = 4
        args.num_layers = 4
        args.dropout = 0.3
        args.pos_encoding = 'fixed'
        args.activation = 'relu'
        args.normalization_layer = 'LayerNorm'
        args.epochs = 0
        # args.city = city
        args.test_only = 'True'
        args.reg_weight = 7
        config = setup(args)
        
        predict(config,args)