#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code script is to load the models.

@author: jhao2
"""

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
    
experiment_name = ["Amity_competition_lag0d_lt0d_pred1d", # Amity_competition_lag0d_lt0d_pred1d_daytime
                   "Donalsonville_competition_lag0d_lt0d_pred1d", # Donalsonville_competition_lag0d_lt0d_pred1d_daytime
                   "San_Antonio_competition_lag0d_lt0d_pred1d", # San_Antonio_competition_lag0d_lt0d_pred1d_daytime
                   "Waianae_competition_lag0d_lt0d_pred1d"]

output_path = 'competition_experiments13'
val_ratio = 0
strategy = ['','narrow','wide'] 
criteria = ["pinball_loss","pinball_regloss"]
idata_dir = "data12/"+experiment_name[0]
# iterate over different models w.r.t loss functions
# only demonstrate for Amity, OR locations.
for iloss in range(2):
    for istrategy in strategy:
        # change this to your model directory
        finetune_path = '/Users/jhao2/Desktop/Load Forecasting/Code/Code/MTS_Transformer/competition_experiments13/Amity_competition_lag0d_lt0d_pred1d_finetune_pinball_loss__2023-10-24_13-23-56_Gil/checkpoints/model_last.pth'
        config = config_settings(comment='predict test y', 
                                 epochs=0, 
                                 experiment_name= 'prediction_'+experiment_name[0]+'_'+criteria[iloss]+'_'+istrategy,  
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
          
        config['test_only'] = 'True'
        device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu') 
        
        # initialize data class for regression task 
        my_data = TSRegressionArchive(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
        
        # initilize model 
        model = model_factory(config, my_data)
     
        if config['load_model']:
            model = utils.load_model(model, config['load_model'])
        model.to(device)