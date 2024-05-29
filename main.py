from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali, test
from utils.config import event_root_code_list
from tqdm import tqdm
from models.GPT4MTS import GPT4MTS

import pdb

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import pickle
import json

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4MTS')

parser.add_argument('--model_id', type=str, default='Informer')
parser.add_argument('--checkpoints', type=str, default='./checkpoints_new/')
parser.add_argument('--res_dir', type=str, default='results/')

parser.add_argument('--root_path', type=str, default='dataset/llmts/')
parser.add_argument('--data_dir', type=str, default='modified_tts_us')
parser.add_argument('--data', type=str, default='llmts')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=0) # for what
parser.add_argument('--target', type=str, default='NumMentions')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--num_nodes', type=int, default=1)

parser.add_argument('--seq_len', type=int, default=15)
parser.add_argument('--pred_len', type=int, default=7)
parser.add_argument('--label_len', type=int, default=7)

parser.add_argument('--decay_fac', type=float, default=0.9)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type3') 
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--enc_in', type=int, default=3)
parser.add_argument('--c_out', type=int, default=3)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='Autoformer')
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)

parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--revin', type=bool, default=False)
parser.add_argument('--channel_independent', type=bool, default=False)

args = parser.parse_args()


SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []

for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'
    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]


    mse_res_file_path = os.path.join(args.res_dir, args.data_dir, f"{args.model}", f"{args.model}_{ii}_mse.json")
    mae_res_file_path = os.path.join(args.res_dir, args.data_dir, f"{args.model}", f"{args.model}_{ii}_mae.json")
    if not os.path.exists(os.path.join(args.res_dir, args.data_dir, f"{args.model}")):
        os.makedirs(os.path.join(args.res_dir, args.data_dir, f"{args.model}"))
    mse_res_dict = {}
    mae_res_dict = {}

    channel_independent_str = "indep" if args.channel_independent else "dep"
    train_datas_pickle_path = os.path.join("pickles", f"{args.data_dir}_batch-{args.batch_size}", f"train_{args.data_dir}_{channel_independent_str}_datas.pkl")
    train_loaders_pickle_path = os.path.join("pickles", f"{args.data_dir}_batch-{args.batch_size}", f"train_{args.data_dir}_{channel_independent_str}_loaders.pkl")
    vali_datas_pickle_path = os.path.join("pickles", f"{args.data_dir}_batch-{args.batch_size}", f"vali_{args.data_dir}_{channel_independent_str}_datas.pkl")
    vali_loaders_pickle_path = os.path.join("pickles", f"{args.data_dir}_batch-{args.batch_size}", f"vali_{args.data_dir}_{channel_independent_str}_loaders.pkl")
    test_datas_pickle_path = os.path.join("pickles", f"{args.data_dir}_batch-{args.batch_size}", f"test_{args.data_dir}_{channel_independent_str}_datas.pkl")
    test_loaders_pickle_path = os.path.join("pickles", f"{args.data_dir}_batch-{args.batch_size}", f"test_{args.data_dir}_{channel_independent_str}_loaders.pkl")

    if os.path.exists(train_datas_pickle_path) and os.path.exists(train_loaders_pickle_path) \
        and os.path.exists(vali_datas_pickle_path) and os.path.exists(vali_loaders_pickle_path) \
        and os.path.exists(test_datas_pickle_path) and os.path.exists(test_loaders_pickle_path):
        print("Loading existing datasets and dataloaders...")
        with open(train_datas_pickle_path, 'rb') as f:
            train_datas = pickle.load(f)
        with open(train_loaders_pickle_path, 'rb') as f:
            train_loaders = pickle.load(f)
        with open(vali_datas_pickle_path, 'rb') as f:
            vali_datas = pickle.load(f)
        with open(vali_loaders_pickle_path, 'rb') as f:
            vali_loaders = pickle.load(f)
        with open(test_datas_pickle_path, 'rb') as f:
            test_datas = pickle.load(f)
        with open(test_loaders_pickle_path, 'rb') as f:
            test_loaders = pickle.load(f)
        total_train_len = sum([len(x) for x in train_loaders.values()])
        total_vali_len = sum([len(x) for x in vali_loaders.values()])
        total_test_len = sum([len(x) for x in test_loaders.values()])
    else:
        print(f"Creating datasets and dataloaders for {train_datas_pickle_path}...")
        all_data_filename = os.listdir(os.path.join(args.root_path, args.data_dir))
        train_datas, train_loaders = {}, {}
        vali_datas, vali_loaders = {}, {}
        test_datas, test_loaders = {}, {}
        total_train_len = 0
        total_vali_len = 0
        total_test_len = 0

        # create a list of train_data and dataloaders through iteration
        for data_filename in all_data_filename:
            for event_root_code in event_root_code_list:
                region_name = data_filename.split(".")[0]
                train_data, train_loader = data_provider(args, 'train', event_root_code=event_root_code, data_filename=data_filename)
                vali_data, vali_loader = data_provider(args, 'val', event_root_code=event_root_code, data_filename=data_filename)
                test_data, test_loader = data_provider(args, 'test', event_root_code=event_root_code, data_filename=data_filename)

                if train_data and train_loader:
                    train_datas[(region_name, event_root_code)] = train_data
                    train_loaders[(region_name, event_root_code)] = train_loader
                    total_train_len += len(train_loader)
                
                if vali_data and vali_loader:
                    vali_datas[(region_name, event_root_code)] = vali_data
                    vali_loaders[(region_name, event_root_code)] = vali_loader
                    total_vali_len += len(vali_loader)

                if test_data and test_loader:
                    test_datas[(region_name, event_root_code)] = test_data
                    test_loaders[(region_name, event_root_code)] = test_loader
                    total_test_len += len(test_loader)

        if not os.path.exists(os.path.join("pickles", f"{args.data_dir}_batch-{args.batch_size}")):
            os.makedirs(os.path.join("pickles", f"{args.data_dir}_batch-{args.batch_size}"))
        with open(train_datas_pickle_path, 'wb') as f:
            pickle.dump(train_datas, f)
        with open(train_loaders_pickle_path, 'wb') as f:
            pickle.dump(train_loaders, f)
        with open(vali_datas_pickle_path, 'wb') as f:
            pickle.dump(vali_datas, f)
        with open(vali_loaders_pickle_path, 'wb') as f:
            pickle.dump(vali_loaders, f)
        with open(test_datas_pickle_path, 'wb') as f:
            pickle.dump(test_datas, f)
        with open(test_loaders_pickle_path, 'wb') as f:
            pickle.dump(test_loaders, f)

    print("total_train_len: ", total_train_len)
    print("total_vali_len: ", total_vali_len)
    print("total_test_len: ", total_test_len)

    device = torch.device('cuda:0')

    time_now = time.time()
    train_steps = total_train_len 

    print("args.model: ", args.model)
    if args.model == 'GPT4MTS':
        model = GPT4MTS(args, device)
        model.to(device)
    else:
        raise Exception(f"{args.model} not implemented yet")

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    criterion = nn.MSELoss()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for (region_name, event_root_code), train_loader in tqdm(train_loaders.items()):
            for i, train_loader_item in enumerate(train_loader):
                (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_summary) = train_loader_item
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                batch_summary = batch_summary.to(device)

                outputs = model(batch_x, ii, batch_summary)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)   
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
                
        train_loss = np.average(train_loss)
        vali_losses = []
        for (region_name, event_root_code), vali_loader in vali_loaders.items():
            vali_loss = vali(model, vali_loader, criterion, args, device, ii)
            vali_losses.append(vali_loss)
        avg_vali_loss = np.average(vali_losses)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, avg_vali_loss))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    tests_mse = {}
    tests_mae = {}
    for (region_name, event_root_code), test_loader in test_loaders.items():
        if region_name not in tests_mse:
            tests_mse[region_name] = {}
        if region_name not in tests_mae:
            tests_mae[region_name] = {}
        mse, mae = test(model, test_loader, args, device, ii)
        tests_mse[region_name][event_root_code] = float(mse)
        tests_mae[region_name][event_root_code] = float(mae)

    with open(mse_res_file_path, 'w') as f:
        json.dump(tests_mse, f)
    with open(mae_res_file_path, 'w') as f:
        json.dump(tests_mae, f)

    print("tests_mse: ", tests_mse)
    print("tests_mae: ", tests_mae)