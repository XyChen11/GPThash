"""
written by cxy
as a alternation of trAISformer.py
"""
#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from src.data_loader_HB_globel_v2 import Dataset_flight, read_data
from src import models
from src import trainers
from src import utils
from tokenizers import Tokenizer
import time
from torch.utils.data import DataLoader
import pandas as pd
from src.metrics import metric
from src.config_trAISformer import Config
import geohash2
from src.Geohash3 import decode_geohash, decode3_exactly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def one_hot_to_binary(one_hot_vector):
    return ''.join(str(int(bit)) for bit in one_hot_vector)

def convert_to_coordinates(one_hot_geohash_trajectory):
    original_coordinates = []
    
    for trajectory in one_hot_geohash_trajectory:
        trajectory_coords = []
        for one_hot_vector in trajectory:
            binary_geohash = one_hot_to_binary(one_hot_vector)
            lat, lon, height = decode_geohash(binary_geohash)
            trajectory_coords.append((lat, lon, height))
        original_coordinates.append(np.array(trajectory_coords))
    
    return original_coordinates
def geohash_matrix_to_coordinates(geohash_matrix):
    """
    将一个二维 Geohash 矩阵转换为一个三维矩阵，其中每个元素包含纬度和经度, 高度。

    参数:
    geohash_matrix (numpy.ndarray): 包含 Geohash 的二维矩阵。

    返回:
    numpy.ndarray: 三维矩阵，形状为 (rows, cols, 3)，其中每个元素为 [latitude, longitude, height]。
    """
    # 获取矩阵的形状
    rows, cols = geohash_matrix.shape

    # 初始化一个空的三维矩阵来存储经纬度
    coordinates_matrix = np.empty((rows, cols, 3), dtype=float)

    # 将每个 Geohash 转换为经纬度
    for i in range(rows):
        for j in range(cols):
            geohash_code = geohash_matrix[i, j]
            latitude, longitude, height, lat_err, lon_err, hei_err = decode3_exactly(geohash_code)
            coordinates_matrix[i, j] = [float(latitude), float(longitude), float(height)]

    return coordinates_matrix


def indices_to_values(index_matrix, tokenizer):
    """
    将一个二维索引矩阵还原为原始数值矩阵。

    参数:
    index_matrix (numpy.ndarray): 包含索引的二维矩阵。
    tokenizer (Tokenizer): 已训练的分词器。

    返回:
    numpy.ndarray: 包含原始数值的二维矩阵。
    """
    # 获取矩阵的形状
    rows, cols = index_matrix.shape

    # 初始化一个空的矩阵来存储原始数值
    values_matrix = np.empty((rows, cols), dtype=object)

    # 将每个索引转换为原始值
    for i in range(rows):
        for j in range(cols):
            index = index_matrix[i, j]
            token = tokenizer.id_to_token(int(index))
            values_matrix[i, j] = token if token is not None else "UNKNOWN"

    return values_matrix

def main(Config, args, device):
    """主执行流程"""
    utils.set_seed(42)
    
    ## Data
    print(f"加载数据集: {Config.dataset_name}")
    dataset_train, dataset_test, velocity_train, velocity_test = read_data(
        str(DATA_DIR / "quin33.sqlite"), Config.max_seqlen, 2, args.precision, args.token_select
    )
    # train_loader = DataLoader(
    #     Dataset_flight(dataset_train, Config.max_seqlen),
    #     batch_size=Config.batch_size, shuffle=True
    # )
    # test_loader = DataLoader(
    #     Dataset_flight(dataset_test, Config.max_seqlen),
    #     batch_size=Config.batch_size, shuffle=False
    # )
    aisdls={}
    data_set_train = Dataset_flight(dataset_train, velocity_train, 60)
    aisdls['train'] = DataLoader(data_set_train,
                                batch_size=Config.batch_size,
                                shuffle=True)

    data_set_test = Dataset_flight(dataset_test, velocity_test, 60)
    aisdls['test'] = DataLoader(data_set_test,
                                batch_size=Config.batch_size,
                                shuffle=False)

    max_seqlen = Config.max_seqlen # 60
    init_seqlen = Config.init_seqlen # 20

    ## model
    model = models.TrAISformer(Config)
    model.to(device)
    if args.train_with_exist == 1:
        model.load_state_dict(torch.load(os.path.join(Config.savedir, args.model_select)))
    ## train
    if Config.retrain:
        print("开始训练...")
        trainer = trainers.Trainer(
            model, 
            data_set_train, 
            data_set_test, 
            Config, 
            args,
            savedir=Config.savedir, 
            device=device,
            INIT_SEQLEN=init_seqlen
        )
        if args.train_with_exist == 1:
            trainer.train(best_valid_loss = args.best_valid_loss)
        else: 
            trainer.train()

    print("开始评估...")
    try:
        model.load_state_dict(torch.load(os.path.join(Config.savedir, args.model_select)))
    except FileNotFoundError:
        print("No best model")

    model.eval()
    tokenizer = Tokenizer.from_file(args.token_select)
    
    pred_all = []
    true_all = []
    in_all = []

    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():
        for it, (seqs, vels, masks) in pbar:
            start = time.time()
            seqs_init = seqs[:, :init_seqlen].to(device) # batch_size, 20
            vels_init = vels[:, :init_seqlen].to(device) # batch_size, 20
            
            masks = masks[:, :max_seqlen].to(device) # batch_size, 60
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - init_seqlen, Config.n_samples)).to(Config.device)
            preds = trainers.sample(model,
                                    seqs_init, vels_init,
                                    max_seqlen - init_seqlen,
                                    temperature=1.0,
                                    sample=True,
                                    sample_mode=Config.sample_mode,
                                    r_vicinity=Config.r_vicinity,
                                    top_k=Config.top_k)
            pred_all.append(preds[:, init_seqlen:max_seqlen].cpu())
            true_all.append(seqs[:, init_seqlen:max_seqlen].cpu())
            in_all.append(seqs[:, :init_seqlen].cpu())
            
        pred_all = np.concatenate(pred_all,axis=0)
        true_all = np.concatenate(true_all,axis=0)
        in_all = np.concatenate(in_all,axis=0)
        pred_all = indices_to_values(pred_all, tokenizer)
        true_all = indices_to_values(true_all, tokenizer)
        pred_all = geohash_matrix_to_coordinates(pred_all)
        true_all = geohash_matrix_to_coordinates(true_all)
        pred_all = np.array(pred_all)
        true_all = np.array(true_all)
    
    def print_metrics(pred, true, name):
        mae, mse, rmse, mape, mspe, rse, _, eulid, rae, rel_rmse = metric(pred, true)
        print(f'{name}: MAE:{mae:.4f}, RAE:{rae:.4f}, MSE:{mse:.4f}, RSE:{rse:.4f}, REL_RMSE:{rel_rmse:.4f}')

    print_metrics(pred_all[:,:,0:1], true_all[:,:,0:1], 'lat')
    print_metrics(pred_all[:,:,1:2], true_all[:,:,1:2], 'lon')
    print_metrics(pred_all[:,:,2:3], true_all[:,:,2:3], 'hei')

    # 绘图配置（启用3D模式）
    fig = plt.figure(figsize=(20, 20), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    colors = {'truth': 'k', 'pred': 'r'}

    for idx in range(1, len(pred_all), 10):
        # 提取三维坐标（经度、纬度、高度）
        x_true = true_all[idx, :, 0]
        y_true = true_all[idx, :, 1]
        z_true = true_all[idx, :, 2]  # 第三维度（如高度）
        
        x_pred = pred_all[idx, :, 0]
        y_pred = pred_all[idx, :, 1]
        z_pred = pred_all[idx, :, 2]  # 第三维度
        
        # 绘制三维轨迹线（真实轨迹）
        ax.plot(x_true, y_true, z_true,
            color=colors['truth'],
            linestyle='-',
            linewidth=1.5,
            alpha=0.8,
            label='Ground Truth' if idx == 0 else '')
        
        # 绘制三维散点（预测轨迹）
        ax.scatter(x_pred, y_pred, z_pred,
               color=colors['pred'],
               s=15,
               edgecolor='k',
               linewidths=0.5,
               label='Prediction' if idx == 0 else '')

    # 坐标轴标签和视角调整
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_zlabel('Height (m)', fontsize=12)  # 第三维度标签（根据数据含义修改）

    # 图例和保存
    ax.legend(loc='upper right', fontsize=10)
    output_image_dir = PROJECT_ROOT / 'testimg_global'
    os.makedirs(output_image_dir, exist_ok=True)
    plt.savefig(output_image_dir / '3D_trajectory_comparison.png', bbox_inches='tight', dpi=300)
    plt.show() 
    plt.close()

    # 保存预测数据
    with open(PROJECT_ROOT / 'trAIS3.pkl', 'wb') as f:
        pickle.dump(pred_all, f)

    print("评估完成，结果已保存")

if __name__ == "__main__":
    """
    python scripts/train.py --word_size 7+1_blur --model_select model_best_better_7+1word_blur.pt --token_select data/tokenizer_3D_7+1word_blur.json --epoch 10 --n_cuda 1 --batch_size 16 --n_embd 256 --n_head 8 --retrain
    """
    Config.savedir = str(PROJECT_ROOT / "results" / Config.base_model)
    Config.save_log = str(PROJECT_ROOT / "results" / "log")
    os.makedirs(Config.savedir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_size', type=str, default='5+1', \
                        help='word size 5+1 or 7+1_blur')
    parser.add_argument('--retrain', action="store_true")
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--n_cuda', type=str, default='0')
    parser.add_argument('--model_select', default='model_best_better_5+1word.pt', type=str, 
                        help='select model with how much precision')
    parser.add_argument('--token_select', default='data/tokenizer_3D_7+1word_blur.json', type=str,help='tokenizer path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument("--n_embd", type=int, default=160, help="embedding size should be Divisible by n_head")
    parser.add_argument("--n_head", type=int, default=8, help="number of attention heads")
    parser.add_argument("--precision", type=int, default=8, help="precision of the word")
    parser.add_argument("--train_with_exist", action='store_true', default=False, help="whether to train with existing model")
    parser.add_argument("--vel_size", type=int, default=50, help="catagorize velocity into x categories")
    parser.add_argument("--best_valid_loss", type=float, default=4.112, help="best valid loss")
    args = parser.parse_args()
    if not os.path.isabs(args.token_select):
        args.token_select = str((PROJECT_ROOT / args.token_select).resolve())
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.n_cuda
    if args.word_size == '5+1':
        mbd_size = 263008
        args.precision = 6
    elif args.word_size == '7+1_blur':
        mbd_size = 536064
        args.precision = 8
    Config.n_embd = args.n_embd
    Config.all_embd = args.n_embd
    Config.n_head = args.n_head
    Config.max_epochs = args.epoch
    Config.batch_size = args.batch_size
    Config.lat_size = mbd_size
    Config.lon_size = mbd_size
    Config.sog_size = mbd_size
    Config.cog_size = mbd_size
    Config.geohash_size = mbd_size
    Config.full_size = mbd_size
    Config.retrain = args.retrain     
    Config.device = torch.device('cuda:'+args.n_cuda if torch.cuda.is_available() else 'cpu')   
    Config.vel_size                                                                                                                                                                                                                                                                                                          
    main(Config, args, Config.device)