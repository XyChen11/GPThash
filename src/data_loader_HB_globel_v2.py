import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from utils.tools import MinMaxScaler
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
import random
warnings.filterwarnings('ignore')
from tokenizers import Tokenizer
from math import log10
from .Geohash3 import encode3

class DatabaseHandle:
    base_table_name = "main"

    def __init__(self, database_path):
        self.connection = sqlite3.connect(database_path)

    def select_distinct(self, column_name, table=None):
        if table is None:
            table = self.base_table_name
        print("分析数据库中...")
        rtn = pd.read_sql_query(f"select DISTINCT {column_name} from {table}",
                                self.connection)
        print("分析完成...")
        return rtn[column_name]

    def select_by(self, key, column_name, select="*", table=None):
        if table is None:
            table = self.base_table_name
        return pd.read_sql_query("select {} from {} where {}=='{}';".format(select, table, column_name, key),
                                 self.connection)

    def close(self):
        self.connection.close()


class FlightPathDatabaseHandle(DatabaseHandle):
    base_table_name = "fw_flightHJ"
    main_key_name = "HBID"
    data_buffer = {}
    def __init__(self, database_path):
        super().__init__(database_path)
        self.main_keys = self.select_distinct(self.main_key_name)

    def select_by(self, key, column_name=None, select="*", table=None):
        if column_name is None:
            column_name = self.main_key_name
        return super().select_by(key, column_name, select, table)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if item in self.data_buffer.keys():
            return self.data_buffer[item]
        else:
            raw_data = self.select_by(self.main_keys[item])
            data = FlightPathDataFrame(raw_data)
            self.data_buffer.update({item:data})
            return data

    def __len__(self):
        return len(self.main_keys)


class FlightPathDataFrame(pd.DataFrame):
    """
    到这一步为止，数据原封不动，没有走任何变化处理
    """
    def __init__(self,df,*args,**kwargs):
        super().__init__(df,*args,**kwargs)
        self.to_datetime("WZSJ")

    def to_datetime(self,column):
        self[column] = pd.to_datetime(self[column])
        return self
    
def resample_dataframe(arr, n):
    df = pd.DataFrame(arr)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%dT%H:%M:%S.%f')

    df = df.set_index(df.columns[0])
    df = df.iloc[:, 0:].apply(pd.to_numeric)
    
    
    df=df[~df.index.duplicated()]
    # df = df.interpolate(method='linear')
    # df = df.interpolate(method='linear')
    df_resampled = df.resample(n).bfill(limit=1).interpolate(method='linear')

    
    return df_resampled

class Dataset_flight(Dataset):
    def __init__(self, data, velocity, max_len):
        self.data = data
        self.velocity = velocity
        self.max_len = max_len

    
    def __getitem__(self, index):
        seqlen = min(len(self.data), self.max_len)
        seq_x = self.data[index]
        vel = self.velocity[index]
        mask = torch.zeros(self.max_len)
        mask[:seqlen] = 1
        
        return seq_x, vel, mask
    
    def __len__(self):
        return len(self.data)

def sliding_window(matrix, window_len, n):
        new_shape = (1 + (matrix.shape[0] - window_len) // n, window_len)
        new_matrix = np.zeros(new_shape)
        for i in range(new_shape[0]):
            new_matrix[i] = matrix[i * n : i * n + window_len]
        if (new_shape[0] - 1) * n + window_len < matrix.shape[0]:
            new_matrix = np.concatenate((new_matrix, matrix[-window_len:][np.newaxis, :]), axis=0)
        return new_matrix

def convert_to_geohash(data, precision=5):
    geohash_data = []
    for trajectory in data:
        # geohash_trajectory = [geohash2.encode(lat, lon, precision=precision) for lon, lat in trajectory]
        geohash_trajectory = [encode3(lat, lon, hei, precision=precision) for lon, lat, hei in trajectory]
        geohash_data.append(geohash_trajectory)
    return geohash_data

def read_data(data_path,max_len,seg_d, precision=8, token_select=''):
    flight_raw = FlightPathDatabaseHandle(data_path)
    tra_num = len(flight_raw)
    train_num = 0.7*tra_num
    test_num = 0.3 * tra_num
    win_len = max_len
    sig_data_train = []
    sig_data_test = []
    data_train = []
    data_test = []
    all_data = []
    velocity = []
    velocity_train = []
    velocity_test = []
    sig_velocity_train = []
    sig_velocity_test = []
    count=0

    for sample in flight_raw:
        # count+=1
        a = sample.iloc[:, [1, 2, 3, 4, 5]].values

        a = a[~np.isnan(a[:,1:].astype(float)).any(axis=1)]
        # for idx in range(a.shape[0]):
        #     c= a[idx,1:].astype(float)
        #     if not(np.any(np.isnan(c))):
        #         a = a[idx:,:]
        #         break

                
        # plt.figure(figsize=(9, 6), dpi=150)
        # plt.scatter(a[:,1],a[:,2],color='r',s=10)
        # plt.show()
        if not np.any(a):
            continue
        b = resample_dataframe(a, '20S')
        b = b.iloc[:,:].values
        # acceleration = np.diff(b[:, 3])  # 使用numpy的差分函数，计算速度列的差分，得到加速度
        # filled_acceleration = np.insert(acceleration, 0, acceleration[0])


# 创建新的5维轨迹数据
        # b = np.concatenate((b, filled_acceleration.reshape(-1, 1)), axis=1)
        if np.any(np.isnan(b)):
            raise ValueError("NaN exist")
        # plt.figure(figsize=(9, 6), dpi=150)
        # plt.scatter(a[:,0],a[:,1],color='r',s=10)
        # plt.show()
        if b.shape[0] < win_len:
            continue
#####################
    #     normalized_data = np.zeros_like(b)

    # # 创建一个MinMaxScaler对象
    #     scaler = MinMaxScaler()

    #     # 对每一行进行归一化
    #     for i in range(b.shape[1]):
    #         row = b[:, i].reshape(1, -1)  # 将行向量转换为列向量
    #         normalized_row = scaler.fit_transform(row)
    #         normalized_data[:, i] = normalized_row
    #     b = normalized_data
        
        # plt.figure(figsize=(9, 6), dpi=150)
        # plt.plot(b[:,0],b[:,1],"o", markersize=3, color='r')
        # plt.show()
        # b = MinMaxScaler().fit_transform(b)
        # b[:,1] = MinMaxScaler().fit_transform(b[:,1].T).T
        # b[:,2] = MinMaxScaler().fit_transform(b[:,2].T).T
        # b[:,3] = MinMaxScaler().fit_transform(b[:,3].T).T

        # plt.figure(figsize=(9, 6), dpi=150)
        # # plt.plot(normalized_res_data[i,:,0],normalized_res_data[i,:,1],"o", markersize=3, color='r')
        # plt.plot(b[:,0],b[:,1],"o", markersize=3, color='r')
        # plt.show()
##################
        all_data.append(b[:,:3]) # 0，1，2三列，经度纬度高度
        # all_data.append(b[:,:2]) # 0，1两列
        velocity.append(b[:, 3]) # 速度
    # all_data 649, 297, 3
    # velocity 649, 297, 1

    combined = list(zip(all_data, velocity))
    random.seed(1)
    random.shuffle(combined)
    all_data, velocity = zip(*combined)

    idxs= convert_to_geohash(all_data, precision=precision) # 649, 182 words
    for count in range(len(idxs)):
        if count < train_num:
            data_train.append(idxs[count])
            velocity_train.append(velocity[count])
        if count > train_num:
            data_test.append(idxs[count])
            velocity_test.append(velocity[count])

    all_data_train = data_train
    all_data_test = data_test

    string_data_1 = [" ".join(seq) for seq in all_data_train]
    string_data_2 = [" ".join(seq) for seq in all_data_test]

    len1 = [len(seq) for seq in all_data_train]
    tokenizer = Tokenizer.from_file(token_select)
    indexed_data_1 = [tokenizer.encode(sequence).ids for sequence in string_data_1]
    indexed_data_2 = [tokenizer.encode(sequence).ids for sequence in string_data_2]

    len11 = [len(seq) for seq in indexed_data_1]
    all_data_train = [np.array(sublist) for sublist in indexed_data_1]
    all_data_test = [np.array(sublist) for sublist in indexed_data_2]
    # print(len(all_data_test), len(velocity_test))
    for i in range(len(all_data_train)):
        sig_data_train.append(sliding_window(all_data_train[i],win_len,seg_d))
        sig_velocity_train.append(sliding_window(velocity_train[i],win_len,seg_d))
    for i in range(len(all_data_test)):
        sig_data_test.append(sliding_window(all_data_test[i],win_len,40))
        sig_velocity_test.append(sliding_window(velocity_train[i],win_len,40))
    # traj_num=np.array([array.shape[0] for array in sig_data_test]).reshape(-1, 1)
    # traj_num = np.cumsum(traj_num)
    # print(traj_num)
    res_data_train = np.concatenate(sig_data_train,axis=0)
    res_data_test = np.concatenate(sig_data_test,axis=0)
    velocity_train = np.concatenate(sig_velocity_train, axis=0)
    velocity_test = np.concatenate(sig_velocity_test, axis=0)
    return res_data_train, res_data_test, velocity_train, velocity_test




        
def normalize_arrays(arr_list):
    normalized_list = []
    min_vals = np.min(np.concatenate(arr_list, axis=0), axis=0)
    max_vals = np.max(np.concatenate(arr_list, axis=0), axis=0)
    for arr in arr_list:
        normalized_arr = (arr - min_vals) / (max_vals - min_vals)
        normalized_list.append(normalized_arr)
    return normalized_list, min_vals, max_vals

def denormalize_arrays(arr_list, min_vals, max_vals):
    denormalized_list = []
    for i in range(arr_list.shape[0]):
        denormalized_arr = arr_list[i,:,:] * (max_vals - min_vals) + min_vals
        denormalized_list.append(denormalized_arr)
    denormalized_list = np.array(denormalized_list)
    return denormalized_list








# def sliding_window(matrix, window_len, n):
#     new_shape = (1 + (matrix.shape[0] - window_len) // n, window_len, matrix.shape[1])
#     new_matrix = np.zeros(new_shape)
#     for i in range(new_shape[0]):
#         new_matrix[i] = matrix[i * n : i * n + window_len]
#     if (new_shape[0] - 1) * n + window_len < matrix.shape[0]:
#         new_matrix = np.concatenate((new_matrix, matrix[-window_len:, :][np.newaxis, :, :]), axis=0)
#     return new_matrix

def read_data_train_token(data_path, max_len, precision):
    flight_raw = FlightPathDatabaseHandle(data_path)
    tra_num = len(flight_raw)
    train_num = 0.7 * tra_num
    test_num = 0.3 * tra_num
    win_len = max_len
    sig_data_train = []
    sig_data_test = []
    data_train = []
    data_test = []
    all_data = []
    count = 0

    for sample in flight_raw:
        # count+=1
        a = sample.iloc[:, [1, 2, 3, 4, 5]].values

        a = a[~np.isnan(a[:, 1:].astype(float)).any(axis=1)]
        # for idx in range(a.shape[0]):
        #     c= a[idx,1:].astype(float)
        #     if not(np.any(np.isnan(c))):
        #         a = a[idx:,:]
        #         break

        # plt.figure(figsize=(9, 6), dpi=150)
        # plt.scatter(a[:,1],a[:,2],color='r',s=10)
        # plt.show()
        if not np.any(a):
            continue
        b = resample_dataframe(a, '20S')
        b = b.iloc[:, :].values
        # acceleration = np.diff(b[:, 3])  # 使用numpy的差分函数，计算速度列的差分，得到加速度
        # filled_acceleration = np.insert(acceleration, 0, acceleration[0])

        # 创建新的5维轨迹数据
        # b = np.concatenate((b, filled_acceleration.reshape(-1, 1)), axis=1)
        if np.any(np.isnan(b)):
            raise ValueError("NaN exist")
        # plt.figure(figsize=(9, 6), dpi=150)
        # plt.scatter(a[:,0],a[:,1],color='r',s=10)
        # plt.show()
        if b.shape[0] < win_len:
            continue
        all_data.append(b[:, :3])  # 0，1，2三列
        # all_data.append(b[:,:2]) # 0，1两列
    random.seed(1)
    random.shuffle(all_data)
    sss = 0
    idxs = convert_to_geohash(all_data, precision=precision)
    return 0, 0, idxs

if __name__ == '__main__':
    a=1
    # database_path = 'FW.sqlite'
    # data_set = Dataset_flight(database_path, 
    #                           10, 
    #                           size=[25, 40],
    #                           data_split = [0.7, 0.3],
    #                           flag='train'
    # )
    # data_loader = DataLoader(
    #     data_set,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=0,
    # )
    # steps = len(data_loader)
    # for i, (batch_x, batch_y) in enumerate(data_loader):
    #     a=1











    # flight_path = FlightPathDatabaseHandle(database_path)

    # # sig_data = []
    # # count=0
    # # for sample in flight_path:
    # #     count+=1
    # #     a = sample.iloc[:, [3, 4, 5, 6]].values
    # #     sig_data.append(sliding_window(a,50,20))
    # #     if count == 20:
    # #         break
    # # res_data = np.concatenate(sig_data,axis=0)
    # # rr = res_data[2,0:20,:]
    # # 可以直接索引
    # for i in range(10):
    #     print(i, "--\n", flight_path[i])

    # # 可以生成迭代对象
    # count = 0
    # for sample in flight_path:
    #     count+=1
    #     print(count, "--\n",sample)
    #     if count==20:
    #         break