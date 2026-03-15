from pathlib import Path
from src.data_loader_HB_globel_v2 import Dataset_flight, read_data, denormalize_arrays, read_data_train_token
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
from tokenizers.processors import TemplateProcessing
from itertools import product
import os
# from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def deduplicate_geohashes(geohash_lists):
    unique_geohashes = set()  # 使用集合来自动去重
    for sublist in geohash_lists:
        for geohash in sublist:
            unique_geohashes.add(geohash)  # 添加到集合中

    return list(unique_geohashes)  # 转换回列表

def generate_six_char_geohashes(three_char_geohashes):
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    six_char_geohashes = []

    for geohash in three_char_geohashes:
        for c1 in base32:
            # for c2 in base32:
                # six_char_geohashes.append(geohash + c1 + c2)
            six_char_geohashes.append(geohash + c1)

    return six_char_geohashes

data_path = str(PROJECT_ROOT / 'data' / 'quin33.sqlite')
save_path = str(PROJECT_ROOT / 'data' / 'tokenizer_3D_7+1word_blur.json')
##globel##
# dataset_train, dataset_test,min_t,max_t = read_data(data_path,50,2)
##local##
dataset_train, dataset_test, idxs = read_data_train_token(data_path,60,7)

dataset_train_1 = deduplicate_geohashes(idxs)

dataset_train_2 = generate_six_char_geohashes(dataset_train_1)
# training_data = [" ".join(dataset_train_2)]
training_data = dataset_train_2

# geohash_chars = "0123456789bcdefghjkmnpqrstuvwxyz"

# # 生成前两位为 'uz' 和 'ux' 的所有 6 位 GeoHash
# uz_geohashes = ['uz' + ''.join(p) for p in product(geohash_chars, repeat=4)]
# ux_geohashes = ['ux' + ''.join(p) for p in product(geohash_chars, repeat=4)]

# # 合并两种组合
# all_geohashes = uz_geohashes + ux_geohashes

# # 将 GeoHash 列表转换为训练数据格式
# training_data = [" ".join(all_geohashes)]

tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

# 使用空格作为分词的预处理器
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 设置训练器，不需要特殊标记
# trainer = trainers.WordLevelTrainer(vocab_size=len(training_data))
trainer = trainers.WordLevelTrainer(
    vocab_size=len(training_data),  
    special_tokens=["[UNK]"], 
    min_frequency=1 
)
# tokenizer.normalizer = normalizers.NFC()
# tokenizer.post_processor = TemplateProcessing(
#     single="[CLS] $A [SEP]",
#     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
#     special_tokens=[
#         ("[CLS]", 1),
#         ("[SEP]", 2),
#     ],
# )
tokenizer.train_from_iterator(training_data, trainer=trainer)

# encoded = tokenizer.encode(" ".join(dataset_train[0]))
# decoded = tokenizer.decode(encoded.ids)

# print("origianl",dataset_train[0] )
# print("Encoded:", encoded.tokens)
# print("Decoded:", decoded)

tokenizer.save(save_path)
# data_set_train = Dataset_flight(dataset_train,60)
# train_data = DataLoader(data_set_train,
#                             batch_size=64,
#                             shuffle=True)

# data_set_test = Dataset_flight(dataset_test, 60)
# test_data = DataLoader(data_set_test,
#                             batch_size=64,
#                             shuffle=False)