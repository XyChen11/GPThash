# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for TrAISformer.
    https://arxiv.org/abs/2109.03958

The code is built upon:
    https://github.com/karpathy/minGPT
"""

import math
import logging
import pdb
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from .Focal_loss import focal_loss
from mamba_ssm import Mamba

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen + 20, config.max_seqlen + 20))
                                    #  .view(1, 1, config.max_seqlen + 20, config.max_seqlen + 20))
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen + 10, config.max_seqlen + 10))
                                     .view(1, 1, config.max_seqlen + 10, config.max_seqlen + 10))

        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        LSTM 网络构造函数
        
        参数说明：
        - input_size:  输入特征维度 (e.g., 时间序列的每个时间步的特征数量)
        - hidden_size: 隐藏层神经元数量
        - num_layers:  LSTM 堆叠层数
        - output_size: 输出维度 (分类任务对应类别数，回归任务对应预测值维度)
        - dropout:      dropout概率 (仅当num_layers>1时生效)
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 核心层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,        # 输入数据格式为 (batch, seq_len, feature)
            dropout=dropout if num_layers > 1 else 0,  # 层间dropout
            bidirectional=False
        )
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播
        
        输入形状: (batch_size, seq_len, input_size)
        输出形状: (batch_size, output_size)
        """
        # 初始化隐藏状态 (h0, c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 前向传播
        # out: (batch_size, seq_len, hidden_size)
        # hn, cn: 最后时刻的隐藏状态
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 全连接层
        out = self.fc(out)
        return out

    def init_weights(self):
        """初始化权重 (示例)"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

class EncTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(EncTokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1,
                                   padding_mode='zeros')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class MambaLayer(nn.Module):
    def __init__(self, d_model = 512, d_state = 128, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=d_state)
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.mamba(self.ln(x))
        x = self.mlp(x)
        return x

class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model = None):
        super().__init__()

        self.config = config
        self.device = config.device
        self.inp_size = config.geohash_size
        self.full_size = config.full_size
        self.all_embd = config.all_embd
        self.BCEloss = torch.nn.BCELoss()
        self.sigmod = nn.Sigmoid()
        self.register_buffer(
            "att_sizes", 
            torch.tensor([config.lat_size]))
        self.register_buffer(
            "emb_sizes", 
            torch.tensor([config.n_embd]))
        
        if hasattr(config,"partition_mode"):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model
        
        if hasattr(config,"blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                self.blur_module = nn.Conv1d(1, 1, 3, padding = 1, padding_mode = 'replicate', groups=1, bias=False)
                if not self.blur_learnable:
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1/3)
            else:
                self.blur_module = None
                
        
        if hasattr(config,"lat_min"): # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max-config.lat_min
            self.lon_range = config.lon_max-config.lon_min
            self.sog_range = 30.
            
        if hasattr(config,"mode"): # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to 
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"
    

        # Passing from the 4-D space to a high-dimentional space
        self.inp_emb = nn.Embedding(self.inp_size, config.n_embd)
        self.vel_emb = nn.Embedding(config.vel_size, config.n_embd)
        # self.inp_emb_1 = EncTokenEmbedding(20, config.n_lat_embd)
        # self.inp_emb_2 = EncTokenEmbedding(20, config.n_lon_embd)
            
            
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.mlp = nn.Linear(config.n_embd * 2, config.n_embd)

        if config.base_model == "Transformer":
            # transformer
            self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        elif config.base_model == "Mamba":
            # mamba
            self.blocks = nn.Sequential(*[MambaLayer(d_model=config.all_embd) for _ in range(5)])
        elif config.base_model == "LSTM":
            # LSTM
            self.blocks = LSTMModel(input_size=config.all_embd, hidden_size=128, num_layers=5, output_size=config.all_embd, dropout=0.2)
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if self.mode in ("mlp_pos","mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False) # Classification head
            
        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
   
    
    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.
        
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            model: currenly only supports "uniform".
        
        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x*self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):
            
            idxs = (x*self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:,:,:2])
            # pdb.set_trace()
            idxs[:,:,0] = torch.round(lat_ids.reshape((bs,seqlen))).long()
            idxs[:,:,1] = torch.round(lon_ids.reshape((bs,seqlen))).long()                               
            return idxs, idxs_uniform

    # 将二进制 Geohash 编码转换为 one-hot 向量
    def binary_to_one_hot(self,binary_geohash):
        return [int(bit) for bit in binary_geohash]

    # 函数将轨迹数据转换为 one-hot 向量形式的二进制 Geohash
    def convert_to_one_hot_geohash(self,tensor, precision=8):
        num_trajectories, trajectory_length, _ = tensor.shape
        one_hot_geohash_list = []
        
        for i in range(num_trajectories):
            trajectory_geohashes = []
            for j in range(trajectory_length):
                lat, lon = tensor[i, j].tolist()
                binary_geohash = self.encode_geohash(lat, lon, precision=precision)
                one_hot_vector = self.binary_to_one_hot(binary_geohash)
                trajectory_geohashes.append(one_hot_vector)
            one_hot_geohash_list.append(trajectory_geohashes)
        
        return torch.tensor(one_hot_geohash_list,dtype=torch.float32)

    def forward(self, x, vel, masks = None, with_targets=False, return_loss_tuple=False):
        """
        10 speeds 分段 prompting, 10 个速度段对应的索引为 0-49, 再嵌入.
        Args:
            x: a Tensor of size (batchsize, seqlen).
            vels: a Tensor of size (batchsize, seqlen).
            masks: a Tensor of the same size of x. masks[idx] = 0. if 
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:], 
                otherwise inputs = x.
        Returns: 
            logits, loss
        """
        
        # if self.mode in ("mlp_pos","mlp",):
        #     idxs, idxs_uniform = x, x # use the real-values of x.
        # else:            
        #     # Convert to indexes
        #     idxs= self.convert_to_one_hot_geohash(x[:,:,:2])
        idxs = x.long()
        vels = vel.long()
        # idxs_l = idxs[:,:,:self.inp_size]
        # idxs_s = idxs[:,:,self.inp_size:]
        # idxs_a = torch.cat((idxs_l,idxs_s),dim=2)
        if with_targets:
            inputs = idxs[:,:-1].contiguous().to(self.device)
            vels = vels[:, :-1].to(self.device)
            targets = idxs[:,1:].contiguous().to(self.device)
        
        else:
            inputs_real = x
            inputs = idxs
            vels = vels
            targets = None
        batchsize, seqlen = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."


        lat_embeddings = self.inp_emb(inputs)

        ## Differential velocity
        # vel_norm = F.normalize(vels.float(), p=2, dim=-1)
        # pos_diff = lat_embeddings[:, 1:, :] - lat_embeddings[:, :-1, :]
        # pos_diff = torch.cat((pos_diff, lat_embeddings[:, -1, :].unsqueeze(1)), dim=1).to(self.device)
        # if with_targets:
        #     velocity_vectors = pos_diff * vel_norm[:, :-1].unsqueeze(-1) # [16, 59, 256]
        # else:
        #     velocity_vectors = pos_diff * vel_norm[:, :].unsqueeze(-1) # [16, 20, 256]
        #     velocity_vectors = torch.cat((velocity_vectors, lat_embeddings[:, -1, :].unsqueeze(1)), dim=1).to(self.device)    
        
        vel_max = vels.max(dim=1)[0].unsqueeze(-1) # [16, 1]
        vel_min = vels.min(dim=1)[0].unsqueeze(-1) # [16, 1]
        vel_scale = (vel_max - vel_min) / (self.config.vel_size) # [16, 1]
        velocity_vectors = (vels - vel_min) + 0.0001
        velocity_vectors = (velocity_vectors // vel_scale).to(int)
        velocity_vectors = torch.clamp(velocity_vectors, 0, 49)
        vel_embeddings = self.vel_emb(velocity_vectors) # [16, 59, 256] or [16, 20, 256]
            
        token_embeddings = lat_embeddings
        position_embeddings = self.pos_emb[:, :seqlen, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
        fea = self.drop(token_embeddings + position_embeddings)
        # fuse position and velocity
        fuse_fea = torch.concatenate((vel_embeddings[:, -10:, :], fea), dim=1) # (batchsize, seqlen + velocity seq len, self.all_embd)
        # fuse_fea = self.mlp(fuse_fea)
        fuse_fea = self.blocks(fuse_fea) # (bs, seqlen + velocity seq len, n_embd)
        fuse_fea = fuse_fea[:, -seqlen:, :] # (bs, seqlen, n_embd) get positions features
        fuse_fea = self.ln_f(fuse_fea) # (bs, seqlen, n_embd)

        logits = self.head(fuse_fea) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)
        
        # logits=self.sigmod(logits)
        lat_logits = logits
        # Calculate the loss
        loss = None
        loss_tuple = None
        
        if targets is not None:
            # lat_loss = self.BCEloss(lat_logits.contiguous().view(-1, 20),
                                # targets[:,:,:20].contiguous().view(-1, 20))
            # lon_loss = self.BCEloss(lon_logits.contiguous().view(-1, 20),
            #                     targets[:,:,20:].contiguous().view(-1, 20))
            # sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), 
            #                            targets[:,:,2].view(-1), 
            #                            reduction="none").view(batchsize,seqlen)
            # cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size), 
            #                            targets[:,:,3].view(-1), 
            #                            reduction="none").view(batchsize,seqlen)
            # lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size), 
            #                            targets[:,:,0].view(-1), 
            #                            reduction="none").view(batchsize,seqlen)
            # lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size), 
            #                            targets[:,:,1].view(-1), 
            #                            reduction="none").view(batchsize,seqlen)                     
           
            # loss_sog = focal_loss(alpha=0.25, gamma=2, num_classes=1000)
            # loss_cog = focal_loss(alpha=0.25, gamma=2, num_classes=500)
            loss_lat = focal_loss(alpha=0.25, gamma=2, num_classes=self.full_size)
            # loss_lon = focal_loss(alpha=0.25, gamma=2, num_classes=1000)
            # sog_loss = loss_sog(sog_logits.view(-1, self.sog_size), 
            #                            targets[:,:,2].view(-1)).view(batchsize,seqlen)
            # cog_loss = loss_cog(cog_logits.view(-1, self.cog_size), 
            #                            targets[:,:,3].view(-1)).view(batchsize,seqlen)
            lat_loss = loss_lat(lat_logits.view(-1, self.full_size), 
                                       targets[:,:].view(-1)).view(batchsize,seqlen)
            # lon_loss = loss_lon(lon_logits.view(-1, self.lon_size), 
            #                            targets[:,:,1].view(-1)).view(batchsize,seqlen)                     

            
            
            
            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1) 

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1,1,self.full_size)).reshape(lat_probs.shape)
            

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.full_size),
                                                  targets[:,:].view(-1),
                                                  reduction="none").view(batchsize,seqlen)

                    lat_loss += self.blur_loss_w*blurred_lat_loss


                    lat_probs = blurred_lat_probs

                    

        
            loss = lat_loss 
            loss = loss.mean()
        
            if masks is not None:
                loss = (loss*masks).sum(dim=1)/masks.sum(dim=1)
        
            loss = loss.mean()
        
        if return_loss_tuple:
            return lat_logits, loss, loss_tuple
        else:
            return logits, loss
        
