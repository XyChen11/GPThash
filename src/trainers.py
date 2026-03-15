

"""Boilerplate for training a neural network.

References:
    https://github.com/karpathy/minGPT
"""

import os
import math
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from . import utils
import random
import time
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
# from trAISformer import TB_LOG

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(model,
           seqs,
           vels,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. 
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed
        vels_cond = vels if vels.size(1) <= max_seqlen else vels[:, -max_seqlen:]  # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        logits, _ = model(seqs_cond, vels_cond)
        d2inf_pred = torch.zeros((logits.shape[0])).to(seqs.device) + 0.5

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1] / temperature  # (batch_size, data_size)

        lat_logits = logits

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs_uniform = seqs_cond[:, -1:]
            lat_idxs= idxs_uniform[:, 0].unsqueeze(1)
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        # pred = (lat_logits >= 0.5).int()


        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)

        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)


        ix = lat_ix
        # # convert to x (range: [0,1))
        # x_sample = (ix.float() + d2inf_pred.unsqueeze(1)) / 106496
        x_sample = ix

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample), dim=1)
        
    return seqs


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 16
    learning_rate = 1e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, 
                 model, 
                 train_dataset, 
                 test_dataset, 
                 config, args, 
                 savedir=None, 
                 device=torch.device("cpu"), 
                 INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir
        self.args = args

        self.device = device
        self.model = model.to(device)
        self.INIT_SEQLEN = INIT_SEQLEN

        self.log_file = None
        self._setup_logging(config.save_log)

    def _setup_logging(self, savedir):
        """创建日志目录并配置日志格式"""
        if savedir:
            # 创建保存目录（如果不存在）
            os.makedirs(savedir, exist_ok=True)
            base_model = self.config.base_model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_template = f"{base_model}_{timestamp}.log"

            # 设置日志文件路径
            self.log_file = os.path.join(savedir, filename_template)
            
            # 配置logging到文件与控制台
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.warning("Log savedir not provided, logging to console only.")

    # def save_checkpoint(self, best_epoch):
    #     # DataParallel wrappers keep raw model object in .module attribute
    #     raw_model = self.model.module if hasattr(self.model, "module") else self.model
    #     #         logging.info("saving %s", self.config.ckpt_path)
    #     # logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
    #     logging.debug("Model state before saving:")
    #     for name, param in raw_model.named_parameters():
    #         if param.requires_grad:
    #             logging.debug(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    #     # torch.save(raw_model.state_dict(), self.config.ckpt_path)
    #     # logging.info(f"Checkpoint saved at {self.config.ckpt_path}")

    def save_checkpoint_last(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        model_path = os.path.join(self.savedir, self.args.model_select)
        logging.info(f"Best epoch: {best_epoch + 1:03d}, saving model to {model_path}")
        logging.debug("Bast Model state before saving:")
        for name, param in raw_model.named_parameters():
            if param.requires_grad:
                logging.debug(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
        torch.save(raw_model.state_dict(), model_path)
        logging.info(f"Checkpoint saved at {model_path}")

    def train(self, best_valid_loss = None):
        logging.info("\n" + "="*50)
        logging.info(f"Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("="*50 + "\n")
        logging.info(f"Training configuration: epoch: {self.config.max_epochs}" + "\n" +\
                     f"base_model: {self.config.base_model}" + "\n" +\
                    f"embedding: {self.config.n_embd}" + "\n" +\
                    f"batch_size: {self.config.batch_size}" + "\n" +\
                    f"geohash_size: {self.config.geohash_size}" + "\n" +\
                    f"word_size: {self.args.word_size}" + "\n" +\
                    f"model name: {self.args.model_select}" + "\n"
                    )

        model, config, INIT_SEQLEN, = self.model, self.config, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=0.00005)# raw_model.configure_optimizers(config)
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, (seqs, vels, masks) in pbar:

                # place data on the correct device
                seqs = seqs.to(self.device)
                vels = vels.to(self.device)
                masks = masks[:, :-1].to(self.device)


                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = model(seqs, vels,
                                                         masks=masks,
                                                         with_targets=True,
                                                         return_loss_tuple=return_loss_tuple)
                    else:
                                        # forward the model
                        logits, loss = model(seqs, vels, masks=masks, with_targets=True)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    optimizer.zero_grad()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (seqs >= 0).sum()  
                            # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                        

                    ###### report progress
                    # pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")


                    # tb logging
                    # if TB_LOG:
                    #     tb.add_scalar("loss",
                    #                   loss.item(),
                    #                   epoch * n_batches + it)
                    #     tb.add_scalar("lr",
                    #                   lr,
                    #                   epoch * n_batches + it)

                    #     for name, params in model.head.named_parameters():
                    #         tb.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                    #         tb.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                    #     if model.mode in ("gridcont_real",):
                    #         for name, params in model.res_pred.named_parameters():
                    #             tb.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                    #             tb.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)

            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, {d_reg_loss / d_n:.5f}, lr {lr:e}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}.")
            else:
                if return_loss_tuple:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")

            if not is_train:
                test_loss = float(np.mean(losses))
                # logging.info("test loss: %f", test_loss)
                return test_loss
        if best_valid_loss is not None:
            best_loss = best_valid_loss
        else:
            best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0

        for epoch in range(config.max_epochs):
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch+1}/{config.max_epochs} - Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            if test_loss <= best_loss:
                self.save_checkpoint_last(epoch)
                best_loss = test_loss
                logging.info(f"New best model at epoch {epoch+1} with loss {best_loss:.4f}")

            # if epoch+1 == config.max_epochs:
            #     self.save_checkpoint(1000)
        logging.info(f"Training Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Logging to file: {self.log_file}")