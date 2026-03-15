
"""Configuration flags to run the main script.
"""

import os
import pickle
import torch


class Config():
    retrain = 0
    tb_log = False
    device = torch.device('cuda:0')
    # device = torch.device("cpu")
    
    max_epochs = 10
    batch_size = 16
    n_samples = 16
    
    init_seqlen = 20
    max_seqlen = 60
    min_seqlen = 10
    
    dataset_name = "ct_dma"

    if dataset_name == "ct_dma": #==============================
   
        # When mode == "grad" or "pos_grad", sog and cog are actually dlat and 
        # dlon
        lat_size = 67395 # 536064
        lon_size = lat_size
        sog_size = lat_size
        cog_size = lat_size

        
        n_lat_embd = 256
        n_lon_embd = 256
        n_sog_embd = 256
        n_cog_embd = 128
    
        lat_min = 55.5
        lat_max = 58.0
        lon_min = 10.3
        lon_max = 13

    
    #===========================================================================
    # Model and sampling flagszq
    base_model = "Transformer" # "Mamba", "LSTM", "Transformer"
    mode = "pos"  #"pos", "pos_grad", "mlp_pos", "mlpgrid_pos", "velo", "grid_l2", "grid_l1", 
                            # "ce_vicinity", "gridcont_grid", "gridcont_real", "gridcont_gridsin", "gridcont_gridsigmoid"
    sample_mode =  "velo" # "pos", "pos_vicinity" or "velo"
    top_k = 10 # int or None 
    r_vicinity = 40 # int
    
    # Blur flags
    #===================================================
    blur = True
    blur_learnable = False
    blur_loss_w = 1.0
    blur_n = 2
    if not blur:
        blur_n = 0
        blur_loss_w = 0
    
    # Data flags
    #===================================================
    datadir = f"./data/{dataset_name}/"
    trainset_name = f"{dataset_name}_train.pkl"
    validset_name = f"{dataset_name}_valid.pkl"
    testset_name = f"{dataset_name}_test.pkl"


    geohash_size = 67395 # 536064
    vel_size = 50
    
    # model parameters
    #===================================================
    n_head = 8
    n_layer = 10
    full_size = 67395 # 536064
    n_embd = 184 # 160 # 768
    all_embd = 184 # 160 # 768
    # base GPT config, params common to all GPT versions
    embd_pdrop = 0.2
    resid_pdrop = 0.2
    attn_pdrop = 0.2
    
    # optimization parameters
    #===================================================
    learning_rate = 6e-4 # 1e-3
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = True
    warmup_tokens = 512*20 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    num_workers = 0 # for DataLoader
    
    filename = f"{dataset_name}"\
        + f"-{mode}-{sample_mode}-{top_k}-{r_vicinity}"\
        + f"-blur-{blur}-{blur_learnable}-{blur_n}-{blur_loss_w}"\
        + f"-data_size-{lat_size}-{lon_size}-{sog_size}-{cog_size}"\
        + f"-embd_size-{n_lat_embd}-{n_lon_embd}-{n_sog_embd}-{n_cog_embd}"\
        + f"-head-{n_head}-{n_layer}"\
        + f"-bs-{batch_size}"\
        + f"-lr-{learning_rate}"\
        + f"-seqlen-{init_seqlen}-{max_seqlen}"
    savedir = "./results/"+base_model+"/" # "./results/"+filename+"/"
    save_log = os.path.join("./results/log/")
    
    checkdir = "./results/"+base_model+"/"
    ckpt_path = os.path.join(checkdir,"model_100_better.pt")
    ckpt_last_path = os.path.join(checkdir,"model_best_better_9word.pt")