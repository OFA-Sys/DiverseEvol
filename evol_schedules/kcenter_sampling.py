import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from evol_schedule_base import EvolSchedule
from tqdm import tqdm
import copy
from utils import compute_kernel_bias, transform_and_normalize  # whitening

class KCenterSampling(EvolSchedule):
    def __init__(self,
        model,
        full_data_path, 
        val_set_size,
        tokenizer,
        data_path_root,
        output_dir_root,
        train_args,
        whiten_n_components=-1,
        max_random_times=-1,  # random how many times -> to compute argmin/max(sbleu/vendi)
        vendi_argmax_rank=-1
    ):
        super(KCenterSampling, self).__init__(
            model,
            full_data_path, 
            val_set_size,
            tokenizer,
            data_path_root,
            output_dir_root,
            train_args,
            whiten_n_components,
            max_random_times,
            vendi_argmax_rank
        )
        self.whiten_n_components = whiten_n_components  # whitening -> reduce to how many dimensions
    
    def query(self, rd, n, use_model_path):
        # get all data embeddings using prev_rd's model
        embeddings_lst = self.get_embeddings_all_data(rd=rd, use_model_path=use_model_path)  # (num_all_data, emb_dim)
        embeddings = torch.stack(embeddings_lst)  # (num_all_data, hidden_dim=4096)
        device = embeddings.device
        embeddings = embeddings.float()  # fp32
        # whitening
        if self.whiten_n_components > 0:
            kernel, bias = compute_kernel_bias(embeddings) # kernel.shape = (emb_dim, emb_dim), bias.shape = (1, emb_dim) 
            kernel = kernel[:, :self.whiten_n_components] # kernel.shape = (emb_dim, n_dim)
            embeddings = transform_and_normalize(embeddings, kernel=kernel, bias=bias)  # (num_all_data, n_dim=N_COMPONENTS)
        print(f"*** Round {rd} **Rank = {torch.distributed.get_rank()}, **Embeddings.shape = {embeddings.shape}")
        dist_mat = torch.empty((0, embeddings.shape[0])).to(device)  # init empy dist_mat .shape=(0, num_all_data)
        # partion embeddings -> to compute distances
        step = 10000000000  # how many data in each partition? -> change this accordingly if you run into OOM on a single GPU
        for i in range(0, embeddings.shape[0], step):
            print(f"*** Round {rd} **Rank = {torch.distributed.get_rank()}, **Dies_e -- Start Computing From Idx = {i}")
            e = embeddings[i:i+step,:]                
            dist_e = torch.matmul(e, embeddings.t())
            torch.cuda.empty_cache()
            dist_mat = torch.concat([dist_mat, dist_e], dim=0)
            print(f"*** Round {rd} **Rank = {torch.distributed.get_rank()}, **Computed dist_e.shape = {dist_e.shape}")
            
        sq = torch.tensor(dist_mat.diagonal(), device=device, requires_grad=False).reshape(len(self.labeled_idx), 1)  # diagonal ->  (num_all_data, 1) 
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.t()
        dist_mat = torch.sqrt(dist_mat)  # -> distances between all_data_embeddings (num_all_data, num_all_data) (diagonal=0)
        labeled_idxs_tmp = copy.deepcopy(self.labeled_idx)
        mat = dist_mat[~labeled_idxs_tmp, :][:, labeled_idxs_tmp]  # unlabeled<->labeled distances (unlabeled_size, labeled_size)
        
        # sample n new datapoints from unlabeled_pool
        for i in tqdm(range(n), ncols=n):
            mat_min = mat.min(dim=1).values   # (unlabeled_size, ) min_distance to existing labeled datapoint
            q_idx_tmp = mat_min.argmax()  # argmax(min_distance)
            q_idx = torch.arange(self.n_pool)[~labeled_idxs_tmp][q_idx_tmp]  # find its index in full dataset
            labeled_idxs_tmp[q_idx] = True  # add to labeled_pool
            left_unlabeled_idxs = torch.ones(mat.shape[0], dtype=torch.bool, device=device, requires_grad=False)  # BOOL: remaining unchosen data (in dist_mat)
            left_unlabeled_idxs[q_idx_tmp] = 0  # newly added -> 0
            mat = mat[left_unlabeled_idxs, :].reshape(-1, mat.shape[1])  # left_unlabeled<->labeled distances (unlabeled_size-(i+1), labeled_size)
            mat = torch.concat((mat, dist_mat[~labeled_idxs_tmp, q_idx][:, None]), dim=1)   # # left_unlabeled<->updated_labeled distances (unlabeled_size-(i+1), labeled_size+(i+1))
        new_idx = torch.arange(self.n_pool)[(self.labeled_idx ^ labeled_idxs_tmp)]  # new datapoints -> indices in full dataset
        
        return new_idx
    

