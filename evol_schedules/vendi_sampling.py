## For Analysis: Diversity->Performance

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from evol_schedule_base import EvolSchedule
from vendi_score import vendi


class VendiSampling(EvolSchedule):
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
        super(VendiSampling, self).__init__(
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
        self.vendi_argmax_rank=vendi_argmax_rank
        self.max_random_times=max_random_times

    def query(self, rd, n, use_model_path):
        embeddings_lst = self.get_embeddings_all_data(rd=rd,            
                                                      use_model_path="decapoda-research/llama-7b-hf", # foundation llm as embed_model
                                                      ) 
        embeddings = torch.stack(embeddings_lst)  # (num_all_data, hidden_dim=4096)
        embeddings = embeddings.float()  # fp32

        random_time_to_add_sample_idx = {}
        random_time_to_vendi = {}
        for random_time in range(self.max_random_times):
            unlabeled_idx = torch.arange(self.n_pool)[~self.labeled_idx.bool()]  
            labeled_idx = torch.arange(self.n_pool)[self.labeled_idx.bool()]  
            random_choice_indices = torch.randperm(unlabeled_idx.shape[0])[:n]  
            add_sample_idx = unlabeled_idx[random_choice_indices]  
            rd_now_labeled_data_idx = torch.concat([labeled_idx, add_sample_idx])  
            rd_now_labeled_data_embeddings = embeddings[rd_now_labeled_data_idx]  
            print(f"rd_now_labeled_data_embeddings.shape: {rd_now_labeled_data_embeddings.shape}")

            emb_norm = rd_now_labeled_data_embeddings / torch.norm(rd_now_labeled_data_embeddings, dim=-1, keepdim=True) 
            similarity = torch.mm(emb_norm, emb_norm.T) 
            sim_K = similarity.cpu().numpy()
            vendi_diversity_score = vendi.score_K(sim_K)

            random_time_to_add_sample_idx[random_time] = add_sample_idx
            random_time_to_vendi[random_time] = vendi_diversity_score

        sorted_random_time_to_vendi = dict(sorted(random_time_to_vendi.items(), key=lambda x: x[1], reverse=True)) 
        print(f"sorted_random_time_to_vendi: {sorted_random_time_to_vendi}")
        choice_random_time = list(sorted_random_time_to_vendi.keys())[self.vendi_argmax_rank] 
        print(f"choice_random_time: {choice_random_time}")
        choice_add_sample_idx = random_time_to_add_sample_idx[choice_random_time] 
        print(f"VENDI-SAMPLING --> CHOICE-RANDOM -> VENDI_SCORE = {sorted_random_time_to_vendi[choice_random_time]}")

        return choice_add_sample_idx


