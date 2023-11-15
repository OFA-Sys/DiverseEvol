import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from evol_schedule_base import EvolSchedule

class RandomSampling(EvolSchedule):
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
        super(RandomSampling, self).__init__(
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

    def query(self, rd, n, use_model_path):
        unlabeled_idx = torch.arange(self.n_pool)[~self.labeled_idx.bool()]  # # current unlabeled_idx
        most_uncertain_indices = torch.randperm(unlabeled_idx.shape[0])[:n]  # num_unlabeled -> randperm -> choose random n samples
        most_uncertain_sample_idx = unlabeled_idx[most_uncertain_indices]
        return most_uncertain_sample_idx
