import torch
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from evol_schedule_base import EvolSchedule

class EntropySampling(EvolSchedule):
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
        super(EntropySampling, self).__init__(
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
        mean_max_logits_scores, mean_logits_margin_scores, mean_entropies = self.get_embeddings_unlabeled_data(rd=rd, 
                                                                                                               use_model_path=use_model_path, 
                                                                                                               requires_lc=False, 
                                                                                                               requires_margin=False, 
                                                                                                               requires_entropy=True)
        mean_entropies = torch.tensor(mean_entropies)  # (num_unlabeled,)
        most_uncertain_indices = mean_entropies.sort(descending=True).indices[:n]  # argmax(entropy) -> most uncertain
        unlabeled_idx = torch.arange(self.n_pool, device=most_uncertain_indices.device)[~self.labeled_idx.bool()]  # current unlabeled_idx
        most_uncertain_sample_idx = unlabeled_idx[most_uncertain_indices]
        return most_uncertain_sample_idx

