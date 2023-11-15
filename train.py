import os
import yaml
import torch
import argparse

from evol_schedules import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, KCenterSampling
from utils import get_tokenizer, smart_tokenizer_and_embedding_resize, get_model, rank0_print

## GET_EVOL_SCHEDULES
def get_evol_schedule(evol_schedule_name):
    if evol_schedule_name == "RandomSampling":
        return RandomSampling
    if evol_schedule_name == "LeastConfidence":
        return LeastConfidence
    elif evol_schedule_name == "MarginSampling":
        return MarginSampling
    elif evol_schedule_name == "EntropySampling":
        return EntropySampling
    elif evol_schedule_name == "KCenterSampling":
        return KCenterSampling
    # elif evol_schedule_name == "VendiSampling":
    #     return VendiSampling


## RUN
def main(config_file):
    # load configuration
    with open(config_file, 'r') as f:
        args = yaml.full_load(f)
    rank0_print('Configuration loaded!')
    rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

    # makedirs    
    args["rd_data_path_root"] = f"evol_res/{args['result_dir_name']}/data"
    args["output_dir_root"] = f"evol_res/{args['result_dir_name']}/output"
    os.makedirs(args["rd_data_path_root"], exist_ok=True)
    os.makedirs(args["output_dir_root"], exist_ok=True)

    # round 0 -> round n
    for rd in range(0, args['n_round']+1):
        rank0_print(f"*** Round {rd} ======================================================================================================")
        if rd==0: # round 0: build core objects: model, tokenizer, evol_schedule
            model = get_model(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"])
            rank0_print('*** Round 0 ** New Model initilized!')
            tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"], model_max_length=args["model_max_length"],)
            rank0_print(f'*** Round 0 ** New Tokenizer initilized!')
            tokenizer, model = smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, 
                                                                    tokenizer=tokenizer, 
                                                                    model=model)  # fix tokenizer's special_token_maps
            rank0_print(f'*** Round 0 ** smart_tokenizer_and_embedding_resize done!')
            evol_schedule = get_evol_schedule(evol_schedule_name=args["evol_schedule_name"])(model=model,
                                                                                             full_data_path=args["full_data_path"], 
                                                                                             val_set_size=args["val_set_size"],
                                                                                             tokenizer=tokenizer,
                                                                                             data_path_root=args["rd_data_path_root"],
                                                                                             output_dir_root=args["output_dir_root"],
                                                                                             train_args=args["train_args"],
                                                                                             whiten_n_components=args["whiten_n_components"] if "whiten_n_components" in args else -1,
                                                                                             max_random_times=args["max_random_times"] if "max_random_times" in args else -1,  # random how many times -> to compute argmin/max(sbleu/vendi)
                                                                                             vendi_argmax_rank=args["vendi_argmax_rank"] if "vendi_argmax_rank" in args else -1,)            
            rank0_print(f'*** Round 0 ** New evol_schedule built!')
            # round 0: initialize labeled pool & unlabeled pool
            evol_schedule.initialize_labeled_data(num=args['init_label_num']) 
            # round 0: get & save round-0 intialized data
            evol_schedule.save_rd_labeled_unlabeled_data(rd=0)
            rank0_print(f"*** Round 0 ** Training-Data-Size = {len(evol_schedule.labeled_idx[evol_schedule.labeled_idx==True])}")
        else:  # round 1->n: query new samples & update core objects: model, tokenizer, evol_schedule
            # query new samples
            if torch.distributed.get_rank()==0:
                query_idx = evol_schedule.query(rd=rd-1, 
                                                n=args['n_query'], 
                                                use_model_path=f"{args['output_dir_root']}/rd_{rd-1}")  # quering new data at the beginning of each round -> using prev_rd model's embedding
                rank0_print(f"*** Round {rd} ** Added-Sample-idx = {query_idx}")
                evol_schedule.update_rd(rd=rd, add_labeled_idx=query_idx,)  # update pools -> label newly-selected samples & save round data
                rank0_print(f"*** Round {rd} ** Training-Data-Size = {len(evol_schedule.labeled_idx[evol_schedule.labeled_idx==True])}")
            
            torch.cuda.empty_cache()
            # update core objects: new model & tokenizer instance, update evol_schedule
            model = get_model(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"]) 
            rank0_print(f'*** Round {rd} ** New Model initilized!')
            tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=args["model_name_or_path"], cache_dir=args["cache_dir"], model_max_length=args["model_max_length"],)
            rank0_print(f'*** Round {rd} ** New Tokenizer initilized!')
            tokenizer, model = smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, 
                                                                    tokenizer=tokenizer,
                                                                    model=model)  # fix tokenizer's special_token_maps
            rank0_print(f'*** Round {rd} ** smart_tokenizer_and_embedding_resize done!') 
            evol_schedule.model = model
            evol_schedule.tokenizer = tokenizer
            rank0_print(f'*** Round {rd} ** New Model & Tokenizer built into evol_schedule!')
        # round {rd}: train
        evol_schedule.train(rd=rd)
        rank0_print(f"*** Round {rd} ** Training Done !!!")

    rank0_print("DiverseEvol Done ^_^")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True,)
    parser.add_argument('--wandb_key', type=str, default='b189be602734f52ac19168f0656370c1bd309771')
    args = parser.parse_args()
    
    import wandb
    wandb.login(key=args.wandb_key)
    
    main(config_file=args.config_file)
