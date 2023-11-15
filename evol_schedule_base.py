import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from nltk import word_tokenize

from utils import jload, jdump, make_supervised_data_module, get_model, rank0_print


# Base Self-Evol EvolSchedule
class EvolSchedule:
    def __init__(self, 
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
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.full_data_path = full_data_path
        if val_set_size > 0: # doesnt fit our piepline
            raise NotImplementedError 
        self.val_data = None
        # load full-sized source data -> for indexing all samples
        with open(self.full_data_path, "r") as f:
            self.train_data = json.load(f)  # fixed -> for indexing all samples
        self.n_pool = len(self.train_data)
        # keep track of labeled/unlabeled (1/0) index
        self.labeled_idx = torch.zeros(self.n_pool, dtype=bool)  
        # keep track labeled_idx for each round 
        self.rd_to_labeled_idx = {}  
        # saving options
        self.data_path_root = data_path_root
        self.output_dir_root = output_dir_root
        train_args["output_dir"] = self.output_dir_root  # dummy init -> to update for each round
        self.training_args = TrainingArguments(**train_args)

    def initialize_labeled_data(self, num):
        """Randomly init labeled pool"""
        if torch.distributed.get_rank() == 0:
            tmp_idxs = torch.randperm(self.n_pool)  # randomly permute indices (total_data_size, )
            self.labeled_idx[tmp_idxs[:num]] = True  # labeled=1, unlabeled=0 (total_data_size,)
            self.rd_to_labeled_idx[0] = tmp_idxs[:num].sort().values  # keep track of labeled_idx (sorted) for each round

    def save_rd_labeled_unlabeled_data(self, rd):
        """update & save current labaled & unlabeled pool"""
        if torch.distributed.get_rank() == 0:
            # obtain & check labeled_idx for current round
            labeled_idx = torch.arange(self.n_pool)[self.labeled_idx.bool()]  # self.labeled_idx -> kept upated
            rd_labeled_idx = self.rd_to_labeled_idx[rd]  # self.rd_to_labeled_idx -> kept track
            assert labeled_idx.equal(rd_labeled_idx)  # check -> self.labeled_idx gets properly updated like tracked.
            # query self.train_data -> current labeled & unlabeled data
            labeled_data_json_format = [self.train_data[_] for _ in labeled_idx] 
            unlabeled_idx = torch.arange(self.n_pool)[~self.labeled_idx.bool()]
            unlabeled_data_json_format = [self.train_data[_] for _ in unlabeled_idx]
            rank0_print(f"*** Round {rd} ** labeled_idx: {labeled_idx}")
            # save current labeled & unlabeld data
            rd_labeled_data_path = f"{self.data_path_root}/rd_{rd}_labeled.json"
            rd_unlabeled_data_path = f"{self.data_path_root}/rd_{rd}_unlabeled.json"
            if torch.distributed.get_rank() == 0:
                retry = 0
                while True:
                    jdump(labeled_data_json_format, rd_labeled_data_path)
                    try:
                        temp_labeled = jload(rd_labeled_data_path)
                        rank0_print(f"*** Round {rd} ** jdump(labeled_data_json_format, rd_labeled_data_path) SUCESSFUL to --> {rd_labeled_data_path}")
                        break
                    except:
                        retry += 1
                        rank0_print(f"*** Round {rd} ** jdump(labeled_data_json_format, rd_labeled_data_path) FAILED to --> {rd_labeled_data_path}")
                        if retry > 5:
                            raise
                        continue
                retry = 0
                while True:
                    jdump(unlabeled_data_json_format, rd_unlabeled_data_path)
                    try:
                        temp_unlabeled = jload(rd_unlabeled_data_path)
                        rank0_print(f"*** Round {rd} ** jdump(unlabeled_data_json_format, rd_unlabeled_data_path) SUCESSFUL to --> {rd_unlabeled_data_path}")
                        break
                    except:
                        retry += 1
                        rank0_print(f"*** Round {rd} ** jdump(unlabeled_data_json_format, rd_unlabeled_data_path) FAILED to --> {rd_unlabeled_data_path}")
                        if retry > 5:
                            raise
                        continue
    
    def get_updated_train_data(self, rd):
        """load & make round labeled data -> training data"""
        data_path = f"{self.data_path_root}/rd_{rd}_labeled.json"
        rd_data_module = make_supervised_data_module(tokenizer=self.tokenizer, data_path=data_path)
        return rd_data_module
    
    def get_unlabeled_data(self, rd):
        """load & make round unlabeled data -> candidate data pool for selecting new samples"""
        data_path = f"{self.data_path_root}/rd_{rd}_unlabeled.json"
        rd_unlabeled_data_module = make_supervised_data_module(tokenizer=self.tokenizer, 
                                                                data_path=data_path)
        return rd_unlabeled_data_module
    
    def train(self, rd):
        # get round labeled data -> for training
        rd_data_module = self.get_updated_train_data(rd)
        # sanity-check
        if torch.distributed.get_rank() == 0:
            for sanity_sample in rd_data_module["train_dataset"]:
                break
            rank0_print(f"*** Round {rd} ** SANITY-CHECK: Training-Sample#1. - TEXT.:\n\n{self.tokenizer.decode(sanity_sample['input_ids'])}\n\n")
        rd_output_dir = f"{self.output_dir_root}/rd_{rd}"
        self.training_args.output_dir = rd_output_dir # update round-output-dir
        trainer = Trainer(model=self.model, 
                          tokenizer=self.tokenizer, 
                          args=self.training_args,
                          **rd_data_module)
        trainer.train()
        trainer.save_state()
        trainer.save_model(output_dir=rd_output_dir)
        rank0_print(f"*** Round {rd} ** Trainer State & Trained Model Saved To --> {rd_output_dir} ***")
        self.model.save_pretrained(f"{rd_output_dir}/pretrained")  # save_model() somehow may result in error -> save_pretrained() again, just in case.
        rank0_print(f"*** Round {rd} ** Trainer State & Trained Model Save-Pretrained To --> {rd_output_dir}/pretrained ***")
    
    def update_rd(self, rd, add_labeled_idx):
        """add newly selected samples to labeled_data & update unlabeled_data pool"""
        if torch.distributed.get_rank() == 0:
            self.labeled_idx[add_labeled_idx.to(self.labeled_idx.device)] = True
            labeled_idx = torch.arange(self.n_pool)[self.labeled_idx.bool()]
            self.rd_to_labeled_idx[rd] = labeled_idx  # keep track of each round's labeled data
            self.save_rd_labeled_unlabeled_data(rd=rd)  # save labeled & unlabeled data
    
    def get_embeddings_all_data(self, rd, use_model_path):
        """compute last hidden states for full dataset -> distance-based schedules"""
        all_data = make_supervised_data_module(tokenizer=self.tokenizer, data_path=self.full_data_path)
        model = get_model(model_name_or_path=use_model_path, cache_dir=None)
        rank0_print(f'*** Round {rd} ** Trained Model loaded!')
        return self.get_embeddings(rd=rd, data=all_data, model=model, requires_lc=False, requires_margin=False, requires_entropy=False)
    
    def get_embeddings_unlabeled_data(self, rd, use_model_path, requires_lc=False, requires_margin=False, requires_entropy=False):
        """compute last hidden states for unlabeled data -> confidence-based schedules"""
        unlabeled_data = self.get_unlabeled_data(rd=rd)
        model = get_model(model_name_or_path=use_model_path, cache_dir=None)
        rank0_print(f'*** Round {rd} ** Trained Model loaded!')
        return self.get_embeddings(rd=rd, data=unlabeled_data, model=model, requires_lc=requires_lc, requires_margin=requires_margin, requires_entropy=requires_entropy)

    def get_embeddings(self, rd, data, model, requires_lc=False, requires_margin=False, requires_entropy=False):
        """compute last hidden states for a data_module"""
        model.cuda()
        model.eval()
        last_hidden_states_avg_all = [] # init empty container -> keep track of all datapoints' last_hidden_states_avg
        mean_max_logits_scores_all = [] # init empty container -> keep track of all datapoints' avg(max_logits)
        mean_logits_margin_all = [] # init empty container -> keep track of all datapoints' avg(logits_margin)
        mean_entropies_all = [] # init empty container -> keep track of all datapoints' avg(entropies)
        with torch.no_grad():
            for _,datapoint in enumerate(data["train_dataset"]):
                input_ids = datapoint["input_ids"].unsqueeze(0).to(model.device)
                labels = datapoint["labels"].unsqueeze(0).to(model.device)
                # sanity-check
                if _ == 0 and torch.distributed.get_rank() == 0:
                    rank0_print(f"*** Round {rd} ** SANITY-CHECK: Predicting-Sample#1. - TEXT.:\n\n{self.tokenizer.decode(datapoint['input_ids'])}\n\n")
                result = model(input_ids=input_ids, labels=labels, return_dict=True, output_hidden_states=True)
                if not (requires_lc or requires_margin or requires_entropy):  # dont need logits -> distance-based schedule
                    hidden_states = result["hidden_states"]
                    last_layer_hidden_states = hidden_states[-1]  # (batch_size=1, seq_len, hidden_dim=4096)
                    # avg_pooling the sequence-hidden-states
                    last_hidden_states_avg = torch.mean(last_layer_hidden_states.squeeze(0), dim=0)  # -> (hidden_dim)
                    last_hidden_states_avg_all.append(last_hidden_states_avg)  # keep track
                else:  # need logits -> confidence-based schedule
                    logits = result["logits"].squeeze(0)  # (seq_len, vocab_size)
                    logits_sorted = torch.sort(logits, dim=-1).values  # (seq_len, vocab_size) sort logits at every token
                    if requires_lc: # least confidence
                        max_logits = logits_sorted[:,-1]  # (seq_len) 
                        mean_max_logits_score = torch.mean(max_logits)  # avg(max_logits)
                        mean_max_logits_scores_all.append(mean_max_logits_score.item())
                    if requires_margin: # margin
                        max_logits = logits_sorted[:,-1]  # (seq_len)
                        second_max_logits = logits_sorted[:,-2]  # (seq_len) 
                        logits_margin = max_logits - second_max_logits  # (seq_len) margin = max_logits - 2nd_max_logits
                        mean_logits_margin = torch.mean(logits_margin)  ## avg(logits_margin)
                        mean_logits_margin_all.append(mean_logits_margin.item())
                    if requires_entropy: # entropy
                        probs = F.softmax(logits, dim=1)  # softmax -> logits->probs (seq_len, vocab_size)
                        entropies = -(torch.log(probs)*probs).sum(dim=1)  # entropy at every token position ->>> (seq_len)
                        mean_entropy = torch.mean(entropies) # avg(entropy)
                        mean_entropies_all.append(mean_entropy.item())
                if _==1 or (_!=0 and _%10000 == 0): # report progress
                    rank0_print(f"*** Round {rd} ** Predict-Progress -- {_} DONE !")
        if not (requires_lc or requires_margin or requires_entropy):
            return last_hidden_states_avg_all # List[tensor. shape=(hidden_dim=4096)] .len=num_data
        else:
            return mean_max_logits_scores_all, mean_logits_margin_all, mean_entropies_all # List[float] .len=num_unlabeld 
              




    def get_prompt_words_lst_from_data(self, data, prompter):
        prompt_words_lst = []
        for i in tqdm(range(len(data)), total=len(data)):
            d = data[i]
            # generate prompt -- 算diveristy时带上output -> 和training对齐
            prompt = prompter.generate_prompt(
                                                instruction=d["instruction"],
                                                input=d["input"], 
                                                label=d["output"],  # 算diveristy时带上output -> 和training对齐
                                                )
            prompt_no_newline = prompt.strip()
            # prompt_same_with_training = " " + prompt_no_newline.replace("\n\n", "\n")
            prompt_same_with_training = " " + prompt_no_newline
            # rank0_print(f"prompt: {prompt}")
            # rank0_print(f"prompt_no_newline: {prompt_no_newline}")
            # rank0_print(f"prompt_same_with_training: {prompt_same_with_training}")
            # -- 算diveristy时带上output -> 和training对齐
            prompt_words = word_tokenize(prompt_same_with_training)
            prompt_words_lst.append(prompt_words)  # 记录tokenize后结果
        return prompt_words_lst
