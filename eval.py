import os
import shutil
import pickle
import jsonlines
import torch
from tqdm import tqdm
from vendi_score import vendi

from consts import *
from utils import jload, Prompter, get_tokenizer, get_model


## GENERATE ANSWERS TO BENCH-QUESTIONS
def generate_answers_all_rds_all_tests(schedule, device, rd_start=0, rd_end=10, testsets=["vicuna", "koala", "wizardlm"], 
                                       generate_args={"do_sample": True, "temperature":0.7, "max_new_tokens":1024},  # "top_p":0.95,
                                       ):
    """all rds' resulting models -> generate answers on all testsets"""
    rd_output_dir_root = f"evol_res/{schedule}/output"
    for test in testsets:
        print(f"*** TESTING: {test}======================================================================================================")
        question_file = f"test_data/{test}_test_set.jsonl"
        save_answer_root = f"evol_answer/{test}_test/{schedule}"
        for i in range(rd_start, rd_end+1):
            model_path = f"{rd_output_dir_root}/rd_{i}"
            os.makedirs(save_answer_root, exist_ok=True)
            save_answer_file = f"{save_answer_root}/rd_{i}.jsonl"
            if os.path.exists(save_answer_file):
                # load generated answers
                with jsonlines.open(save_answer_file, "r") as f:
                    num_line = len([_ for _ in f])
                should_num_line = TESTSETS_CONFIGS["should_num_line"]
                if num_line == should_num_line: # all done for rd_model
                    print(f"*** Test -{test} ** Rd -{i} -> **Already Generated!")
                    continue
                else: # start again
                    os.remove(save_answer_file)
            generate_answer(model_path=model_path, device=device, save_answer_file=save_answer_file, question_file=question_file, generate_args=generate_args)
            print(f"*** Test -{test} ** Rd -{i} -> **Generated! -> **Saved to: {save_answer_file}")

def generate_answer(model_path, device, question_file, generate_args, save_answer_file):
    """use model to generate answers to questions"""
    # load tokenizer, model, prompter
    prompter = Prompter(template_name="alpaca", verbose=False) # by default -> alpca-style prompt -> in line with training
    try: # checkpoints save_model() might have error somehow
        tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=model_path, cache_dir=None, model_max_length=512, )
        model = get_model(model_name_or_path=model_path, cache_dir=None)
    except: # use checkpoints from save_model_pretrained() in case
        model_path_save_pretrained = f"{model_path}/pretrained"
        print(f"model_path bad --> {model_path}")
        print(f"switcing to model_path_save_pretrained --> {model_path_save_pretrained}")
        # copy tokenizer configs
        shutil.copy(f"{model_path}/added_tokens.json", f"{model_path_save_pretrained}/")
        shutil.copy(f"{model_path}/special_tokens_map.json", f"{model_path_save_pretrained}/")
        shutil.copy(f"{model_path}/tokenizer_config.json", f"{model_path_save_pretrained}/")
        shutil.copy(f"{model_path}/tokenizer.model", f"{model_path_save_pretrained}/")
        shutil.copy(f"{model_path}/trainer_state.json", f"{model_path_save_pretrained}/")
        shutil.copy(f"{model_path}/training_args.bin", f"{model_path_save_pretrained}/")
        tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=model_path_save_pretrained, cache_dir=None, model_max_length=512, )  # todo.
        model = get_model(model_name_or_path=model_path_save_pretrained, cache_dir=None)
    model = model.to(f"cuda:{device}")
    # load questions
    questions = []
    with jsonlines.open(question_file, 'r') as f:
        for line in f:
            questions.append(line)
    # generating answers --- 
    i_temp = 0
    for q in tqdm(questions, total=len(questions)):
        i_temp += 1
        if "vicuna" in question_file:
            instruction = q["text"]
        elif "koala" in question_file:
            instruction = q["prompt"]
        elif "wizardlm" in question_file:
            instruction = q["Instruction"]
        prompt = prompter.generate_prompt(instruction=instruction)
        # formatting in line with training inputs:
        prompt_no_newline = prompt.strip()
        # prompt_same_with_training = " " + prompt_no_newline.replace("\n\n", "\n")
        prompt_same_with_training = " " + prompt_no_newline
        # sanity check
        if i_temp == 1:
            print(f"Model Inference Input Text - same with training: {prompt_same_with_training}")
        tokenized_input = tokenizer(prompt_same_with_training, return_tensors="pt", add_special_tokens=True)
        generated = model.generate(tokenized_input["input_ids"].to(f"cuda:{device}"), **generate_args)
        answer_with_prompt = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = answer_with_prompt.split("### Response:")[1]
        q["answer"] = answer
        # save answers
        with jsonlines.open(save_answer_file, "a") as f:
            f.write(q)

 
## CALCULATE DATASET DIVERSITY (Vendi-Score)
def analyse_diversity_results_all_rds(schedule, rd_start, rd_end, device, measures, embed_model_path,):  
    # load embedding model & tokenizer
    model = get_model(model_name_or_path=embed_model_path, cache_dir=None)
    tokenizer, special_tokens_dict = get_tokenizer(model_name_or_path=embed_model_path, cache_dir=None, model_max_length=512, )
    model.to(f"cuda:{device}")
    print(f"Using Embedding-Model -> Loaded from: {embed_model_path}")
    rd_to_vendi = {}
    # iterate over round selected data
    for rd in range(rd_start, rd_end+1):
        rd_labeled_data_file = f"evol_res/{schedule}/data/rd_{rd}_labeled.json"
        diversity_res = calculate_data_diversity_from_data_file(labeled_data_file=rd_labeled_data_file,
                                                                measures=measures,
                                                                model=model,
                                                                tokenizer=tokenizer,
                                                                )
        rd_to_vendi[rd] = diversity_res["vendi"]
    return {"vendi": rd_to_vendi,}

def calculate_data_diversity_from_data_file(labeled_data_file, measures, model, tokenizer, ):
    # load data
    print(f"Loading labeled data from: --> {labeled_data_file}")
    data = jload(labeled_data_file)
    # load prompter
    prompter = Prompter(template_name="alpaca", verbose=False)
    # compute embed
    embeddings = []
    for i in tqdm(range(len(data)), total=len(data)):
        d = data[i]
        # generate prompt -> including output as part of the training data 
        prompt = prompter.generate_prompt(
                                            instruction=d["instruction"],
                                            input=d["input"], 
                                            label=d["output"],  # calculate diversity: including output as part of the training data 
                                            )
        # formatting in line with training inputs:
        prompt_no_newline = prompt.strip()
        # prompt_same_with_training = " " + prompt_no_newline.replace("\n\n", "\n")
        prompt_same_with_training = " " + prompt_no_newline

        # infer -> get avg_pooled embeddings
        with torch.no_grad():
            tokenized_input = tokenizer(prompt_same_with_training, return_tensors="pt", add_special_tokens=True)
            tokenized_input = {k:v.to(model.device) for k,v in tokenized_input.items()}                
            model.eval()
            result = model(**tokenized_input, return_dict=True, output_hidden_states=True)
            hidden_states = result["hidden_states"]
            last_layer_hidden_states = hidden_states[-1]  # (batch_size=1, seq_len, hidden_dim=4096)
            last_hidden_states_avg = torch.mean(last_layer_hidden_states.squeeze(0), dim=0)  # -> (hidden_dim)
            embeddings.append(last_hidden_states_avg)            
    embeddings = torch.stack(embeddings)
    print(f"Embeddings.shape = {embeddings.shape}")

    # compute diversity measures
    diversity_res = {}
    for measure in measures:
        if measure=="vendi": # vendi score
            emb_norm = embeddings / torch.norm(embeddings, dim=-1, keepdim=True) 
            similarity = torch.mm(emb_norm, emb_norm.T) 
            sim_K = similarity.cpu().numpy()
            vendi_diversity_score = vendi.score_K(sim_K)
            diversity_res["vendi"] = vendi_diversity_score
    return diversity_res
    



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_stage', type=str, required=True,
                        help="which evaluation function? {generate_answer, analyse_diversity}.")
    parser.add_argument('--schedule', type=str,  required=True,
                        help="result_dir_name. name of the folder containing all results from the DiverseEvol run.")
    parser.add_argument('--device', type=int, default=0,
                        help="gpu-idx.")
    parser.add_argument('--rd_start', type=int, default=0,
                        help="evaluation from which round's results.")
    parser.add_argument('--rd_end', type=int, default=10,
                        help="evaluation till which round's results.")
    parser.add_argument('--testsets', nargs='+', type=str, default=["vicuna", "koala", "wizardlm"],
                        help="generate answers to which testsets?")
    parser.add_argument('--embed_model_path', type=str, default=None, 
                        help="path to the foundation model. its embedding is used to calculated vendi-score.")
    args = parser.parse_args()
    print(args)

    if args.eval_stage == "generate_answer":
        generate_answers_all_rds_all_tests(schedule=args.schedule,
                                           device=args.device,
                                           rd_start=args.rd_start,
                                           rd_end=args.rd_end,
                                           testsets=args.testsets,
                                           generate_args={"do_sample": True, "temperature":0.7, "max_new_tokens":1024}) # "top_p":0.95

    elif args.eval_stage == "analyse_diversity":
        schedule_to_rd_diversity_res = {}
        print(f"** calculating diversity of -> {args.schedule} ...")
        if "baseline" in args.schedule:
            rd_end_temp = 0
            measures = ["vendi"]
        else:
            rd_end_temp = args.rd_end
            measures = ["vendi"]
        save_diversity_results_file = f"evol_diversity/{args.schedule}_RD={args.rd_start}-{rd_end_temp}_MEASURES={'_'.join(measures)}.pkl"  # path to saving calculated diversity results
        if os.path.exists(save_diversity_results_file):
            with open(save_diversity_results_file, "rb") as f:
                diversity_results = pickle.load(f)
            print(f"Loading diversity_results from: <-- {save_diversity_results_file}")
        else:
            diversity_results = analyse_diversity_results_all_rds(schedule=args.schedule,
                                                                  rd_start=args.rd_start,
                                                                  rd_end=rd_end_temp,
                                                                  device=args.device,
                                                                  measures=measures,
                                                                  embed_model_path=args.embed_model_path, # always use the same foundation model
                                                                  )
            with open(save_diversity_results_file, "wb") as f:
                pickle.dump(diversity_results, f)
            print(f"Saving diversity_results to: --> {save_diversity_results_file}")  # saving diversity_results
            schedule_to_rd_diversity_res[args.schedule] = diversity_results
            print(f"** schedule: {args.schedule} - rd_to_diversity_res --> {diversity_results}")  # printing diversity_results

