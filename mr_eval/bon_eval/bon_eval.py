import pandas as pd
import numpy as np
import random
from accelerate import Accelerator
import torch
import os
from tqdm import tqdm
from copy import deepcopy
from collections import Counter
import time

from prm_eval_utils import *

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def compute_metrics(dataset_name, scored_results, key_list):
    
    sample_nums = [1, 2, 4, 8, 16, 32, 64]

    metrics = {"w/o sc":{}, "w/ sc":{}, "sc": {}, "pass@k": {}}

    for use_sc in [False, True]:
        sc_key = "w/ sc" if use_sc else "w/o sc"
        for n in sample_nums:
            splitted_completions = split_query(scored_results, n, max(sample_nums))
            if not args.baseline:
                for kind in key_list:
                    if not use_sc:
                        selected_completions = best_of_n(splitted_completions,kind)
                    else:
                        selected_completions = []
                        for comps in splitted_completions:
                            selected_completions += comps
                    acc_list = [comp["correctness"] for comp in selected_completions]
                    output_list = [comp["extracted_output"] for comp in selected_completions]

                    if not use_sc:
                        acc = sum(acc_list) / len(acc_list)
                        metrics[sc_key][n] = round(acc * 100, 1)
                    else:
                        correct,sumv = 0,0
                        for ii in range(len(splitted_completions)):
                            answer_dict = {k:0 for k in set(output_list[ii*n:(ii+1)*n])} # collect answers for this prompt
                            reward_list = [ele[kind] for ele in selected_completions[ii*n:(ii+1)*n]] # corresponding rewards
                            for ele,reward in zip(output_list[ii*n:(ii+1)*n],reward_list):
                                if "implicit_prm" in args.type:
                                    answer_dict[ele]+=torch.sigmoid(torch.tensor(reward)).item() # should we do sigmoid or not?
                                else:
                                    answer_dict[ele]+=torch.tensor(reward).item()
                            answer_dict = dict(sorted(answer_dict.items(), key=lambda x: x[1], reverse=True))
                            select_answer = max(answer_dict, key=answer_dict.get)
                            is_correct = select_answer == splitted_completions[ii][0]['reference'] # we have preprocessed the outputs: if correct, output==gt
                            correct += is_correct
                            sumv+=1

                        acc = correct/sumv

                    metrics[sc_key][n] = round(acc * 100, 1)
    
    # baselines
    for n in sample_nums:
        splitted_completions = split_query(scored_results, n, max(sample_nums))
        selected_completions = []
        for comps in splitted_completions:
            selected_completions += comps
        
        acc_list = [comp["correctness"] for comp in selected_completions]
        output_list = [comp["extracted_output"] for comp in selected_completions]
        
        total_index = int(len(acc_list) / n)

        pass_k = sum([1 for ii in range(total_index) if any(acc_list[ii*n:(ii+1)*n])])/total_index
        consistent_outputs = [Counter(output_list[ii*n:(ii+1)*n]).most_common(1)[0][0] for ii in range(total_index)]  # (num_instructions, )
        position_of_consistent_outputs = [output_list[ii*n:(ii+1)*n].index(consistent_outputs[ii]) for ii in range(total_index)]  # (num_instructions, )
        acc_of_consistency = [acc_list[ii*n:(ii+1)*n][idx_of_split] for ii, idx_of_split in enumerate(position_of_consistent_outputs)]
        sc = sum(acc_of_consistency)/total_index

        metrics["sc"][n] = round(sc * 100, 1)
        metrics["pass@k"][n] = round(pass_k * 100, 1)

    return metrics



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--baseline", type=int, default=0)
    parser.add_argument("--combine", type=int, default=0)
    parser.add_argument("--type", type=str, default='implicit_prm',choices=['implicit_prm','baseline-value-head','baseline-ntp','implicit_prm-orm'])

    parser.add_argument("--begin-of-action-token", type=str, default='')
    parser.add_argument("--prm-token", type=str, default=None)
    parser.add_argument("--reward-integration", type=str, default='min',choices=['min','sum','mean'])

    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--ref-tokenizer-path", type=str, default=None)
    parser.add_argument("--ref-load", type=str,default=None)
    parser.add_argument("--config-load", type=str, default=None)

    parser.add_argument("--bon-dataset", type=str,default="math",choices=['math'])
    parser.add_argument("--save-dir", type=str,default="./output_rewards_final")

    args = parser.parse_args()
    print(args)
    accelerator = Accelerator()

    
    file_list, origin_dataset = get_raw_data(args.bon_dataset)
    
    tokenizer, ref_tokenizer = get_tokenizer(args.load, args.ref_load)

    begin_of_action_token = args.begin_of_action_token
    prm_token = args.prm_token if args.prm_token else ""
    good_token, bad_token = "+", "-"
    prm_token_id, good_token_id, bad_token_id = set_special_token_ids(prm_token, good_token, bad_token, tokenizer)

    if accelerator.is_main_process:
        print('PRM_token:',prm_token,prm_token_id, 'Good token, Bad token:',good_token,good_token_id,bad_token,bad_token_id)


    model, ref_model, ref_logits_path_list = init_ds_models(args.type, args.load, args.ref_load, args.bon_dataset)

    special_ids = [tokenizer.encode('\nStep', add_special_tokens=False)[-1],
                   tokenizer.encode(' Step', add_special_tokens=False)[-1],
                   tokenizer.encode('Step 2:', add_special_tokens=False)[-1]] # [Step, _Step, :]
    prm_special_ids = {"prm_token_id":prm_token_id, "good_token_id":good_token_id, "bad_token_id":bad_token_id}

    for file_index, file_name in enumerate(file_list):
        queries = load_data(file_name, origin_dataset)
        if accelerator.is_main_process:
            print('Current evaluated file:',file_name)

        random.seed(0)
        dataloader, ref_dataloader = get_dataloader(args.type, queries, args.batch_size, tokenizer, ref_tokenizer, special_ids, prm_special_ids, accelerator)

        dataloader = devide_dataloader_to_devices(dataloader, accelerator, args.local_rank)
        ref_dataloader = devide_dataloader_to_devices(ref_dataloader, accelerator, args.local_rank) if ref_dataloader != None else None

        save_name = args.load.split('/')[-1] if args.load.split('/')[-1]!='' else args.load.split('/')[-2]
        output_file = os.path.join(args.save_dir,f"scored_{file_name.split('/')[-1][:-5]}-{save_name if 'orm' not in args.type else save_name+'-orm'}.json")
        os.makedirs(args.save_dir, exist_ok=True)
        if accelerator.is_local_main_process:
            print("output_file:", output_file)
        if not args.baseline and not os.path.exists(output_file):
            if 'implicit_prm' in args.type:
                if accelerator.is_local_main_process:
                    print("ref_file:", ref_logits_path_list[file_index])

                if os.path.exists(ref_logits_path_list[file_index].replace(".json", f"-{accelerator.device}.pickle")): # already have one; directly load
                    all_ref_logits = torch.load(ref_logits_path_list[file_index].replace(".json", f"-{accelerator.device}.pickle"))
                    all_ref_logits = [torch.tensor(ref_per_token_logps).to(accelerator.device) for ref_per_token_logps in all_ref_logits]
                elif ref_dataloader: # need to compute ref logits
                    all_ref_logits = []
                    with torch.no_grad():
                        for inputs in tqdm(ref_dataloader, desc="ref forward"):
                            ref_per_token_logps = get_logps(ref_model,inputs,args.type)
                            all_ref_logits.append(ref_per_token_logps)
                    try:
                        all_ref_logits_for_save = [ref_per_token_logps.tolist() for ref_per_token_logps in deepcopy(all_ref_logits)]
                        torch.save(all_ref_logits_for_save, ref_logits_path_list[file_index].replace(".json", f"-{accelerator.device}.pickle"))#, pickle=True)
                    except Exception as e:
                        print(e)
                        pass
                print(f"{len(all_ref_logits)=}")

            print(f"{len(dataloader)=}")

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for batch_id, inputs in tqdm(enumerate(dataloader), desc="policy forward"):
                
                if "implicit_prm" in args.type:
                    batch_rewards, reward_idxes = get_reward(model, inputs, args.type, accelerator, ref_per_token_logps=all_ref_logits[batch_id])
                else:
                    batch_rewards, reward_idxes = get_reward(model, inputs, args.type, accelerator, good_token_id=good_token_id, bad_token_id=bad_token_id)

                batch_rewards, queries = manipulate_rewards(batch_rewards, queries, reward_idxes, accelerator)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if accelerator.is_main_process:
                print(queries[0])
                print('Save file at:',output_file)
                print('Spending Time:', elapsed)
                pd.DataFrame(queries).to_json(output_file, indent=4, orient="records")
        elif os.path.exists(output_file):
            import pandas as pd
            if accelerator.is_local_main_process:
                print("load saved rewards")
            queries = pd.read_json(output_file).to_dict("records")

        results = {}
        for ref_setup in queries[0]["reward"].keys():
            for beta_method in queries[0]["reward"][ref_setup].keys():
                for reward_approach in queries[0]["reward"][ref_setup][beta_method]:
                    for method in ["min", "sum"]:
                        for query in queries:
                            query[f"{ref_setup}-{beta_method}-{reward_approach}-{method}-reward"] = query["reward"][ref_setup][beta_method][reward_approach][method]
                        results[f"{ref_setup}-{beta_method}-{reward_approach}-{method}"] = compute_metrics(args.bon_dataset, queries, [f"{ref_setup}-{beta_method}-{reward_approach}-{method}-reward"])
                        for query in queries:
                            del query[f"{ref_setup}-{beta_method}-{reward_approach}-{method}-reward"]
                        if accelerator.is_main_process:
                            print(f"{ref_setup}-{beta_method}-{reward_approach}-{method}:", results[f"{ref_setup}-{beta_method}-{reward_approach}-{method}"])

        if accelerator.is_main_process:
            os.makedirs("metrics", exist_ok=True)
            pd.DataFrame(results).to_json(f"metrics/{file_name.split('/')[-1][:-5]}-{save_name if 'orm' not in args.type else save_name+'-orm'}.json", orient="records", lines=True)
            f = open(f"metrics/{file_name.split('/')[-1][:-5]}-{save_name if 'orm' not in args.type else save_name+'-orm'}.txt", "w")
            for k, v in results.items():
                print(f"{k}: {v}")