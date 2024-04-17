import csv
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from utils import feature_engineering, pooling_modes


MAX_LEN = 4000
DAY_LEN = 250

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--method_type", type=str, default="retnet")
    argparser.add_argument("--model_path", type=str, default="")
    argparser.add_argument("--tokenizer_path", type=str, default="./tokenizer/dna")
    argparser.add_argument("--dataset_dir", type=str, default="./data/dna_test_5k.csv")
    argparser.add_argument("--batch_size", type=int, default=16)
    args = argparser.parse_args()
    
    # get data
    with open(args.dataset_dir, "r") as f:
        raw_data = list(csv.reader(f, delimiter="\t"))[1:]
        print(f"Number of users: {len(raw_data)}")
        
        user_history = [d[0] for d in raw_data]
        user_context = [d[1] for d in raw_data]
    
    ori_batch_size = args.batch_size
    method_type = args.method_type
    
    modes = pooling_modes[method_type]
    history_embeddings, _, _ = feature_engineering(train_event_sequences = user_history,
                                                    test_event_sequences = [],
                                                    train_time_sequences = [],
                                                    test_time_sequences = [],
                                                    method = method_type,
                                                    modes = modes,
                                                    max_model_input_size = MAX_LEN,
                                                    model_path = args.model_path,
                                                    tokenizer_path = args.tokenizer_path,
                                                    batch_size = ori_batch_size,
                                                    recurrent_input_size = 500 if method_type in ["bert", "gpt2"] else -1,
                                                    split_mode = "split" if method_type in ["bert", "gpt2"] else None,
                                                    last_segment_only = False,
                                                    )
    for split_mode in ["split", None]:
        if split_mode == None and method_type != "retnet":
            continue
        
        context_embeddings, _, _ = feature_engineering(train_event_sequences = user_context,
                                                test_event_sequences = [],
                                                train_time_sequences = [],
                                                test_time_sequences = [],
                                                method = method_type,
                                                modes = ["no_pooling"],
                                                max_model_input_size = MAX_LEN,
                                                model_path = args.model_path,
                                                tokenizer_path = args.tokenizer_path,
                                                batch_size = ori_batch_size,
                                                recurrent_input_size = DAY_LEN,
                                                split_mode = split_mode,
                                                last_segment_only = False,
                                                )
        context_embeddings = context_embeddings["no_pooling"]
        
        for last_segment_only in [True, False]:
            if split_mode == None and last_segment_only == True:
                continue
                
            for cur_len in range(DAY_LEN, MAX_LEN+1, DAY_LEN):

                def get_rank(x):
                    vals = x[range(len(x)), range(len(x))]
                    return (x > vals[:, None]).long().sum(1) + 1

                for mode in modes:
                    embeddings_query = context_embeddings[:, cur_len-DAY_LEN:cur_len, :].mean(axis=1) if last_segment_only else context_embeddings[:, :cur_len, :].mean(axis=1)
                    embeddings_candidate = history_embeddings[mode]
                    embeddings_query = torch.tensor(embeddings_query) if type(embeddings_query) == list else torch.tensor(np.array(embeddings_query))
                    embeddings_candidate = torch.tensor(embeddings_candidate) if type(embeddings_candidate) == list else torch.tensor(np.array(embeddings_candidate))
                    embeddings_query = F.normalize(embeddings_query.float())
                    embeddings_candidate = F.normalize(embeddings_candidate.float())
                    feature_dim = embeddings_query.shape[-1]

                    similarities = torch.einsum('ih,jh->ij', embeddings_query, embeddings_candidate) # [num_sample, num_sample] 
                    ranks = get_rank(similarities)

                    mrr = torch.mean(1.0 / ranks)
                    top_1_acc = ((ranks <= 1).sum() / len(ranks)).item()
                    top_3_acc = ((ranks <= 3).sum() / len(ranks)).item()
                    top_5_acc = ((ranks <= 5).sum() / len(ranks)).item()
                    top_10_acc = ((ranks <= 10).sum() / len(ranks)).item()

                    message = f"{method_type}_{mode}_{cur_len}_{split_mode}_{last_segment_only}: mrr {mrr}, top1 {top_1_acc}, top3 {top_3_acc}, top5 {top_5_acc}, top10 {top_10_acc}"
                    print(message)


if __name__ == "__main__":
    main()