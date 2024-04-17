import csv
import argparse
import torch.nn.functional as F
import numpy as np
from utils import feature_engineering
from eval_user import pooling_modes
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MAX_LEN = 4000
DAY_LEN = 250
RANDOM_SEEDS = [0,1,2]

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--method_type", type=str, default="retnet")
    argparser.add_argument("--model_path", type=str, default="")
    argparser.add_argument("--tokenizer_path", type=str, default="./tokenizer/dna")
    argparser.add_argument("--dataset_dir", type=str, default="./data/dna_test_5k.csv")
    argparser.add_argument("--batch_size", type=int, default=16)
    args = argparser.parse_args()
    
    ori_batch_size = args.batch_size
    method_type = args.method_type
    
    # get data
    with open(args.dataset_dir, "r") as f:
        raw_data = list(csv.reader(f, delimiter="\t"))[1:]
        print(f"Number of users: {len(raw_data)}")
        
        user_history = [d[0] for d in raw_data]
        user_context = [d[1] for d in raw_data]
        
        labels_train = [int(i) for i in range(len(raw_data))]
        labels_eval = [int(i) for i in range(len(raw_data))]
    

    for seed in RANDOM_SEEDS:
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
        
        mode = modes[0]
        embeddings = history_embeddings[mode]
        X_train, y_train = embeddings, labels_train
        
        mlpclass = MLPClassifier(
                    hidden_layer_sizes = (1024,),
                    learning_rate = "adaptive",
                    alpha = 0.01,
                    max_iter = 3000,
                    random_state = seed,
                    early_stopping=False,
                    n_iter_no_change=20,
                    validation_fraction=0.1,
                )
        mlp = Pipeline([("scaler", StandardScaler()), ("clf", mlpclass)])  #

        mlp.fit(X_train, y_train)
        print("MLP is trained")

        for eval_mode in ["split", None]:
            if eval_mode == None and method_type != "retnet":
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
                                                    split_mode = eval_mode,
                                                    last_segment_only = False,
                                                    )
            context_embeddings = context_embeddings["no_pooling"]
            
            for last_segment_only in [True, False]:
                if eval_mode == None and last_segment_only == True:
                    continue
            
                for cur_len in range(DAY_LEN, MAX_LEN+1, DAY_LEN):
                    for mode in modes:
                        if last_segment_only:
                            X_test = context_embeddings[:, cur_len-DAY_LEN:cur_len, :].mean(axis=1)
                        else:
                            X_test = context_embeddings[:, :cur_len, :].mean(axis=1)
                        y_pred = mlp.predict(X_test)
                        y_logits = mlp.predict_proba(X_test)
                        y_test = labels_eval
                        
                        auc = roc_auc_score(y_test, y_logits, multi_class="ovr")
                        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                        print(f"{method_type}: auc: {auc}, f1: {f1}")
                        

if __name__ == "__main__":
    main()