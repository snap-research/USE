import logging
import os
import csv
import random
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from model import EventBertForMaskedLM
from modeling_retnet import (
    RetNetModelWithLMHead,
)
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BigBirdForMaskedLM,
    DebertaV2ForMaskedLM,
    GPT2LMHeadModel,
)

logger = logging.getLogger()

POOLING_MODES = ["mean", "max", "weighted_mean", "weighted_max"]
TIME_AWARE_METHODS = ["bert"]
LANGUAGE_MODELS = ["bert", "bigbird", "deberta-v2", "gpt2", "rwkv", "retnet"]



pooling_modes = {
    "tf": ["na"],
    "tf-l2": ["na"],
    "tf-idf": ["na"],
    "n-grams": ["na"],
    "sgns": ["mean"],
    "random": ["mean"],
    "bert": ["mean"],
    "gpt2": ["mean"],
    "retnet": ["mean"],
    "rwkv": ["mean"],
}



def make_path(path: str):
    """This function makes a path if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def read_sequences(
    data_dir: str = "./data/train.txt",
):  
    if data_dir.endswith(".txt"):
        with open(data_dir, "r") as f:
            sequences = f.read().splitlines()
        ids = [str(i) for i in range(len(sequences))]
    elif data_dir.endswith(".csv"):
        with open(data_dir, "r") as f:
            data = list(csv.reader(f))
        sequences = [row[0] for row in data]
        ids = [row[1] for row in data]
    else:
        raise ValueError("Invalid file format. Please provide a .txt or .csv file.")

    time_sequences = ["_" for _ in sequences] # a dummy time sequence
    
    return sequences, time_sequences, ids
        

def extract_embeddings(
    event_sequences: List[str],
    time_sequences: List[str],
    method: str,
    modes: List[str],
    max_model_input_size: int,
    model_path: str,
    tokenizer_path: str,
    batch_size: int,
    # use in recurrent embedding generation. Set as a positive number to activate recurent embedding.
    recurrent_input_size: int = -1,
    split_mode: str = None,  # select from [None, rnn, split]
    last_segment_only: bool = False,
):
    """
    This function takes in a list of user event sequences and
    converts them to sentence embeddings based on a specified
    trained transformer-based model.
    """

    if recurrent_input_size > 0:
        assert max_model_input_size % recurrent_input_size == 0

    model = import_trained_model(
        method=method,
        model_path=model_path,
    )

    tokenizer = import_tokenizer(
        tokenizer_path=tokenizer_path,
        max_model_input_size=max_model_input_size,
    )

    n_gpu = torch.cuda.device_count()
    device = "cuda" if n_gpu > 0 else "cpu"
    if n_gpu > 0:
        model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # build data loader to accelerate parallel computation
    use_time = time_sequences and time_sequences[0]
    if use_time:
        data = Dataset.from_dict({"text": event_sequences, "time": time_sequences})
    else:
        data = Dataset.from_dict({"text": event_sequences})
    dataset = DatasetDict()
    dataset["eval"] = data

    def _preprocess_for_masked_language_modeling(raw_sample: dict) -> dict:
        tokenizer_output = tokenizer(
            raw_sample["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=max_model_input_size,
        )

        sample = {
            "input_ids": tokenizer_output["input_ids"],
            "attention_mask": tokenizer_output["attention_mask"],
        }

        if use_time:
            sample["time"] = raw_sample["time"]

        return sample

    num_samples = len(event_sequences)
    # Generate a variable "batches" as a list of batch,
    # Each batch is a list of samples, where each sample is a dictionary:
    # e.g., {"event": "E1 E2 E3", "time": "T1 T2 T3"}
    batches = [
        [
            {"event": event_sequences[j], "time": time_sequences[j]}
            if time_sequences
            else {"event": event_sequences[j]}
            for j in range(i, min(i + batch_size, num_samples))
        ]
        for i in range(0, num_samples, batch_size)  # drop the last batch
    ]

    dataset = dataset.map(
        _preprocess_for_masked_language_modeling,
        batched=True,
        remove_columns=["text"],
        num_proc=16,
    )
    dataset.set_format("torch")

    data_loader = DataLoader(dataset["eval"], batch_size=batch_size * n_gpu)

    aggregated_embeddings = {}
    for mode in modes:
        aggregated_embeddings[mode] = torch.empty(0)

    with torch.no_grad():
        for batch_inputs in tqdm(data_loader, desc="Processing batches"):
            if use_time and method in TIME_AWARE_METHODS:
                batch_inputs["time_ids"] = inputs["time_ids"].to(
                    batch_inputs["input_ids"].device
                )

            if split_mode is None:
                last_layer_embeddings = (
                    model(**batch_inputs).hidden_states[-1].cpu().detach()
                )

            elif split_mode == "state":
                outputs = model(**batch_inputs, forward_impl="chunkwise")
                prev_states = outputs.prev_states[-1]
                last_layer_embeddings = prev_states.mean(dim=1)
                bs = last_layer_embeddings.shape[0]
                last_layer_embeddings = last_layer_embeddings.reshape([bs, 1, -1])

            elif split_mode == "rnn":
                prev_states = []
                num_segment = max_model_input_size // recurrent_input_size
                all_hidden_states = []
                for i in range(num_segment):
                    segment_input_ids = batch_inputs["input_ids"][
                        :, recurrent_input_size * i : recurrent_input_size * (i + 1)
                    ]
                    segment_attention_mask = batch_inputs["attention_mask"][
                        :, recurrent_input_size * i : recurrent_input_size * (i + 1)
                    ]
                    rnn_out = model(
                        input_ids=segment_input_ids,
                        attention_mask=segment_attention_mask,
                        forward_impl="chunkwise",
                        prev_states=prev_states,
                        use_cache=True,
                        sequence_offset=i * recurrent_input_size,
                        output_hidden_states=True,
                    )

                    all_hidden_states.append(rnn_out.hidden_states[-1].cpu().detach())
                    prev_states = rnn_out.prev_states
                all_hidden_states = (
                    all_hidden_states[-1] if last_segment_only else all_hidden_states
                )
                last_layer_embeddings = torch.concat(all_hidden_states, dim=1)
            elif split_mode == "split":
                num_segment = max_model_input_size // recurrent_input_size
                all_hidden_states = []
                for i in range(num_segment):
                    if last_segment_only and (i < num_segment - 1):
                        continue

                    segment_input_ids = batch_inputs["input_ids"][
                        :, recurrent_input_size * i : recurrent_input_size * (i + 1)
                    ]
                    segment_attention_mask = batch_inputs["attention_mask"][
                        :, recurrent_input_size * i : recurrent_input_size * (i + 1)
                    ]
                    split_outputs = (
                        model(
                            input_ids=segment_input_ids,
                            attention_mask=segment_attention_mask,
                        )
                        .hidden_states[-1]
                        .cpu()
                        .detach()
                    )
                    all_hidden_states.append(split_outputs)
                last_layer_embeddings = torch.concat(all_hidden_states, dim=1)
            else:
                raise ValueError("Only support split_node of [None, rnn, split]")

            for mode in modes:
                if mode == "mean":
                    embedding = torch.mean(last_layer_embeddings, dim=1)
                elif mode == "max":
                    embedding = torch.max(last_layer_embeddings, dim=1)[0]
                elif mode == "cls":
                    embedding = last_layer_embeddings[:, 0, :]
                elif mode == "no_pooling":
                    embedding = last_layer_embeddings
                elif mode == "last_seg_mean":
                    embedding = torch.mean(
                        last_layer_embeddings[:, -recurrent_input_size:, :], dim=1
                    )
                elif mode == "last":
                    # find the last non-padding token and take its embedding as sequence embedding
                    sequence_lengths = (
                        torch.eq(batch_inputs["input_ids"], tokenizer.pad_token_id)
                        .long()
                        .argmax(-1)
                        - 2
                    ).to(last_layer_embeddings.device)
                    embedding = last_layer_embeddings[
                        torch.arange(
                            batch_inputs["input_ids"].shape[0],
                            device=last_layer_embeddings.device,
                        ),
                        sequence_lengths,
                    ]
                else:
                    weights_tensor = torch.tensor(
                        get_weights(last_layer_embeddings.shape[1])
                    ).to(last_layer_embeddings.device)
                    weighted_last_layer_embeddings = (
                        last_layer_embeddings * weights_tensor.view(1, -1, 1)
                    )
                    if mode == "weighted_mean":
                        embedding = torch.mean(weighted_last_layer_embeddings, dim=1)
                    elif mode == "weighted_max":
                        embedding = torch.max(weighted_last_layer_embeddings, dim=1)[0]
                aggregated_embeddings[mode] = torch.cat(
                    (aggregated_embeddings[mode], embedding.cpu()), dim=0
                )

    return aggregated_embeddings


def get_data_split_indices(
    labels: list, test_size: float = 0.2, random_state: int = 42
):

    """
    This function splits a list into train and test partitions,
    and returns their respective indices.
    """

    numbers = list(range(len(labels)))
    random.seed(random_state)
    test_ind = random.sample(numbers, k=len(labels) * test_size)
    train_ind = list(set(numbers) - set(test_ind))

    return train_ind, test_ind


def get_weights(num_tokens):
    """
    This function computes weights for the tokens in an event sequence.
    The weights increase linearly from left to right.
    """
    weights = np.arange(1, num_tokens + 1)
    weights = weights / np.sum(weights)
    return weights


def aggregate_token_embeddings_by_mode(embeddings, mode):
    """
    This function takes in a list of token embeddings, and
    aggregates these embeddings into sentence embeddings,
    based on mean, max, weighted_mean, and weighted_max.
    This is used for the baseline models: sgns and random.
    """
    if len(embeddings) > 0:
        if mode == "mean":
            return np.mean(embeddings, axis=0)
        elif mode == "max":
            return np.max(embeddings, axis=0)
        else:
            weights = get_weights(len(embeddings[0]))
            weighted_embeddings = [
                weight * embedding for weight, embedding in zip(weights, embeddings)
            ]
            if mode == "weighted_mean":
                return np.mean(weighted_embeddings, axis=0)
            elif mode == "weighted_max":
                return np.max(weighted_embeddings, axis=0)
    else:
        return None


def aggregate_sgns_embedding(sequence, sgns_model, mode):
    """
    This function takes in an event sequence, a trained sgns model,
    a specified aggregation mode, and return the corresponding
    aggregated sentence embedding.
    """

    if len(sequence.split(" ")) == 0:
        embeddings = np.zeros(sgns_model.vector_size)
        return embeddings

    else:
        embeddings = []
        for token in sequence.split(" "):
            if token in sgns_model.wv.index_to_key:
                embeddings.append(sgns_model.wv[token])
            else:
                embeddings.append(np.zeros(sgns_model.vector_size))

    return aggregate_token_embeddings_by_mode(embeddings, mode)


def create_random_embeddings_dict(sequences, embedding_size=768):
    """
    This function creates a dictionary of random embeddings, based on
    an input list of event sequences.
    """
    unique_tokens = list(
        set(token for sequence in sequences for token in sequence.split(" "))
    )

    random_embeddings_dict = {}
    np.random.seed(42)
    for token in unique_tokens:
        random_embeddings_dict[token] = np.random.uniform(
            low=-0.05, high=0.05, size=embedding_size
        )

    return random_embeddings_dict


def aggregate_random_embedding(sequence, dictionary, mode, embedding_size):
    """
    This function aggregates token-level random embeddings into
    a sentence-level embedding.
    """
    embedding_size = len(dictionary[next(iter(dictionary))])

    if len(sequence.split(" ")) == 0:
        embeddings = np.zeros(embedding_size)
        return embeddings

    else:
        embeddings = []
        for token in sequence.split(" "):
            if token in dictionary:
                embeddings.append(dictionary[token])
            else:
                embeddings.append(np.zeros(embedding_size))

    return aggregate_token_embeddings_by_mode(embeddings, mode)


def trim_event_sequences(
    event_sequences: list, time_sequences: list, max_model_input_size: int
):
    """
    This function trims event sequences into a specified length.
    """
    event_sequences_split_and_trimmed = [
        " ".join(sequence.split(" ")[max_model_input_size - 1 :: -1])
        for sequence in event_sequences
    ]
    time_sequence_split_and_trimmed = [
        sequence[max_model_input_size - 1 :: -1] for sequence in time_sequences
    ]

    return event_sequences_split_and_trimmed, time_sequence_split_and_trimmed


def feature_engineering(
    train_event_sequences: list,
    test_event_sequences: list,
    train_time_sequences: list,
    test_time_sequences: list,
    modes: List[str],
    method: str = "tf",
    max_model_input_size: int = 512,
    model_path: str = None,
    tokenizer_path: str = None,
    n_pca_dims: int = -1,
    batch_size: int = 32,
    # use in recurrent embedding generation. Set as a positive number to activate recurrent embedding.
    recurrent_input_size: int = -1,
    split_mode: str = None,
    last_segment_only: bool = False,
    **kwargs,
):
    """
    This function turns input event sequences into numerical representations.
    """
    assert method in [
        "tf",
        "tf-l2",
        "tf-idf",
        "n-grams",
        "sgns",
        "random",
        "bert",
        "bigbird",
        "deberta-v2",
        "retnet",
        "rwkv",
        "gpt2",
    ]

    # truncate sequences with max_model_input_size
    train_event_sequences = [
        " ".join(seq.split()[-max_model_input_size:]) for seq in train_event_sequences
    ]
    test_event_sequences = [
        " ".join(seq.split()[-max_model_input_size:]) for seq in test_event_sequences
    ]
    train_time_sequences = [seq[-max_model_input_size:] for seq in train_time_sequences]
    test_time_sequences = [seq[-max_model_input_size:] for seq in test_time_sequences]

    train_features = {}
    test_features = {}

    if method == "tf":
        vectorizer = CountVectorizer()
        train_features["na"] = vectorizer.fit_transform(train_event_sequences).toarray()
        if test_event_sequences:
            test_features["na"] = vectorizer.transform(test_event_sequences).toarray()

    elif method == "n-grams":
        vectorizer = CountVectorizer(ngram_range=(1, 4))
        train_features["na"] = vectorizer.fit_transform(train_event_sequences).toarray()
        if test_event_sequences:
            test_features["na"] = vectorizer.transform(test_event_sequences).toarray()

    elif method == "tf-l2":
        vectorizer = TfidfVectorizer(use_idf=False, norm="l2")
        train_features["na"] = vectorizer.fit_transform(train_event_sequences).toarray()
        if test_event_sequences:
            test_features["na"] = vectorizer.transform(test_event_sequences).toarray()

    elif method == "tf-idf":
        vectorizer = TfidfVectorizer()
        train_features["na"] = vectorizer.fit_transform(train_event_sequences).toarray()
        if test_event_sequences:
            test_features["na"] = vectorizer.transform(test_event_sequences).toarray()



    elif method == "random":
        assert all([mode in POOLING_MODES for mode in modes])
        random_embeddings_dict = create_random_embeddings_dict(train_event_sequences)
        embedding_size = len(random_embeddings_dict[next(iter(random_embeddings_dict))])

        for mode in modes:
            train_features[mode] = [
                aggregate_random_embedding(
                    sequence, random_embeddings_dict, mode, embedding_size
                )
                for sequence in tqdm(train_event_sequences)
            ]
            if test_event_sequences:
                test_features[mode] = [
                    aggregate_random_embedding(
                        sequence, random_embeddings_dict, mode, embedding_size
                    )
                    for sequence in tqdm(test_event_sequences)
                ]

    elif method in LANGUAGE_MODELS:

        train_features = extract_embeddings(
            event_sequences=train_event_sequences,
            time_sequences=train_time_sequences,
            method=method,
            modes=modes,
            max_model_input_size=max_model_input_size,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            batch_size=batch_size,
            recurrent_input_size=recurrent_input_size,
            split_mode=split_mode,
        )
        if test_event_sequences:
            test_features = extract_embeddings(
                event_sequences=test_event_sequences,
                time_sequences=test_time_sequences,
                method=method,
                modes=modes,
                max_model_input_size=max_model_input_size,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                batch_size=batch_size,
                recurrent_input_size=recurrent_input_size,
                split_mode=split_mode,
            )

    if n_pca_dims > 0:
        # only perform PCA-based dimension reduction when the target dimension is smaller than
        for mode in train_features.keys():
            if n_pca_dims < len(train_features[mode][0]):
                pca = PCA(n_components=n_pca_dims)
                train_features[mode] = pca.fit_transform(train_features[mode])
                test_features[mode] = pca.transform(test_features[mode])

    if method in ["tf", "tf-l2", "tf-idf"]:
        feature_names = vectorizer.get_feature_names_out().tolist()
    else:
        feature_names = []

    return train_features, test_features, feature_names


def classifier_tune_and_fit(
    classifier: str,
    train_features: list,
    test_features: list,
    train_labels: list,
    test_labels: list,
    tune_n_iter: int,
    n_class: int,
    negative_label: str,
    class_list: list,
    task: str,
    feature_names: list,
    save_partial_dependence_results: bool,
    save_feature_importance_results: bool,
    save_predictions: bool,
):
    """
    This function does hyperparameter tuning for a specified classifier
    using random search and k-fold cross validation, fits the best
    hyperparameter values to the complete data, and returns classification
    results (accuracy, f1, precision, recall, auc). In addition,
    for multiclass prediction, a confusion matrix is returned.
    User can also specify the model to save partial dependence results,
    feature importance scores, and individual predictions.
    """
    assert classifier in ["lr", "mlp"]

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels + test_labels)

    if n_class > 2:
        average_choice = "macro"
        scoring = "f1_macro"
        train_labels = label_encoder.transform(train_labels)
        test_labels = label_encoder.transform(test_labels)
    else:
        average_choice = "binary"
        scoring = "f1"
        train_labels = [0 if label == negative_label else 1 for label in train_labels]
        test_labels_original = test_labels
        test_labels = [0 if label == negative_label else 1 for label in test_labels]
        class_list = [
            class_name for class_name in class_list if class_name != negative_label
        ]

    if classifier == "lr":
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression())]
        )  #

        param_distributions = {
            "clf__penalty": ["l1", "l2", "elasticnet"],
            "clf__C": np.logspace(-4, 4, 20),
            "clf__l1_ratio": np.arange(0.1, 1, 0.1),
            "clf__solver": ["saga"],
            "clf__max_iter": [2000],
            "clf__random_state": [42],
            "clf__class_weight": ["balanced"],
            "clf__n_jobs": [-1],
        }

    else:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier())])  #

        param_distributions = {
            "clf__hidden_layer_sizes": [(100,), (50, 50), (20, 20, 20)],
            "clf__activation": ["tanh", "relu"],
            "clf__solver": ["adam", "sgd"],
            "clf__alpha": np.logspace(-5, 3, 9),
            "clf__learning_rate": ["constant", "invscaling", "adaptive"],
            "clf__max_iter": [2000],
            "clf__random_state": [42],
            "clf__early_stopping": [True, False],
        }

    #         oversampler = RandomOverSampler(
    #             sampling_strategy="not majority", random_state=42
    #         )

    #         train_features, train_labels = oversampler.fit_resample(
    #             train_features, train_labels
    #         )

    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=tune_n_iter,
        cv=5,
        random_state=42,
        n_jobs=-1,
        scoring=scoring,
    )
    random_search.fit(train_features, train_labels)

    best_estimator = random_search.best_estimator_
    best_estimator.fit(train_features, train_labels)

    if save_partial_dependence_results:
        clf = best_estimator.named_steps.clf
        n_features = len(feature_names)
        partial_dependence_df = pd.DataFrame()
        for feature_index in range(n_features):
            partial_dependence_results = partial_dependence(
                clf,
                features=[feature_index],
                X=train_features,
                percentiles=(0, 1),
                grid_resolution=10,
            )

            df_subset = pd.DataFrame()
            df_subset["y"] = partial_dependence_results["average"][0]
            df_subset["x"] = partial_dependence_results["values"][0]
            df_subset["feature"] = feature_names[feature_index]
            partial_dependence_df = pd.concat(
                [partial_dependence_df, df_subset], axis=0, ignore_index=True
            )

    else:
        partial_dependence_df = pd.DataFrame()

    if save_feature_importance_results:
        clf = best_estimator.named_steps.clf
        feature_importances_df = pd.DataFrame()
        for metric in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            feature_importances_results = permutation_importance(
                clf,
                test_features,
                test_labels,
                n_repeats=10,
                random_state=42,
                scoring=metric,
                n_jobs=20,
            )
            feature_importances_results = pd.DataFrame(
                {
                    "importances_mean": feature_importances_results["importances_mean"],
                    "importances_std": feature_importances_results["importances_std"],
                    "metric": metric,
                    "feature": feature_names,
                }
            )

            feature_importances_df = pd.concat(
                [feature_importances_df, feature_importances_results],
                axis=0,
                ignore_index=True,
            )

    else:
        feature_importances_df = pd.DataFrame()

    predictions = best_estimator.predict(test_features)
    pred_probs = best_estimator.predict_proba(test_features)

    if save_predictions and n_class == 2:
        predictions_df = pd.DataFrame(
            {
                "prediction": predictions,
                "pred_prob": pred_probs[:, 1],
                "label": test_labels,
            }
        )

    else:
        predictions_df = pd.DataFrame()

    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average=average_choice
    )

    results_per_class = {}

    # compute auc and class- or label_name-level performance (if relevant)
    if n_class > 2:
        class_counts = Counter(label_encoder.inverse_transform(test_labels))

        auc = roc_auc_score(
            test_labels, pred_probs, average=average_choice, multi_class="ovo"
        )

        precisions, recalls, f1s, _ = precision_recall_fscore_support(
            test_labels, predictions, average=None
        )

        aucs = roc_auc_score(test_labels, pred_probs, multi_class="ovr", average=None)

        class_list = label_encoder.classes_.tolist()

        for class_idx, class_name in enumerate(class_list):
            results_per_class[f"precision_{class_name}"] = precisions[class_idx]
            results_per_class[f"recall_{class_name}"] = recalls[class_idx]
            results_per_class[f"f1_{class_name}"] = f1s[class_idx]
            results_per_class[f"auc_{class_name}"] = aucs[class_idx]
            results_per_class[f"n_{class_name}_for_metrics"] = class_counts[class_name]

        cm = confusion_matrix(test_labels, predictions)
        df_cm = pd.DataFrame(cm, index=class_list, columns=class_list)

    else:
        auc = roc_auc_score(test_labels, pred_probs[:, 1])

        if len(class_list) >= 2:
            # if it's a binary prediction task and there are at least two unique label_names
            # that are not the negative label; allows us to look at label_name level performance

            for class_name in class_list:
                indices_positive_class = [
                    index
                    for index, label in enumerate(test_labels_original)
                    if label == class_name
                ]
                indices_negative_class = [
                    index
                    for index, label in enumerate(test_labels_original)
                    if label == negative_label
                ]
                random.seed(42)
                indices_negative_class = random.sample(
                    indices_negative_class, k=len(indices_positive_class)
                )

                class_pred_prob = [
                    pred_probs[:, 1][index] for index in indices_positive_class
                ] + [pred_probs[:, 1][index] for index in indices_negative_class]

                class_predictions = [
                    predictions[index] for index in indices_positive_class
                ] + [predictions[index] for index in indices_negative_class]

                class_test_labels = [1] * len(indices_positive_class) + [0] * len(
                    indices_negative_class
                )
                (
                    class_precision,
                    class_recall,
                    class_f1,
                    _,
                ) = precision_recall_fscore_support(
                    class_test_labels, class_predictions, average=average_choice
                )

                class_auc = roc_auc_score(class_test_labels, class_pred_prob)

                results_per_class[f"precision_{class_name}"] = class_precision
                results_per_class[f"recall_{class_name}"] = class_recall
                results_per_class[f"f1_{class_name}"] = class_f1
                results_per_class[f"auc_{class_name}"] = class_auc
                results_per_class[f"n_{class_name}_for_metrics"] = len(
                    indices_positive_class
                )

        df_cm = pd.DataFrame()

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    results.update(results_per_class)

    return results, df_cm, partial_dependence_df, feature_importances_df, predictions_df


def split_sequence(event_sequence, time_sequence, max_len=510):
    """
    This function splits an event sequence into multiple chunks,
    if the length of the sequence is at least twice the max_len.
    """
    n = len(event_sequence)
    assert n >= max_len * 2

    event_seqs = []
    time_seqs = []

    i = 0
    while i < n // max_len:

        if i == 0:
            start = 0
            end = max_len
        else:
            start = end
            end = end + max_len

        event = event_sequence[start:end]
        time = time_sequence[start:end]
        event_seqs.append(" ".join(event))
        time_seqs.append(time)

        i = i + 1

    return event_seqs, time_seqs


def get_split_event_sequences_sample(
    event_sequences, time_sequences, max_len=510, sample_size=100
):
    """
    This function takes in a list of event sequences, reads up to sample_size,
    splits these sequences into two or multiple ones, and returns both the split
    sequences and their respective ghost_user_ids.
    """

    split_event_sequences_sample = []
    split_time_sequences_sample = []
    user_id_ls = []

    for i in range(sample_size):
        try:
            split_event_sequences, split_time_sequences = split_sequence(
                event_sequences[i].split(), time_sequences[i], max_len=max_len
            )
            split_event_sequences_sample += split_event_sequences
            split_time_sequences_sample += split_time_sequences
            user_id_ls += [i] * len(split_event_sequences)
        except:
            pass

    return split_event_sequences_sample, split_time_sequences_sample, user_id_ls


def extract_model_name(string):
    """
    This function extract model name based on a string (e.g., model paths).
    """
    if "deberta-v2" in string:
        return "deberta-v2"
    elif "bigbird" in string:
        return "bigbird"
    elif "gpt2" in string:
        return "gpt2"
    elif "rwkv" in string:
        return "rwkv"
    elif "retnet" in string:
        return "retnet"
    else:
        return "bert"


def get_available_tasks(scope: str = "all"):
    """
    This function returns all the downstream tasks we have atm.
    Note that the same function exists in utils_preproccess.py.
    Remember to synchronize the two functions.
    """
    assert scope in ["all", "test", "downstream"]

    all_tasks = [
        "train",
        "test_reported_user_prediction",
        "test_locked_user_prediction",
        "test_account_self_deletion_prediction",
        "test_ad_click_binary_prediction",
        "test_prop_of_ad_clicks_prediction",
        "test_ad_view_time_prediction",
    ]

    if scope == "test":
        all_tasks.remove("train")
    elif scope == "downstream":
        all_tasks.remove("train")
        all_tasks.remove("test_prop_of_ad_clicks_prediction")

    return all_tasks


def import_tokenizer(
    tokenizer_path: str,
    max_model_input_size: int,
):
    """
    This function imports a trained tokenizer from gcs.
    """
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=max_model_input_size,
        trust_remote_code=True,
        )


def import_trained_model(
    method: str,
    model_path: str,
):
    """
    This function imports a trained transformer-based model from gcs.
    """
    # download the model from GCP if not exists locally

    CLASS_MAP = {
        "bert": EventBertForMaskedLM,
        "bigbird": BigBirdForMaskedLM,
        "deberta-v2": DebertaV2ForMaskedLM,
        "retnet": RetNetModelWithLMHead,
        "gpt2": GPT2LMHeadModel,
    }

    if method in CLASS_MAP:
        model = CLASS_MAP[method].from_pretrained(
            model_path, output_hidden_states=True
        )
    else:
        raise Exception(
            "Currently, only bert, bigbird and deberta-v2 are supported."
        )
    return model

def strip_and_split_by_space(string):
    """
    This function strips a string and splits it into a list.
    """
    return string.strip().split(" ")


def truncate_event_and_time_sequence(
    event_sequences,
    time_sequences,
    get_random_segment: bool = False,
    length_to_keep: int = 512,
):

    processed_events = []
    processed_times = []

    # we always skip the first token, since sometimes there is a false positive [New Session] token
    for e, t in zip(event_sequences, time_sequences):
        e = e.split()
        if get_random_segment and len(e) - length_to_keep > 1:
            start = random.randint(1, len(e) - length_to_keep)
        else:
            start = 1

        processed_events.append(" ".join(e[start : start + length_to_keep]))
        processed_times.append(t[start : start + length_to_keep])

    return processed_events, processed_times
