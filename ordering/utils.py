import os
import sys
import time
import json
import random

import torch
import scipy
import scipy.special
import scipy.stats

import numpy as np
import sklearn.metrics
from tqdm import tqdm

from torchmetrics.classification import MulticlassCalibrationError
from torchmetrics.classification import BinaryCalibrationError

from data_utils import template_test
from data_utils import template_null
from data_utils import template_demonstration
from data_utils import get_label_mapping
from data_utils import get_permutations

class AutoregressiveDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, tokenizer):
        self.data = input_data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        tokenized_input = self.tokenizer(row, return_tensors="pt")
        return {
            "input_ids": tokenized_input.input_ids.squeeze(dim=0),
            "attention_mask": tokenized_input.attention_mask.squeeze(dim=0)
        }

def make_inference(model, input_ids, label_idx=None, device="cpu"):
    with torch.no_grad():
        output = model(
            input_ids=input_ids.to(device),
            return_dict=True,
            output_hidden_states=True
        )
    logits = output.logits # (bz, seq_len, vocab_size)
    logits = logits[:,-1,:].to('cpu').squeeze(dim=0).to(torch.float32)[label_idx]
    last_hidden_states = output["hidden_states"][-1] # (bz, seq_len, hidden_dim)

    return logits, last_hidden_states[:, -1, :].cpu() # (1, hidden_dim)
    # return softmax(logits)[label_idx]

def make_inference_perplexity(model, input_ids, tokenized_labels, device="cpu"):
    if 'facebook/opt' in model.name_or_path:
        tokenized_labels = tokenized_labels[:, 1:] # remove the bos token
    elif 'llama' in model.name_or_path:
        tokenized_labels = tokenized_labels[:, 2:] # remove bos and pad token
    elif 'EleutherAI' in model.name_or_path or 'gpt2' in model.name_or_path:
        pass
    perplexity = []
    last_hidden_states = []
    loss_fct = torch.nn.CrossEntropyLoss()

    # try to batch operation
    labels = torch.concat((input_ids.repeat(tokenized_labels.shape[0], 1), tokenized_labels), dim=1) # (num_labels, seq_len+label_len)
    # TODO: replace hardcoded padding
    labels[:, :input_ids.shape[1]] = -100
    labels[labels==2] = -100

    with torch.no_grad():
        output = model(
            input_ids=torch.concat((input_ids.repeat(tokenized_labels.shape[0], 1), tokenized_labels), dim=1),
            output_hidden_states=True,
            return_dict=True
        ) # logits.shape (num_labels, seq_len, |V|)
        logits = output.logits[:, :-1, :]
        labels = labels[:, 1:]
        for i in range(labels.shape[0]):
            loss = loss_fct(logits[i, :, :].to(torch.float32), labels[i, :])
            perplexity.append(-loss.cpu().item())
            last_hidden_states.append(output["hidden_states"][-1][i, -1, :].cpu())

    # for i in range(tokenized_labels.shape[0]):
    #     labels = torch.concat((input_ids, tokenized_labels[i, :].unsqueeze(dim=0)), dim=1) # (1, seq_len+label_len)
    #     labels[:, :input_ids.shape[1]] = -100
    #     labels[labels==2] = -100
    #     with torch.no_grad():
    #         output = model(
    #             input_ids=torch.concat((input_ids, tokenized_labels[i, :].unsqueeze(dim=0)), dim=1).to(device),
    #             labels=labels.to(device),
    #             output_hidden_states=True,
    #             return_dict=True
    #         )
    #     perplexity.append(-output.loss.cpu().item())
    #     last_hidden_states.append(output["hidden_states"][-1][:, -1, :].cpu()) # (1, hidden_dim)

    return torch.tensor(perplexity), last_hidden_states[-1]

def get_label_distribution(testset, args):
    id2label, labels, label_mapping = get_label_mapping(args.dataset)
    distribution = torch.zeros(len(labels))
    for i, row in enumerate(testset):
        label = label_mapping[row["label"]]
        distribution[labels.index(label)] += 1.
    return distribution/distribution.sum().tolist()


def globelE_prompting(tokenizer, model, trainset, testset, args, sample_ids=None, device="cpu", logger=None):
    # TODO implement
    model.eval()
    permutations = get_permutations(sample_ids, num_samples=24)
    id2label, labels, label_mapping = get_label_mapping(args.dataset)
    tokenized_labels = tokenizer(labels, return_tensors='pt', padding=True)
    if 'facebook/opt' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt").input_ids[:, 1].view(-1)
    elif 'llama' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt", padding=True).input_ids[:, 2].view(-1)
    elif 'EleutherAI' in model.name_or_path or 'gpt2' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt").input_ids.view(-1)
    else:
        print("using perplexity scoring!")
    if 'openllama' in model.name_or_path or 'llama' in model.name_or_path:
        perplexity_scoring=False
    else:
        perplexity_scoring=False

    softmax = torch.nn.Softmax(dim=0)
    null_input = template_null(args.dataset)
    null_logits_holder = []
    for i, permutation in enumerate(permutations):
        prompt = [f"{template_demonstration(args.dataset, trainset[sample_id]['sentence'], trainset[sample_id]['label'])}" for sample_id in permutation]
        if perplexity_scoring:
            logits, _ = make_inference_perplexity(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
        else:
            logits, _ = make_inference(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, label_idx, device=device)
        logits = softmax(logits)
        null_logits_holder.append(logits.tolist())

    uncalibrated_acc_holder = []
    calibrated_acc_holder = []
    uncalibrated_ece_holder = []
    calibrated_ece_holder = []
    uncalibrated_predicted_distribution_holder = []
    calibrated_predicted_distribution_holder = []
    uniform_uncalibrated_kl_holder = []
    uniform_calibrated_kl_holder = []

    accuracy_holder, f1_holder, ece_holder, entropy_holder = [], [], [], []
    for i, permutation in enumerate(permutations):
        prompt = [f"{template_demonstration(args.dataset, trainset[sample_id]['sentence'], trainset[sample_id]['label'])}" for sample_id in permutation]
        uncalibrated_prediction_logits, calibrated_prediction_logits = [], []
        references = []
        uncalibrated_predictions = []
        calibrated_predictions = []
        print(f"\ncurrent permutation id -> {i}")
        print(f"current permutation -> {permutation}")
        null_logits = null_logits_holder[i]
        for j, row in tqdm(enumerate(testset), desc='running GlobalE...'):
            test_instance = template_test(args.dataset, row['sentence'], label_mapping[row['label']])
            test_instance = f"".join(prompt)+f"{test_instance}"
            tokenized_input = tokenizer(test_instance, return_tensors='pt', padding=False, truncation=True, max_length=1024) 
            if perplexity_scoring:
                uncalibrated_logits, _ = make_inference_perplexity(model, tokenizer(test_instance, return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
            else:
                uncalibrated_logits, _ = make_inference(model, tokenizer(test_instance, return_tensors='pt').input_ids, label_idx, device=device)
            uncalibrated_logits = softmax(uncalibrated_logits)

            if args.calibration_mode == 'pmi':
                calibrated_logits = uncalibrated_logits/torch.tensor(null_logits)
            elif args.calibration_mode == 'contextual':
                W = (torch.ones_like(uncalibrated_logits)/uncalibrated_logits.shape[0])/torch.tensor(null_logits)
                calibrated_logits = W * uncalibrated_logits
            calibrated_logits = softmax(calibrated_logits)
            uncalibrated_prediction_logits.append(uncalibrated_logits.tolist())
            calibrated_prediction_logits.append(calibrated_logits.tolist())
            references.append(labels.index(label_mapping[row['label']]))

        uniform = (np.ones(len(labels))/len(labels)).tolist()
        # uncalibrated_predicted_distribution = (np.bincount(np.argmax(uncalibrated_prediction_logits, axis=1))/len(uncalibrated_prediction_logits)).tolist()
        # calibrated_predicted_distribution = (np.bincount(np.argmax(calibrated_prediction_logits, axis=1))/len(calibrated_prediction_logits)).tolist()

        uncalibrated_predicted_distribution = [0 for i in range(len(labels))]
        for i in np.argmax(uncalibrated_prediction_logits, axis=1):
            uncalibrated_predicted_distribution[i] += 1
        uncalibrated_predicted_distribution = [i/sum(uncalibrated_predicted_distribution) for i in uncalibrated_predicted_distribution]
        calibrated_predicted_distribution = [0 for i in range(len(labels))]
        for i in np.argmax(calibrated_prediction_logits, axis=1):
            calibrated_predicted_distribution[i] += 1
        calibrated_predicted_distribution = [i/sum(calibrated_predicted_distribution) for i in calibrated_predicted_distribution] 

        uncalibrated_predicted_distribution_holder.append(scipy.stats.entropy(uncalibrated_predicted_distribution, uniform))
        calibrated_predicted_distribution_holder.append(scipy.stats.entropy(calibrated_predicted_distribution, uniform))
        uniform_uncalibrated_kl_holder.append(np.mean(scipy.stats.entropy(uncalibrated_prediction_logits, uniform)))
        uniform_calibrated_kl_holder.append(np.mean(scipy.stats.entropy(calibrated_prediction_logits, uniform)))

        ece_metric = MulticlassCalibrationError(num_classes=len(uncalibrated_prediction_logits[0]), n_bins=100, norm="l1")
        uncalibrated_ece_holder.append(ece_metric(torch.tensor(uncalibrated_prediction_logits), torch.tensor(references)).item())
        calibrated_ece_holder.append(ece_metric(torch.tensor(calibrated_prediction_logits), torch.tensor(references)).item())
        print(f"uncalibrated ece -> {ece_metric(torch.tensor(uncalibrated_prediction_logits), torch.tensor(references)):.3f}")
        print(f"calibrated ece -> {ece_metric(torch.tensor(calibrated_prediction_logits), torch.tensor(references)):.3f}")
        uncalibrated_predictions = torch.argmax(torch.tensor(uncalibrated_prediction_logits), dim=1).tolist()
        calibrated_predictions = torch.argmax(torch.tensor(calibrated_prediction_logits), dim=1).tolist()
        print(f"uncalibrated accuracy -> {sklearn.metrics.accuracy_score(y_true=references, y_pred=uncalibrated_predictions):.3f}")
        print(f"calibrated accuracy -> {sklearn.metrics.accuracy_score(y_true=references, y_pred=calibrated_predictions):.3f}")
        uncalibrated_acc_holder.append(sklearn.metrics.accuracy_score(y_true=references, y_pred=uncalibrated_predictions))
        calibrated_acc_holder.append(sklearn.metrics.accuracy_score(y_true=references, y_pred=calibrated_predictions))

    selected_uncalibrated_globalE_kl = np.argsort(uncalibrated_predicted_distribution_holder)[:4]
    selected_calibrated_globalE_kl = np.argsort(calibrated_predicted_distribution_holder)[:4]
    selected_uncalibrated_localE_kl = np.argsort(uniform_uncalibrated_kl_holder)[:4]
    selected_calibrated_localE_kl = np.argsort(uniform_calibrated_kl_holder)[:4]

    selected_uncalibrated_globalE_acc = [uncalibrated_acc_holder[i] for i in selected_uncalibrated_globalE_kl]
    selected_uncalibrated_globalE_ece = [uncalibrated_ece_holder[i] for i in selected_uncalibrated_globalE_kl]
    selected_calibrated_globalE_acc = [calibrated_acc_holder[i] for i in selected_calibrated_globalE_kl]
    selected_calibrated_globalE_ece = [calibrated_ece_holder[i] for i in selected_calibrated_globalE_kl]
    selected_uncalibrated_localE_acc = [uncalibrated_acc_holder[i] for i in selected_uncalibrated_localE_kl]
    selected_uncalibrated_localE_ece = [uncalibrated_ece_holder[i] for i in selected_uncalibrated_localE_kl]
    selected_calibrated_localE_acc = [calibrated_acc_holder[i] for i in selected_calibrated_localE_kl]
    selected_calibrated_localE_ece = [calibrated_ece_holder[i] for i in selected_calibrated_localE_kl]

    selected_oracle_uncalibrated_idx = np.argsort(uncalibrated_acc_holder)[-4:]
    selected_oracle_uncalibrated_acc = [uncalibrated_acc_holder[i] for i in selected_oracle_uncalibrated_idx]
    selected_oracle_uncalibrated_ece = [uncalibrated_ece_holder[i] for i in selected_oracle_uncalibrated_idx]
    selected_oracle_calibrated_idx = np.argsort(calibrated_acc_holder)[-4:]
    selected_oracle_calibrated_acc = [calibrated_acc_holder[i] for i in selected_oracle_calibrated_idx]
    selected_oracle_calibrated_ece = [calibrated_ece_holder[i] for i in selected_oracle_calibrated_idx]

    return {
        "uncalibrated_ece": uncalibrated_ece_holder,
        "uncalibrated_acc": uncalibrated_acc_holder,
        "calibrated_ece": calibrated_ece_holder,
        "calibrated_acc": calibrated_acc_holder,
        "oracle_uncalibrated_ece": selected_oracle_uncalibrated_ece,
        "oracle_uncalibrated_acc": selected_oracle_uncalibrated_acc,
        "oracle_calibrated_ece": selected_oracle_calibrated_ece,
        "oracle_calibrated_acc": selected_oracle_calibrated_acc,
        "selected_uncalibrated_globalE_acc": selected_uncalibrated_globalE_acc,
        "selected_uncalibrated_globalE_ece": selected_uncalibrated_globalE_ece,
        "selected_calibrated_globalE_acc": selected_calibrated_globalE_acc,
        "selected_calibrated_globalE_ece": selected_calibrated_globalE_ece,
        "selected_uncalibrated_localE_acc": selected_uncalibrated_localE_acc,
        "selected_uncalibrated_localE_ece": selected_uncalibrated_localE_ece,
        "selected_calibrated_localE_acc": selected_calibrated_localE_acc,
        "selected_calibrated_localE_ece": selected_calibrated_localE_ece,
    }


def prompting(tokenizer, model, trainset, testset, args, sample_ids=None, device="cpu", logger=None):
    model.eval()
    permutations = get_permutations(sample_ids, num_samples=24)
    id2label, labels, label_mapping = get_label_mapping(args.dataset)
    tokenized_labels = tokenizer(labels, return_tensors='pt', padding=True)
    if 'facebook/opt' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt").input_ids[:, 1].view(-1)
    elif 'llama' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt", padding=True).input_ids[:, 2].view(-1)
    elif 'EleutherAI' in model.name_or_path or 'gpt2' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt").input_ids.view(-1)
    else:
        print("using perplexity scoring!")

    if 'openllama' in model.name_or_path or 'llama' in model.name_or_path:
        perplexity_scoring=False
    else:
        perplexity_scoring=False
    
    softmax = torch.nn.Softmax(dim=0)
    label_distribution = get_label_distribution(testset, args)
    print(f"label distribution -> {label_distribution}")
    null_input = template_null(args.dataset)
    null_logits_holder = []
    for i, permutation in enumerate(permutations):
        prompt = [f"{template_demonstration(args.dataset, trainset[sample_id]['sentence'], trainset[sample_id]['label'])}" for sample_id in permutation]
        if perplexity_scoring:
            logits, _ = make_inference_perplexity(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
        else:
            logits, _ = make_inference(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, label_idx, device=device)
        logits = softmax(logits)
        null_logits_holder.append(logits.tolist())

    null_uncalibrated_acc_holder = []
    null_calibrated_acc_holder = []
    uniform_uncalibrated_acc_holder = []
    uniform_calibrated_acc_holder = []
    prior_uncalibrated_acc_holder = []
    prior_calibrated_acc_holder = []

    null_uncalibrated_ece_holder = []
    null_calibrated_ece_holder = []
    uniform_uncalibrated_ece_holder = []
    uniform_calibrated_ece_holder = []
    prior_uncalibrated_ece_holder = []
    prior_calibrated_ece_holder = []

    uncalibrated_acc_holder = []
    calibrated_acc_holder = []
    uncalibrated_ece_holder = []
    calibrated_ece_holder = []

    null_kl_holder = []
    uniform_uncalibrated_kl_holder = []
    uniform_calibrated_kl_holder = []
    prior_uncalibrated_kl_holder = []
    prior_calibrated_kl_holder = []

    for i, permutation in enumerate(permutations):
        prompt = [f"{template_demonstration(args.dataset, trainset[sample_id]['sentence'], trainset[sample_id]['label'])}" for sample_id in permutation]
        uncalibrated_prediction_logits, calibrated_prediction_logits = [], []
        references = []
        uncalibrated_predictions = []
        calibrated_predictions = []
        print(f"\ncurrent permutation id -> {i}")
        print(f"current permutation -> {permutation}")
        null_logits = null_logits_holder[i]

        # print(f"null logits -> {null_logits}")

        embeddings_holder = []
        for j, row in tqdm(enumerate(testset), desc='running prompting...'):
            test_instance = template_test(args.dataset, row['sentence'], label_mapping[row['label']])
            test_instance = f"".join(prompt)+f"{test_instance}"
            tokenized_input = tokenizer(test_instance, return_tensors='pt', padding=False, truncation=True, max_length=1024) 
            if perplexity_scoring:
                uncalibrated_logits, embeddings = make_inference_perplexity(model, tokenizer(test_instance, return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
            else:
                uncalibrated_logits, embeddings = make_inference(model, tokenizer(test_instance, return_tensors='pt').input_ids, label_idx, device=device)
            # print(f"\nlogits -> {logits}")
            uncalibrated_logits = softmax(uncalibrated_logits)
            embeddings_holder.append(embeddings.squeeze().tolist())

            if args.calibration_mode == 'pmi':
                calibrated_logits = uncalibrated_logits/torch.tensor(null_logits)
            elif args.calibration_mode == 'contextual':
                W = (torch.ones_like(uncalibrated_logits)/uncalibrated_logits.shape[0])/torch.tensor(null_logits)
                calibrated_logits = W * uncalibrated_logits
            calibrated_logits = softmax(calibrated_logits)
            
            uncalibrated_prediction_logits.append(uncalibrated_logits.tolist())
            calibrated_prediction_logits.append(calibrated_logits.tolist())
            references.append(labels.index(label_mapping[row['label']]))

        uniform = (np.ones(len(labels))/len(labels)).tolist()
        null_kl_holder.append(scipy.stats.entropy(null_logits, uniform))
        uniform_uncalibrated_kl_holder.append(scipy.stats.entropy(np.mean(uncalibrated_prediction_logits, axis=0), uniform))
        uniform_calibrated_kl_holder.append(scipy.stats.entropy(np.mean(calibrated_prediction_logits, axis=0), uniform))
        prior_uncalibrated_kl_holder.append(scipy.stats.entropy(np.mean(uncalibrated_prediction_logits, axis=0), label_distribution))
        prior_calibrated_kl_holder.append(scipy.stats.entropy(np.mean(calibrated_prediction_logits, axis=0), label_distribution))

        """ # to alter the kl divergence items
        null_kl_holder.append(scipy.stats.entropy(uniform, null_logits))
        uniform_uncalibrated_kl_holder.append(scipy.stats.entropy(uniform, np.mean(uncalibrated_prediction_logits, axis=0)))
        uniform_calibrated_kl_holder.append(scipy.stats.entropy(uniform, np.mean(calibrated_prediction_logits, axis=0)))
        prior_uncalibrated_kl_holder.append(scipy.stats.entropy(label_distribution, np.mean(uncalibrated_prediction_logits, axis=0)))
        prior_calibrated_kl_holder.append(scipy.stats.entropy(label_distribution, np.mean(calibrated_prediction_logits, axis=0)))
        """

        ece_metric = MulticlassCalibrationError(num_classes=len(uncalibrated_prediction_logits[0]), n_bins=100, norm="l1")
        uncalibrated_ece_holder.append(ece_metric(torch.tensor(uncalibrated_prediction_logits), torch.tensor(references)).item())
        calibrated_ece_holder.append(ece_metric(torch.tensor(calibrated_prediction_logits), torch.tensor(references)).item())
        print(f"uncalibrated ece -> {ece_metric(torch.tensor(uncalibrated_prediction_logits), torch.tensor(references)):.3f}")
        print(f"calibrated ece -> {ece_metric(torch.tensor(calibrated_prediction_logits), torch.tensor(references)):.3f}")

        uncalibrated_predictions = torch.argmax(torch.tensor(uncalibrated_prediction_logits), dim=1).tolist()
        calibrated_predictions = torch.argmax(torch.tensor(calibrated_prediction_logits), dim=1).tolist()

        print(f"uncalibrated accuracy -> {sklearn.metrics.accuracy_score(y_true=references, y_pred=uncalibrated_predictions):.3f}")
        print(f"calibrated accuracy -> {sklearn.metrics.accuracy_score(y_true=references, y_pred=calibrated_predictions):.3f}")
        uncalibrated_acc_holder.append(sklearn.metrics.accuracy_score(y_true=references, y_pred=uncalibrated_predictions))
        calibrated_acc_holder.append(sklearn.metrics.accuracy_score(y_true=references, y_pred=calibrated_predictions))

    # print(uncalibrated_acc_holder)
    # print(f"mean acc -> {np.mean(uncalibrated_acc_holder):.3f}")
    # print(calibrated_acc_holder)
    # print(f"mean acc -> {np.mean(calibrated_acc_holder):.3f}")
    # print(f"null kl holder -> {null_kl_holder}\n")
    # print(f"uniform uncalibrated kl holder -> {uniform_uncalibrated_kl_holder}\n")
    # print(f"uniform calibrated kl holder -> {uniform_calibrated_kl_holder}\n")
    # print(f"prior uncalibrated kl holder -> {prior_uncalibrated_kl_holder}\n")
    # print(f"prior calibrated kl holder -> {prior_calibrated_kl_holder}")

    selected_null_kl = np.argsort(null_kl_holder)[:4]
    selected_uniform_uncalibrated_kl = np.argsort(uniform_uncalibrated_kl_holder)[:4]
    selected_uniform_calibrated_kl = np.argsort(uniform_calibrated_kl_holder)[:4]
    selected_prior_uncalibrated_kl = np.argsort(prior_uncalibrated_kl_holder)[:4]
    selected_prior_calibrated_kl = np.argsort(prior_calibrated_kl_holder)[:4]

    selected_null_uncalibrated_acc = [uncalibrated_acc_holder[i] for i in selected_null_kl]
    selected_null_uncalibrated_ece = [uncalibrated_ece_holder[i] for i in selected_null_kl]
    selected_null_calibrated_acc = [calibrated_acc_holder[i] for i in selected_null_kl]
    selected_null_calibrated_ece = [calibrated_ece_holder[i] for i in selected_null_kl]
    selected_uniform_uncalibrated_acc = [uncalibrated_acc_holder[i] for i in selected_uniform_uncalibrated_kl]
    selected_uniform_uncalibrated_ece = [uncalibrated_ece_holder[i] for i in selected_uniform_uncalibrated_kl]
    selected_uniform_calibrated_acc = [calibrated_acc_holder[i] for i in selected_uniform_calibrated_kl]
    selected_uniform_calibrated_ece = [calibrated_ece_holder[i] for i in selected_uniform_calibrated_kl]
    selected_prior_uncalibrated_acc = [uncalibrated_acc_holder[i] for i in selected_prior_uncalibrated_kl]
    selected_prior_uncalibrated_ece = [uncalibrated_ece_holder[i] for i in selected_prior_uncalibrated_kl]
    selected_prior_calibrated_acc = [calibrated_acc_holder[i] for i in selected_prior_calibrated_kl]
    selected_prior_calibrated_ece = [calibrated_ece_holder[i] for i in selected_prior_calibrated_kl]

    selected_oracle_uncalibrated_idx = np.argsort(uncalibrated_acc_holder)[-4:]
    selected_oracle_uncalibrated_acc = [uncalibrated_acc_holder[i] for i in selected_oracle_uncalibrated_idx]
    selected_oracle_uncalibrated_ece = [uncalibrated_ece_holder[i] for i in selected_oracle_uncalibrated_idx]

    selected_oracle_calibrated_idx = np.argsort(calibrated_acc_holder)[-4:]
    selected_oracle_calibrated_acc = [calibrated_acc_holder[i] for i in selected_oracle_calibrated_idx]
    selected_oracle_calibrated_ece = [calibrated_ece_holder[i] for i in selected_oracle_calibrated_idx]

    return {
        "null_kl": null_kl_holder,
        "uniform_uncalibrated_kl": uniform_uncalibrated_kl_holder,
        "uniform_calibrated_kl": uniform_calibrated_kl_holder,
        "prior_uncalibrated_kl": prior_uncalibrated_kl_holder,
        "prior_calibrated_kl": prior_calibrated_kl_holder,
        "uncalibrated_ece": uncalibrated_ece_holder,
        "uncalibrated_acc": uncalibrated_acc_holder,
        "calibrated_ece": calibrated_ece_holder,
        "calibrated_acc": calibrated_acc_holder,
        "oracle_uncalibrated_ece": selected_oracle_uncalibrated_ece,
        "oracle_uncalibrated_acc": selected_oracle_uncalibrated_acc,
        "oracle_calibrated_ece": selected_oracle_calibrated_ece,
        "oracle_calibrated_acc": selected_oracle_calibrated_acc,
        "selected_null_uncalibrated_ece": selected_null_uncalibrated_ece,
        "selected_null_uncalibrated_acc": selected_null_uncalibrated_acc,
        "selected_null_calibrated_ece": selected_null_calibrated_ece,
        "selected_null_calibrated_acc": selected_null_calibrated_acc,
        "selected_uniform_uncalibrated_ece": selected_uniform_uncalibrated_ece,
        "selected_uniform_uncalibrated_acc": selected_uniform_uncalibrated_acc,
        "selected_uniform_calibrated_ece": selected_uniform_calibrated_ece,
        "selected_uniform_calibrated_acc": selected_uniform_calibrated_acc,
        "selected_prior_uncalibrated_ece": selected_prior_uncalibrated_ece,
        "selected_prior_uncalibrated_acc": selected_prior_uncalibrated_acc,
        "selected_prior_calibrated_ece": selected_prior_calibrated_ece,
        "selected_prior_calibrated_acc": selected_prior_calibrated_acc,
    }


    # selected_mi = np.argsort(kl_div_holder)[:4]
    # print(f"selected mutual information -> {selected_mi}")
    # print(f"mean kl divergence selected -> {np.mean([kl_div_holder[i] for i in selected_mi]):.3f}")
    # print(f"mean random kl divergence -> {np.mean(kl_div_holder):.3f}")
    # print(f"mean selected accuracy -> {np.mean([accuracy_holder[i] for i in selected_mi]):.3f}")
    # print(f"mean random accuracy -> {np.mean(accuracy_holder):.3f}")
    # print(f"mean selected macro f1 -> {np.mean([f1_holder[i] for i in selected_mi]):.3f}")
    # print(f"mean random macro f1 -> {np.mean(f1_holder):.3f}")
    # # print(f"entropy -> {scipy.stats.entropy(np.array(logits_holder), axis=1)}")
    # return {
    #     "selected_accuracy": [accuracy_holder[i] for i in selected_mi],
    #     "accuracy": accuracy_holder,
    #     "selected_f1": [f1_holder[i] for i in selected_mi],
    #     "f1": f1_holder,
    #     "selected_ece": [ece_holder[i] for i in selected_mi],
    #     "ece": ece_holder,
    #     "selected_kl_div": np.sort(kl_div_holder)[:4],
    #     "kl_div": kl_div_holder,
    #     "entropy": scipy.stats.entropy(np.array(logits_holder), [(np.ones(len(labels))/len(labels)).tolist()], axis=1)
    # }

    # return [accuracy_holder[i] for i in selected_mi], [f1_holder[i] for i in selected_mi], [ece_holder[i] for i in selected_mi], accuracy_holder, f1_holder, ece_holder

def mi_prompting_instance(tokenizer, model, trainset, testset, args, sample_ids=None, device="cpu", logger=None):
    model.eval()
    permutations = get_permutations(sample_ids, num_samples=24)
    id2label, labels, label_mapping = get_label_mapping(args.dataset)
    tokenized_labels = tokenizer(labels, return_tensors='pt')
    if 'facebook/opt' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt").input_ids[:, 1:].view(-1)
    elif 'EleutherAI' in model.name_or_path or 'gpt2' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt").input_ids.view(-1)
    else:
        print("using perplexity scoring!")

    if 'openllama' in model.name_or_path or 'llama' in model.name_or_path:
        perplexity_scoring=True
    else:
        perplexity_scoring=False
    
    softmax = torch.nn.Softmax(dim=0)
    label_distribution = get_label_distribution(testset, args)
    print(f"label distribution -> {label_distribution}")
    null_input = template_null(args.dataset)
    logits_holder = []
    for i, permutation in enumerate(permutations):
        prompt = [f"{template_demonstration(args.dataset, trainset[sample_id]['sentence'], trainset[sample_id]['label'])}" for sample_id in permutation]
        if perplexity_scoring:
            logits = make_inference_perplexity(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
        else:
            logits = make_inference(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, label_idx, device=device)
        logits = softmax(logits)
        logits_holder.append(logits.tolist())
    prediction_logits_holder = []
    for i, permutation in enumerate(permutations):
        prediction_logits = []
        targets = []
        predictions, references = [], []
        print(f"current permutation id -> {i}")
        print(f"current permutation -> {permutation}")
        prompt = [f"{template_demonstration(args.dataset, trainset[sample_id]['sentence'], trainset[sample_id]['label'])}" for sample_id in permutation]
        null_input = template_null(args.dataset)
        tokenized_null_input = tokenizer(f"".join(prompt)+null_input, return_tensors="pt")
        if perplexity_scoring:
            logits_ = make_inference_perplexity(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
        else:
            logits_ = make_inference(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, label_idx, device=device)
        logits_ = softmax(logits_)

        for j, row in tqdm(enumerate(testset), desc='running mutual information estimation prompting...'):
            test_instance = template_test(args.dataset, row['sentence'], label_mapping[row['label']])
            test_instance = f"".join(prompt)+f"{test_instance}"
            # print(f"test instance -> {[test_instance]}")
            tokenized_input = tokenizer(test_instance, return_tensors='pt', padding=False, truncation=True, max_length=1024) 
            if perplexity_scoring:
                logits = make_inference_perplexity(model, tokenizer(test_instance, return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
            else:
                logits = make_inference(model, tokenizer(test_instance, return_tensors='pt').input_ids, label_idx, device=device)
            logits = softmax(logits)
            if args.do_calibration:
                if args.calibration_mode == 'pmi':
                    logits = logits/logits_
                elif args.calibration_mode == 'contextual':
                    W = (torch.ones_like(logits) / logits.shape[0]) / logits_
                    logits = W * logits
                logits = softmax(logits)
            prediction_logits.append(logits.tolist())
            targets.append(labels.index(label_mapping[row['label']]))
        print(f"mean logits -> {np.mean(np.array(prediction_logits), axis=0)}")
        predictions = np.argmax(np.array(prediction_logits), axis=1)
        label_distribution_ = [0 for i in labels]
        for k in predictions:
            label_distribution_[k] += 1
        print(f"predicted label distribution -> {label_distribution_/np.sum(label_distribution_)}")
        prediction_logits_holder.append(prediction_logits)
    sys.exit()
    
    # print(np.array(prediction_logits_holder).shape)
    prediction_logits_holder = np.transpose(np.array(prediction_logits_holder), (1,0,2))
    prediction_logits = [ ]
    for i, row in enumerate(testset):
        permutation_logits = prediction_logits_holder[i,:,:] # [num_permutations, num_classes]
        kl_div = scipy.stats.entropy(permutation_logits, [label_distribution.tolist()], axis=1)
        prediction_logits.append([prediction_logits_holder[i,j,:] for j in np.argsort(kl_div)[:4]]) # minimum kl divergence between predictive logits and label distributions
    prediction_logits = np.array(prediction_logits)
    # print(prediction_logits)
    prediction_logits = prediction_logits.reshape(prediction_logits.shape[0]*prediction_logits.shape[1], prediction_logits.shape[2])
    # print(prediction_logits)
    # sys.exit()
    predictions = torch.argmax(torch.tensor(prediction_logits), dim=1).tolist()
    selected_accuracy = sklearn.metrics.accuracy_score(y_true=np.array(targets), y_pred=predictions[:256])
    print(f"selected accuracy -> {selected_accuracy}")
    sys.exit()
    predictions = np.argmax(prediction_logits_holder.reshape(prediction_logits_holder.shape[0]*prediction_logits_holder.shape[1], prediction_logits_holder.shape[2]), axis=1)
    mean_accuracy = sklearn.metrics.accuracy_score(y_true=np.array(targets*24), y_pred=predictions)
    print(f"mean accuracy -> {mean_accuracy}")
    sys.exit()

def calibration_prompting(tokenizer, model, trainset, testset, args, sample_ids=None, device="cpu", logger=None):
    model.eval()
    permutations = get_permutations(sample_ids, num_samples=24)
    id2label, labels, label_mapping = get_label_mapping(args.dataset)
    tokenized_labels = tokenizer(labels, return_tensors='pt')

    if 'facebook/opt' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt").input_ids[:, 1:].view(-1)
    elif 'EleutherAI' in model.name_or_path or 'gpt2' in model.name_or_path:
        label_idx = tokenizer(labels, return_tensors="pt").input_ids.view(-1)
    else:
        print("using perplexity scoring!")

    if 'openllama' in model.name_or_path or 'llama' in model.name_or_path:
        perplexity_scoring=True
    else:
        perplexity_scoring=False

    null_input = template_null(args.dataset)
    entropy, logits_holder = [], []
    softmax = torch.nn.Softmax(dim=0)
    for i, permutation in enumerate(permutations):
        prompt = [f"{template_demonstration(args.dataset, trainset[sample_id]['sentence'], trainset[sample_id]['label'])}" for sample_id in permutation]
        if perplexity_scoring:
            logits, _ = make_inference_perplexity(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
        else:
            logits, _ = make_inference(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, label_idx, device=device)
        logits = softmax(logits)
        entropy.append(scipy.stats.entropy(logits.tolist()))
        logits_holder.append(logits.tolist())

    topk_permutations = [permutations[i] for i in np.argsort(entropy)[-4:]]
    for i in np.argsort(entropy)[-4:]:
        print(f"selected permutation -> {permutations[i]}")
        print(f"selected logits -> {logits_holder[i]}")
        print(f"entropy -> {scipy.stats.entropy(logits_holder[i]):.3f}")

    accuracy_holder, ece_holder = [], []
    for selected_permutation in topk_permutations:
        prediction_logits = []
        targets = []
        predictions, references = [], []
        print(f'current permutation -> {selected_permutation}')
        prompt = [f"{template_demonstration(args.dataset, trainset[sample_id]['sentence'], trainset[sample_id]['label'])}" for sample_id in selected_permutation]
        null_input = template_null(args.dataset)
        tokenized_null_input = tokenizer(f"".join(prompt)+null_input, return_tensors="pt")
        if perplexity_scoring:
            logits_, _ = make_inference_perplexity(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
        else:
            logits_, _ = make_inference(model, tokenizer(f"".join(prompt)+f"{null_input}", return_tensors='pt').input_ids, label_idx, device=device)
        logits_ = softmax(logits_)

        for i, row in tqdm(enumerate(testset), desc='running calibration prompting...'):
            test_instance = template_test(args.dataset, row['sentence'], label_mapping[row['label']])
            test_instance = f"".join(prompt)+f"{test_instance}"
            # print(f"test instance -> {[test_instance]}")
            tokenized_input = tokenizer(test_instance, return_tensors='pt', padding=False, truncation=True, max_length=1024) 
            if perplexity_scoring:
                logits, _ = make_inference_perplexity(model, tokenizer(test_instance, return_tensors='pt').input_ids, tokenized_labels.input_ids, device=device)
            else:
                logits, _ = make_inference(model, tokenizer(test_instance, return_tensors='pt').input_ids, label_idx, device=device)
            logits = softmax(logits)
            
            if args.do_calibration:
                if args.calibration_mode == 'pmi':
                    logits = logits/logits_
                elif args.calibration_mode == 'contextual':
                    W = (torch.ones_like(logits) / logits.shape[0]) / logits_
                    logits = W * logits
                logits = softmax(logits)
            prediction_logits.append(logits.tolist())
            targets.append(labels.index(label_mapping[row['label']]))

        ece_metric = MulticlassCalibrationError(num_classes=len(prediction_logits[0]), n_bins=100, norm='l1')
        calibration_error = ece_metric(torch.tensor(prediction_logits), torch.tensor(targets))
        print(f"\ncalibration error -> {calibration_error:.3f}")

        if perplexity_scoring:
            predictions = torch.argmin(torch.tensor(prediction_logits), dim=1).tolist()
        else:
            predictions = torch.argmax(torch.tensor(prediction_logits), dim=1).tolist()
        print(f"length of predictions -> {len(predictions)}")
        print(f"accuracy -> {sklearn.metrics.accuracy_score(y_true=targets, y_pred=predictions):.3f}\n")
        accuracy_holder.append(sklearn.metrics.accuracy_score(y_true=targets, y_pred=predictions))
        ece_holder.append(calibration_error)

    return accuracy_holder, ece_holder
