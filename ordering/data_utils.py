import os
import sys
import argparse
import logging
import random
import json
import time
from tqdm import tqdm
import scipy

import numpy as np
import torch


def get_label_mapping(dataset_name):

    # sentiment analysis
    if dataset_name in ['sst2', 'mr', 'cr', 'imdb', 'rotten_tomatoes', 'yelp']:
        id2label = {0: " negative", 1: " positive"}
        labels = [" negative", " positive"]
        label_mapping = {"0": " negative", "1": " positive"}
    if 'mpqa' in dataset_name:
        id2label = {0: " negative", 1: " positive"}
        labels = [" negative", " positive"]
        label_mapping = {"0": " negative", "1": " positive"}
    if 'sst5' in dataset_name:
        id2label = {0: " terrible", 1: " bad", 2: " okay", 3: " good", 4: " great"}
        labels = [" terrible", " bad", " okay", " good", " great"]
        label_mapping = {"0": " terrible", "1": " bad", "2": " okay", "3": " good", "4": " great"}
    if 'financial_phrasebank' in dataset_name:
        id2label = {0: " negative", 1: " neutral", 2: " positive"}
        labels = [" negative", " neutral", " positive"]
        label_mapping = {0: " negative", 1: " neutral", 2: " positive"}

    # topic classification
    if 'dbpedia' in dataset_name:
        id2label = {
            0: " company", 1: " school", 2: " artist", 3: " athlete", 4: " politics", 5: " transportation", \
            6: " building", 7: " nature", 8: " village", 9: " animal", 10: " plant", 11: " album", 12: " film", \
            13: " book"
        }
        labels = [' company', ' school', ' artist', ' athlete', ' politics', ' transportation', \
                ' building', ' nature', ' village', ' animal', ' plant', ' album', ' film', ' book']
        label_mapping = {
            '1': ' company', '2': ' school', '3': ' artist', '4': ' athlete', '5': ' politics', '6': ' transportation', \
            '7': ' building', '8': ' nature', '9': ' village', '10': ' animal', '11': ' plant', '12': ' album', \
            '13': ' film', '14': ' book'
            }
    if 'trec' in dataset_name:
        id2label = {0: " description", 1: " entity", 2: " expression", 3: " human", 4: " location", 5: " number"}
        labels = [" description", " entity", " expression", " human", " location", " number"]
        label_mapping = {"0": " description", "1": " entity", "2": " expression", "3": " human", "4": " location", "5": " number"}
    if 'agnews' in dataset_name:
        id2label = {0: " World", 1: " Sports", 2: " Business", 3: " Technology"}
        labels = [" World", " Sports", " Business", " Technology"]
        label_mapping = {"1": " World", "2": " Sports", "3": " Business", "4": " Technology"}
    if 'yahoo_topic' in dataset_name:
        id2label = {0: " culture", 1: " science", 2: " health", 3: " education", 4: " electronics", 5: " sports", 6: " business", 7: " entertainment", 8: " relationship", 9: " politics"}
        labels = [" culture", " science", " health", " education", " electronics", " sports", " business", " entertainment", " relationship", " politics"]
        label_mapping = {0: " culture", 1: " science", 2: " health", 3: " education", 4: " electronics", 5: " sports", 6: " business", 7: " entertainment", 8: " relationship", 9: " politics"}

    if 'subj' in dataset_name:
        id2label = {0: " subjective", 1: " objective"}
        labels = [" subjective", " objective"]
        label_mapping = {"0": " subjective", "1": " objective"}

    # natural language inference
    if 'rte' in dataset_name:
        id2label = {0: " False", 1: " True"}
        labels = [" False", " True"]
        label_mapping = {'not_entailment': ' False', 'entailment': ' True'}
    if 'cb' in dataset_name:
        id2label = {0: " false", 1: " true", 2: " neither"}
        labels = [" false", " true", " neither"]
        label_mapping = {"contradiction": " false", "entailment": " true", "neutral": " neither"}

    if 'boolq' in dataset_name:
        id2label = {0: "no", 1: "yes"}
        labels = ["no", "yes"]
        label_mapping = {"no": "no", "yes": "yes"}

    if 'tweet_hate' in dataset_name:
        id2label = {0: " neutral", 1: " hate"}
        labels = [" neutral", " hate"]
        label_mapping = {0: " neutral", 1: " hate"}
    if 'tweet_irony' in dataset_name:
        id2label = {0: " neutral", 1: " ironic"}
        labels = [" neutral", " ironic"]
        label_mapping = {0: " neutral", 1: " ironic"}
    if 'tweet_offensive' in dataset_name:
        id2label = {0: " neutral", 1: " offensive"}
        labels = [" neutral", " offensive"]
        label_mapping = {0: " neutral", 1: " offensive"}
    if 'ethos_binary' in dataset_name:
        id2label = {0: " neutral", 1: " hate"}
        labels = [" neutral", " hate"]
        label_mapping = {0: " neutral", 1: " hate"}

    return id2label, labels, label_mapping


def template_demonstration(dataset_name, input, target):
    id2label, labels, label_mapping = get_label_mapping(dataset_name)

    # sentiment
    if 'sst2' in dataset_name:
        # return f"{input}, Sentiment: {label_mapping[target]}. "
        template = f"Review: {input} \nSentiment:{label_mapping[target]}\n\n"
        return template
    if 'sst5' in dataset_name:
        template = f"Review: {input} \nSentiment:{label_mapping[target]}\n\n"
        return template
    if 'mr' in dataset_name:
        return f"Review: {input} \nSentiment:{label_mapping[target]}\n\n"
    if 'cr' in dataset_name:
        return f"Review: {input} \nSentiment:{label_mapping[target]}\n\n"
    if 'mpqa' in dataset_name:
        return f"Review: {input} \nSentiment:{label_mapping[target]}\n\n"
    if 'financial_phrasebank' in dataset_name:
        return f"News: {input} \nSentiment:{label_mapping[target]}\n\n"

    if 'imdb' in dataset_name:
        return f"Review: {input} \nSentiment:{label_mapping[target]}\n\n"
    if 'yelp' in dataset_name:
        return f"Review: {input} \nSentiment:{label_mapping[target]}\n\n"

    # topic classification
    if 'agnews' in dataset_name:
        return f"input: {input} \ntype:{label_mapping[target]}\n\n"
    if 'trec' in dataset_name:
        return f"question: {input} \ntype:{label_mapping[target]}\n\n"
    if 'subj' in dataset_name:
        return f"input: {input} \ntype:{label_mapping[target]}\n\n"
    if 'dbpedia' in dataset_name:
        return f"input: {input} \ntype:{label_mapping[target]}\n\n"
    if 'yahoo_topic' in dataset_name:
        return f"question: {input} \ntopic:{label_mapping[target]}\n\n"

    # natural language inference
    if 'rte' in dataset_name:
        sentence_1, sentence_2 = input[0], input[1]
        return f"premise: {sentence_1}\nhypothesis: {sentence_2}\nprediction:{label_mapping[target]}\n\n"
    if 'cb' in dataset_name:
        sentence_1, sentence_2 = input[0], input[1]
        return f"premise: {sentence_1}\nhypothesis: {sentence_2}\nprediction:{label_mapping[target]}\n\n"


    if 'boolq' in dataset_name:
        question, context = input.split("[SEP]")
        return f"Input: {context}, answering the following question according to the above context, {question}? yes or no? Output: {label_mapping[target]}"

    if 'tweet_hate' in dataset_name:
        return f"Tweet: {input} \nLabel:{label_mapping[target]}"
    if 'tweet_offensive' in dataset_name:
        return f"Tweet: {input} \nLabel:{label_mapping[target]}"
    if 'tweet_irony' in dataset_name:
        return f"Tweet: {input} \nLabel:{label_mapping[target]}"
    if 'ethos_binary' in dataset_name:
        return f"Text: {input} \nLabel:{label_mapping[target]}"


def template_test(dataset_name, input, target=None):

    # sentiment classification
    if 'sst2' in dataset_name or 'sst5' in dataset_name:
        # return f"{input}, Sentiment: "
        template = f"Review: {input} \nSentiment:"
        return template
    if 'mr' in dataset_name:
        return f"Review: {input} \nSentiment:"
    if 'cr' in dataset_name:
        return f"Review: {input} \nSentiment:"
    if 'mpqa' in dataset_name:
        return f"Review: {input} \nSentiment:"
    if 'financial_phrasebank' in dataset_name:
        return f"News: {input} \nSentiment:"

    if 'rotten_tomatoes' in dataset_name:
        return f"review: {input} \nsentiment: "
    if 'imdb' in dataset_name:
        return f"review: {input} \nsentiment:"
    if 'yelp' in dataset_name:
        return f"review: {input} \nsentiment:"

    # topic classification
    if 'agnews' in dataset_name:
        return f"input: {input} \ntype:"
    if 'trec' in dataset_name:
        return f"question: {input} \ntype:"
    if 'subj' in dataset_name:
        return f"input: {input} \ntype:"
    if 'dbpedia' in dataset_name:
        return f"input: {input} \ntype:"
    if 'yahoo_topic' in dataset_name:
        return f"question: {input} \ntopic:"

    # natural language inference
    if 'rte' in dataset_name:
        sentence_1, sentence_2 = input[0], input[1]
        return f"premise: {sentence_1}\nhypothesis: {sentence_2}\nprediction:"
    if 'cb' in dataset_name:
        sentence_1, sentence_2 = input[0], input[1]
        return f"premise: {sentence_1}\nhypothesis: {sentence_2}\nprediction:"

    if 'boolq' in dataset_name:
        question, context = input.split("[SEP]")
        return f"Input: {context}, answering the following question according to the above context, {question}? yes or no? Output: "
    if 'tweet_hate' in dataset_name:
        return f"Tweet: {input} \nLabel:"
    if 'tweet_irony' in dataset_name:
        return f"Tweet: {input} \nLabel:"
    if 'tweet_offensive' in dataset_name:
        return f"Tweet: {input} \nLabel:"
    if 'ethos_binary' in dataset_name:
        return f"Text: {input} \nLabel:"

def template_null(dataset_name, input=None, target=None):

    # sentiment
    if 'sst2' in dataset_name or 'sst5' in dataset_name:
        return f"Review:  \nSentiment:"
    if 'mr' in dataset_name:
        return f"Review:  \nSentiment:"
    if 'cr' in dataset_name:
        return f"Review:  \nSentiment:"
    if 'mpqa' in dataset_name:
        return f"Review:  \nSentiment:"
    if 'financial_phrasebank' in dataset_name:
        return f"News:  \nSentiment:"

    if 'rotten_tomatoes' in dataset_name:
        return f"review:  \nsentiment:"
    if 'imdb' in dataset_name:
        return f"review:  \nsentiment:"
    if 'yelp' in dataset_name:
        return f"review:  \nsentiment:"


    # topic classification
    if 'agnews' in dataset_name:
        return f"input:   \ntype:"
    if 'trec' in dataset_name:
        return f"input:   \ntype:"
    if 'subj' in dataset_name:
        return f"input:   \ntype:"
    if 'dbpedia' in dataset_name:
        return f"input:   \ntype:"
    if 'yahoo_topic' in dataset_name:
        return f"question:   \ntopic:"

    # natural language inference
    if 'rte' in dataset_name:
        return f"premise:  \nhypothesis:  \nprediction:"
    if 'cb' in dataset_name:
        return f"premise:  \nhypothesis:  \nprediction:"

    # toxicity
    if 'tweet_hate' in dataset_name:
        return f"Tweet:  \nLabel:"
    if 'tweet_irony' in dataset_name:
        return f"Tweet:  \nLabel:"
    if 'tweet_offensive' in dataset_name:
        return f"Tweet:  \nLabel:"
    if 'ethos_binary' in dataset_name:
        return f"Text:  \nLabel:"


def load_metaicl_dataset(dataset_name, seed=13):
    dataset_path = f"/raid/brutusxu/fewshot/datasets/{dataset_name}/"
    train_path = dataset_path+f"{dataset_name}_16_{seed}_train.jsonl"
    dev_path = dataset_path+f"{dataset_name}_16_{seed}_dev.jsonl"
    test_path = dataset_path+f"{dataset_name}_16_{seed}_test.jsonl"
    with open(train_path, 'r') as fin:
        trainset = []
        json_list = list(fin)
        for i, json_str in enumerate(json_list):
            if i > 160:
                break
            trainset.append(json.loads(json_str))
    with open(dev_path, 'r') as fin:
        validset = []
        json_list = list(fin)
        for json_str in json_list:
            validset.append(json.loads(json_str))
    with open(test_path, 'r') as fin:
        testset = []
        json_list = list(fin)
        for i, json_str in enumerate(json_list):
            if i > 2000:
                break
            testset.append(json.loads(json_str))
    return trainset, validset, testset

def load_knnprompting_dataset(dataset_name):
    dataset_path = f"/home/sci/zhichao.xu/fewshot/data/{dataset_name}/"
    train_path = os.path.join(dataset_path, 'train.jsonl')
    dev_path = os.path.join(dataset_path, 'dev_subsample.jsonl')
    test_path = os.path.join(dataset_path, 'test.jsonl')

    if dataset_name in ['sst2', 'sst5', 'mr', 'cr', 'mpqa', 'agnews', \
    'subj', 'trec', 'dbpedia', 'financial_phrasebank', 'yahoo_topic', \
    'tweet_offensive', 'tweet_irony', 'tweet_hate', 'ethos_binary',]:
        with open(train_path, 'r') as fin:
            trainset = []
            json_list = list(fin)
            for i, json_str in tqdm(enumerate(json_list), desc='loading train dataset...'):
                trainset.append(json.loads(json_str))
        with open(dev_path, 'r') as fin:
            validset = []
            json_list = list(fin)
            for json_str in json_list:
                validset.append(json.loads(json_str))
        with open(test_path, 'r') as fin:
            testset = []
            json_list = list(fin)
            for i, json_str in tqdm(enumerate(json_list), desc='loading test dataset...'):
                if i > 2000:
                    break
                testset.append(json.loads(json_str))

    if dataset_name in ['rte']:
        with open(train_path, 'r') as fin:
            trainset = []
            json_list = list(fin)
            for i, json_str in tqdm(enumerate(json_list), desc='loading train dataset...'):
                json_dict = json.loads(json_str)
                trainset.append({'sentence': [json_dict['sentence_1'], json_dict['sentence_2']], 'label': json_dict['label']})
        with open(dev_path, 'r') as fin:
            validset = []
            json_list = list(fin)
            for json_str in json_list:
                json_dict = json.loads(json_str)
                validset.append({'sentence': [json_dict['sentence_1'], json_dict['sentence_2']], 'label': json_dict['label']})
        with open(test_path, 'r') as fin:
            testset = []
            json_list = list(fin)
            for i, json_str in tqdm(enumerate(json_list), desc='loading test dataset...'):
                if i > 2000:
                    break
                json_dict = json.loads(json_str)
                testset.append({'sentence': [json_dict['sentence_1'], json_dict['sentence_2']], 'label': 'na'})

    if dataset_name in ['cb']:
        with open(train_path, 'r') as fin:
            trainset = []
            json_list = list(fin)
            for i, json_str in tqdm(enumerate(json_list), desc='loading train dataset...'):
                json_dict = json.loads(json_str)
                trainset.append({'sentence': [json_dict['premise'], json_dict['hypothesis']], 'label': json_dict['label']})
        with open(dev_path, 'r') as fin:
            validset = []
            json_list = list(fin)
            for json_str in json_list:
                json_dict = json.loads(json_str)
                validset.append({'sentence': [json_dict['premise'], json_dict['hypothesis']], 'label': json_dict['label']})
        with open(test_path, 'r') as fin:
            testset = []
            json_list = list(fin)
            for i, json_str in tqdm(enumerate(json_list), desc='loading test dataset...'):
                if i > 2000:
                    break
                json_dict = json.loads(json_str)
                testset.append({'sentence': [json_dict['premise'], json_dict['hypothesis']], 'label': 'na'})

    return trainset, validset, testset


def order_samples(sampled, sampled_pvis):
    hard2easy = np.argsort(sampled_pvis)
    return [sampled[i] for i in hard2easy]


def order_samples_inside_out(sample_ids, sample_pvis):
    # input = [very hard, hard, easy, very easy]
    # return [easy, very hard, hard, very easy]

    num_samples = len(sample_ids)
    hard2easy = np.argsort(sample_pvis)
    stack = hard2easy[:num_samples//2]
    for idx, i in enumerate(hard2easy[num_samples//2:]):
        if idx % 2 == 0:
            stack.insert(0, i)
        else:
            stack.append(i)
    return stack

def order_samples_outside_in(sample_ids, sample_pvis):
    # input = [very hard, hard, easy, very easy]
    # return [very hard, very easy, easy, hard]
    num_samples = len(sample_ids)
    hard2easy = np.argsort(sample_ids).tolist()
    stack = hard2easy[num_samples//2:]
    for idx, i in enumerate(hard2easy[:num_samples//2]):
        if idx % 2 == 0:
            stack.insert(0, sample_ids[i])
        else:
            stack.append(sample_ids[i])
    return stack

def sampling_w_pvi(dataset, pvis, num_shots=4, seed=1, strategy='hard'):
    type_dict = {}
    for i, row in enumerate(dataset):
        if row['output'] not in type_dict:
            type_dict[row['output']] = []
        type_dict[row['output']].append(i)
    sorted_pvis = np.sort(pvis)
    sorted_samples = np.argsort(pvis)

    assert strategy in ['hard', 'easy', 'balanced'], 'Unrecognized strategy, force exit'
    sampled = []
    for k, v in type_dict.items():
        pvi_ = [pvis[i] for i in v]
        if strategy == 'hard':
            weight = scipy.special.softmax([-i for i in pvi_])
        elif strategy == 'easy':
            weight = scipy.special.softmax(pvi_)
        elif strategy == 'balanced':
            weight = [1/len(v) for i in v]
        sampled.extend(np.random.choice(v, num_shots//len(type_dict), p=weight).tolist())
    return sampled


def get_permutations(sampled, seed=None, num_samples=24):
    if len(sampled) > 6:
        return get_permutations_from_numpy(sampled, num_samples=num_samples)
    else:
        import itertools
        import random
        if seed:
            random.seed(seed)
        permutations = list(itertools.permutations(sampled))
        random.shuffle(permutations)
        return permutations[:num_samples]

def get_permutations_from_numpy(sampled, num_samples=24):
    permutations = []
    for i in range(num_samples):
        np.random.seed(i)
        permutations.append(np.random.permutation(sampled).tolist())
    
    return permutations

def balanced_sampling(dataset, seed=1, num_shots=4):
    random.seed(seed)
    pool = {}
    for i, row in enumerate(dataset):
        if row['label'] not in pool:
            pool[row['label']] = []
        pool[row['label']].append(i)
    
    assert num_shots % len(pool) == 0, "# icl examples must can be divided by pool size"
    sampled = []
    for k, v in pool.items():
        sampled.extend(list(random.sample(v, num_shots//len(pool))))
    return sampled

def uniform_sampling(dataset, seed=1, num_shots=4):
    random.seed(seed)
    idx = [i for i in range(len(dataset))]
    sampled = random.sample(idx, num_shots)
    return list(sampled)