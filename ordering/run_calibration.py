import os
import sys
import argparse
import logging
import random
import json
import time
from tqdm import tqdm

import numpy as np

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM

from transformers import BitsAndBytesConfig

import data_utils
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="agnews")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--loading_mode", type=str, choices=["int4", "int8", "fp16"])

    parser.add_argument("--sample_ids", type=str, default="")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_shots", type=int, default=4)
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--sampling_strategy", type=str, choices=["balanced", "uniform"], default="balanced")

    parser.add_argument("--do_calibration", action="store_true")
    parser.add_argument("--calibration_mode", type=str, choices=["pmi", "contextual"], default="pmi")
    parser.add_argument("--do_print", action="store_true")
    parser.add_argument("--logging", type=str, default="./loggings/default_logging.log")

    parser.add_argument("--running_mode", type=str, choices=["kl_prompting", "baseline"])
    parser.add_argument("--prior_mode", type=str, choices=["oracle", "uniform"], default="oracle")
    parser.add_argument("--add_context", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=5)

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        filename=args.logging, 
        filemode='a',
        )
    logger = logging.getLogger(__name__)

    if 'llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if args.loading_mode == 'int4':
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
            )
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=nf4_config, device_map="balanced")
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.loading_mode == 'int8':
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="balanced", torch_dtype=torch.float16)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.loading_mode == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="balanced", torch_dtype=torch.float16, trust_remote_code=True)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(DEVICE)
    
    trainset, validset, testset = data_utils.load_knnprompting_dataset(args.dataset)

    for k, v in vars(args).items():
        logger.info(f"{k} -> {v}")
    
    if args.running_mode == "baseline":
        args.calibration_mode = "pmi"
        uncalibrated_ece = []
        uncalibrated_acc = []
        calibrated_ece = []
        calibrated_acc = []
        oracle_uncalibrated_ece = []
        oracle_uncalibrated_acc = []
        oracle_calibrated_ece = []
        oracle_calibrated_acc = []

        selected_globalE_uncalibrated_acc = []
        selected_globalE_uncalibrated_ece = []
        selected_globalE_calibrated_acc = []
        selected_globalE_calibrated_ece = []
        selected_localE_uncalibrated_acc = []
        selected_localE_uncalibrated_ece = []
        selected_localE_calibrated_acc = []
        selected_localE_calibrated_ece = []

        oracle_acc_holder = []
        oracle_ece_holder = []
        oracle_var_holder = []

        if args.sample_ids:
            sample_ids = [int(i) for i in args.sample_ids.split(',')]
            print(f"manually designated samples -> {sample_ids}")
            pass

        meta_dict = {}
        meta_dict["args"] = vars(args)
        for seed in range(args.num_seeds):
            if args.sampling_strategy == "balanced":
                sample_ids = data_utils.balanced_sampling(dataset=trainset, seed=seed, num_shots=args.num_shots)
            elif args.sampling_strategy == "uniform":
                sample_ids = data_utils.uniform_sampling(dataset=trainset, seed=seed, num_shots=args.num_shots)

            logger.info(f'sample ids -> {sample_ids}')
            result_dict = utils.globelE_prompting(
                tokenizer=tokenizer,
                model=model,
                trainset=trainset,
                testset=validset,
                args=args,
                sample_ids=sample_ids,
                device=DEVICE,
                logger=logger,
            )
            meta_dict[f"seed_{seed}"] = result_dict
            uncalibrated_ece.extend(result_dict["uncalibrated_ece"])
            uncalibrated_acc.extend(result_dict["uncalibrated_acc"])
            calibrated_ece.extend(result_dict["calibrated_ece"])
            calibrated_acc.extend(result_dict["calibrated_acc"])
            oracle_uncalibrated_ece.extend(result_dict["oracle_uncalibrated_ece"])
            oracle_uncalibrated_acc.extend(result_dict["oracle_uncalibrated_acc"])
            oracle_calibrated_ece.extend(result_dict["oracle_calibrated_ece"])
            oracle_calibrated_acc.extend(result_dict["oracle_calibrated_acc"])
            selected_globalE_uncalibrated_acc.extend(result_dict["selected_uncalibrated_globalE_acc"])
            selected_globalE_uncalibrated_ece.extend(result_dict["selected_uncalibrated_globalE_ece"])
            selected_globalE_calibrated_acc.extend(result_dict["selected_calibrated_globalE_acc"])
            selected_globalE_calibrated_ece.extend(result_dict["selected_calibrated_globalE_ece"])
            selected_localE_uncalibrated_acc.extend(result_dict["selected_uncalibrated_localE_acc"])
            selected_localE_uncalibrated_ece.extend(result_dict["selected_uncalibrated_localE_ece"])
            selected_localE_calibrated_acc.extend(result_dict["selected_calibrated_localE_acc"])
            selected_localE_calibrated_ece.extend(result_dict["selected_calibrated_localE_ece"])

        print(f"uncalibrated_acc -> {np.mean(uncalibrated_acc):.3f}\n")
        print(f"uncalibrated_ece -> {np.mean(uncalibrated_ece):.3f}\n")
        print(f"uncalibrated acc var -> {np.std(uncalibrated_acc):.3f}\n")
        logger.info(f"")
        logger.info(f"uncalibrated_acc -> {np.mean(uncalibrated_acc):.3f}")
        logger.info(f"uncalibrated_ece -> {np.mean(uncalibrated_ece):.3f}")
        logger.info(f"uncalibrated acc var -> {np.std(uncalibrated_acc):.3f}\n")

        print(f"calibrated_acc -> {np.mean(calibrated_acc):.3f}\n")
        print(f"calibrated_ece -> {np.mean(calibrated_ece):.3f}\n")
        print(f"calibrated acc var -> {np.std(calibrated_acc):.3f}\n")
        logger.info(f"calibrated_acc -> {np.mean(calibrated_acc):.3f}")
        logger.info(f"calibrated_ece -> {np.mean(calibrated_ece):.3f}")
        logger.info(f"calibrated acc var -> {np.std(calibrated_acc):.3f}\n")

        logger.info(f"oracle uncalibrated acc -> {np.mean(oracle_uncalibrated_acc):.3f}")
        logger.info(f"oracle uncalibrated ece -> {np.mean(oracle_uncalibrated_ece):.3f}")
        logger.info(f"oracle uncalibrated var -> {np.std(oracle_uncalibrated_acc):.3f}\n")

        logger.info(f"oracle calibrated acc -> {np.mean(oracle_calibrated_acc):.3f}")
        logger.info(f"oracle calibrated ece -> {np.mean(oracle_calibrated_ece):.3f}")
        logger.info(f"oracle calibrated var -> {np.std(oracle_calibrated_acc):.3f}\n")

        print(f"selected_globalE_uncalibrated_acc -> {np.mean(selected_globalE_uncalibrated_acc):.3f}")
        print(f"selected_globalE_uncalibrated_ece -> {np.mean(selected_globalE_uncalibrated_ece):.3f}")
        print(f"selected_globalE_uncalibrated_var -> {np.std(selected_globalE_uncalibrated_acc)}\n")
        logger.info(f"selected_globalE_uncalibrated_acc -> {np.mean(selected_globalE_uncalibrated_acc):.3f}")
        logger.info(f"selected_globalE_uncalibrated_ece -> {np.mean(selected_globalE_uncalibrated_ece):.3f}")
        logger.info(f"selected_globalE_uncalibrated_var -> {np.std(selected_globalE_uncalibrated_acc):.3f}\n")

        print(f"selected_globalE_calibrated_acc -> {np.mean(selected_globalE_calibrated_acc):.3f}")
        print(f"selected_globalE_calibrated_ece -> {np.mean(selected_globalE_calibrated_ece):.3f}")
        print(f"selected_globalE_calibrated_var -> {np.std(selected_globalE_calibrated_acc):.3f}\n")
        logger.info(f"selected_globalE_calibrated_acc -> {np.mean(selected_globalE_calibrated_acc):.3f}")
        logger.info(f"selected_globalE_calibrated_ece -> {np.mean(selected_globalE_calibrated_ece):.3f}")
        logger.info(f"selected_globalE_calibrated_var -> {np.std(selected_globalE_calibrated_acc):.3f}\n")

        print(f"selected_localE_uncalibrated_acc -> {np.mean(selected_localE_uncalibrated_acc):.3f}")
        print(f"selected_localE_uncalibrated_ece -> {np.mean(selected_localE_uncalibrated_ece):.3f}")
        print(f"selected_localE_uncalibrated_var -> {np.std(selected_localE_uncalibrated_acc)}\n")
        logger.info(f"selected_localE_uncalibrated_acc -> {np.mean(selected_localE_uncalibrated_acc):.3f}")
        logger.info(f"selected_localE_uncalibrated_ece -> {np.mean(selected_localE_uncalibrated_ece):.3f}")
        logger.info(f"selected_localE_uncalibrated_var -> {np.std(selected_localE_uncalibrated_acc):.3f}\n")

        print(f"selected_localE_calibrated_acc -> {np.mean(selected_localE_calibrated_acc):.3f}")
        print(f"selected_localE_calibrated_ece -> {np.mean(selected_localE_calibrated_ece):.3f}")
        print(f"selected_localE_calibrated_var -> {np.std(selected_localE_calibrated_acc)}\n")
        logger.info(f"selected_localE_calibrated_acc -> {np.mean(selected_localE_calibrated_acc):.3f}")
        logger.info(f"selected_localE_calibrated_ece -> {np.mean(selected_localE_calibrated_ece):.3f}")
        logger.info(f"selected_localE_calibrated_var -> {np.std(selected_localE_calibrated_acc):.3f}\n")


    if args.running_mode == "kl_prompting":
        args.calibration_mode = "pmi"

        null_kl = []
        uniform_uncalibrated_kl = []
        uniform_calibrated_kl = []
        prior_uncalibrated_kl = []
        prior_calibrated_kl = []

        uncalibrated_ece = []
        uncalibrated_acc = []
        calibrated_ece = []
        calibrated_acc = []
        oracle_uncalibrated_ece = []
        oracle_uncalibrated_acc = []
        oracle_calibrated_ece = []
        oracle_calibrated_acc = []
        selected_null_uncalibrated_acc = []
        selected_null_uncalibrated_ece = []
        selected_null_calibrated_acc = []
        selected_null_calibrated_ece = []
        selected_uniform_uncalibrated_acc = []
        selected_uniform_uncalibrated_ece = []
        selected_uniform_calibrated_acc = []
        selected_uniform_calibrated_ece = []
        selected_prior_uncalibrated_acc = []
        selected_prior_uncalibrated_ece = []
        selected_prior_calibrated_acc = []
        selected_prior_calibrated_ece = []

        oracle_acc_holder = []
        oracle_ece_holder = []
        oracle_var_holder = []

        if args.sample_ids:
            sample_ids = [int(i) for i in args.sample_ids.split(',')]
            print(f"manually designated samples -> {sample_ids}")
            pass

        meta_dict = {}
        meta_dict["args"] = vars(args)
        for seed in range(args.num_seeds):
            if args.sampling_strategy == "balanced":
                sample_ids = data_utils.balanced_sampling(dataset=trainset, seed=seed, num_shots=args.num_shots)
            elif args.sampling_strategy == "uniform":
                sample_ids = data_utils.uniform_sampling(dataset=trainset, seed=seed, num_shots=args.num_shots)

            logger.info(f'sample ids -> {sample_ids}')
            result_dict = utils.prompting(
                tokenizer=tokenizer,
                model=model,
                trainset=trainset,
                testset=validset,
                args=args,
                sample_ids=sample_ids,
                device=DEVICE,
                logger=logger,
            )
            meta_dict[f"seed_{seed}"] = result_dict

            null_kl.extend(result_dict["null_kl"])
            uniform_uncalibrated_kl.extend(result_dict["uniform_uncalibrated_kl"])
            uniform_calibrated_kl.extend(result_dict["uniform_calibrated_kl"])
            prior_uncalibrated_kl.extend(result_dict["prior_uncalibrated_kl"])
            prior_calibrated_kl.extend(result_dict["prior_calibrated_kl"])

            uncalibrated_ece.extend(result_dict["uncalibrated_ece"])
            uncalibrated_acc.extend(result_dict["uncalibrated_acc"])
            calibrated_ece.extend(result_dict["calibrated_ece"])
            calibrated_acc.extend(result_dict["calibrated_acc"])
            oracle_uncalibrated_ece.extend(result_dict["oracle_uncalibrated_ece"])
            oracle_uncalibrated_acc.extend(result_dict["oracle_uncalibrated_acc"])
            oracle_calibrated_ece.extend(result_dict["oracle_calibrated_ece"])
            oracle_calibrated_acc.extend(result_dict["oracle_calibrated_acc"])
            selected_null_uncalibrated_ece.extend(result_dict["selected_null_uncalibrated_ece"])
            selected_null_uncalibrated_acc.extend(result_dict["selected_null_uncalibrated_acc"])
            selected_null_calibrated_acc.extend(result_dict["selected_null_calibrated_acc"])
            selected_null_calibrated_ece.extend(result_dict["selected_null_calibrated_ece"])
            selected_uniform_uncalibrated_acc.extend(result_dict["selected_uniform_uncalibrated_acc"])
            selected_uniform_uncalibrated_ece.extend(result_dict["selected_uniform_uncalibrated_ece"])
            selected_uniform_calibrated_acc.extend(result_dict["selected_uniform_calibrated_acc"])
            selected_uniform_calibrated_ece.extend(result_dict["selected_uniform_calibrated_ece"])
            selected_prior_uncalibrated_acc.extend(result_dict["selected_prior_uncalibrated_acc"])
            selected_prior_uncalibrated_ece.extend(result_dict["selected_prior_uncalibrated_ece"])
            selected_prior_calibrated_acc.extend(result_dict["selected_prior_calibrated_acc"])
            selected_prior_calibrated_ece.extend(result_dict["selected_prior_calibrated_ece"])

        logger.info(f"")
        logger.info(f"null kl -> {null_kl}")
        logger.info(f"uniform_uncalibrated_kl -> {uniform_uncalibrated_kl}")
        logger.info(f"uniform_calibrated_kl -> {uniform_calibrated_kl}")
        logger.info(f"prior_uncalibrated_kl -> {prior_uncalibrated_kl}")
        logger.info(f"prior_calibrated_kl -> {prior_calibrated_kl}")

        print(f"uncalibrated_acc -> {np.mean(uncalibrated_acc):.3f}\n")
        print(f"uncalibrated_ece -> {np.mean(uncalibrated_ece):.3f}\n")
        print(f"uncalibrated acc var -> {np.std(uncalibrated_acc):.3f}\n")
        logger.info(f"")
        logger.info(f"uncalibrated_acc -> {np.mean(uncalibrated_acc):.3f}")
        logger.info(f"uncalibrated_ece -> {np.mean(uncalibrated_ece):.3f}")
        logger.info(f"uncalibrated acc var -> {np.std(uncalibrated_acc):.3f}\n")

        print(f"calibrated_acc -> {np.mean(calibrated_acc):.3f}\n")
        print(f"calibrated_ece -> {np.mean(calibrated_ece):.3f}\n")
        print(f"calibrated acc var -> {np.std(calibrated_acc):.3f}\n")
        logger.info(f"calibrated_acc -> {np.mean(calibrated_acc):.3f}")
        logger.info(f"calibrated_ece -> {np.mean(calibrated_ece):.3f}")
        logger.info(f"calibrated acc var -> {np.std(calibrated_acc):.3f}\n")

        logger.info(f"oracle uncalibrated acc -> {np.mean(oracle_uncalibrated_acc):.3f}")
        logger.info(f"oracle uncalibrated ece -> {np.mean(oracle_uncalibrated_ece):.3f}")
        logger.info(f"oracle uncalibrated var -> {np.std(oracle_uncalibrated_acc):.3f}\n")

        logger.info(f"oracle calibrated acc -> {np.mean(oracle_calibrated_acc):.3f}")
        logger.info(f"oracle calibrated ece -> {np.mean(oracle_calibrated_ece):.3f}")
        logger.info(f"oracle calibrated var -> {np.std(oracle_calibrated_acc):.3f}\n")

        print(f"selected_null_uncalibrated_acc -> {np.mean(selected_null_uncalibrated_acc):.3f}\n")
        print(f"selected_null_uncalibrated_ece -> {np.mean(selected_null_uncalibrated_ece):.3f}\n")
        print(f"selected_null_uncalibrated_acc var -> {np.std(selected_null_uncalibrated_acc):.3f}\n")
        logger.info(f"selected_null_uncalibrated_acc -> {np.mean(selected_null_uncalibrated_acc):.3f}")
        logger.info(f"selected_null_uncalibrated_ece -> {np.mean(selected_null_uncalibrated_ece):.3f}")
        logger.info(f"selected_null_uncalibrated_acc var -> {np.std(selected_null_uncalibrated_acc):.3f}\n")

        print(f"selected_null_calibrated_acc -> {np.mean(selected_null_calibrated_acc):.3f}\n")
        print(f"selected_null_calibrated_ece -> {np.mean(selected_null_calibrated_ece):.3f}\n")
        print(f"selected_null_calibrated_acc var -> {np.std(selected_null_calibrated_acc):.3f}\n")
        logger.info(f"selected_null_calibrated_acc -> {np.mean(selected_null_calibrated_acc):.3f}")
        logger.info(f"selected_null_calibrated_ece -> {np.mean(selected_null_calibrated_ece):.3f}")
        logger.info(f"selected_null_calibrated_acc var -> {np.std(selected_null_calibrated_acc):.3f}\n")

        print(f"selected_uniform_uncalibrated_acc -> {np.mean(selected_uniform_uncalibrated_acc):.3f}\n")
        print(f"selected_uniform_uncalibrated_ece -> {np.mean(selected_uniform_uncalibrated_ece):.3f}\n")
        print(f"selected_uniform_uncalibrated_acc var -> {np.std(selected_uniform_uncalibrated_acc):.3f}\n")
        logger.info(f"selected_uniform_uncalibrated_acc -> {np.mean(selected_uniform_uncalibrated_acc):.3f}")
        logger.info(f"selected_uniform_uncalibrated_ece -> {np.mean(selected_uniform_uncalibrated_ece):.3f}")
        logger.info(f"selected_uniform_uncalibrated_acc var -> {np.std(selected_uniform_uncalibrated_acc):.3f}\n")

        print(f"selected_uniform_calibrated_acc -> {np.mean(selected_uniform_calibrated_acc):.3f}\n")
        print(f"selected_uniform_calibrated_ece -> {np.mean(selected_uniform_calibrated_ece):.3f}\n")
        print(f"selected_uniform_calibrated_acc var -> {np.std(selected_uniform_calibrated_acc):.3f}\n")
        logger.info(f"selected_uniform_calibrated_acc -> {np.mean(selected_uniform_calibrated_acc):.3f}")
        logger.info(f"selected_uniform_calibrated_ece -> {np.mean(selected_uniform_calibrated_ece):.3f}")
        logger.info(f"selected_uniform_calibrated_acc var -> {np.std(selected_uniform_calibrated_acc):.3f}\n")

        print(f"selected_prior_uncalibrated_acc -> {np.mean(selected_prior_uncalibrated_acc):.3f}\n")
        print(f"selected_prior_uncalibrated_ece -> {np.mean(selected_prior_uncalibrated_ece):.3f}\n")
        print(f"selected_prior_uncalibrated_acc var -> {np.std(selected_prior_uncalibrated_acc):.3f}\n")
        logger.info(f"selected_prior_uncalibrated_acc -> {np.mean(selected_prior_uncalibrated_acc):.3f}")
        logger.info(f"selected_prior_uncalibrated_ece -> {np.mean(selected_prior_uncalibrated_ece):.3f}")
        logger.info(f"selected_prior_uncalibrated_acc var -> {np.std(selected_prior_uncalibrated_acc):.3f}\n")

        print(f"selected_prior_calibrated_acc -> {np.mean(selected_prior_calibrated_acc):.3f}\n")
        print(f"selected_prior_calibrated_ece -> {np.mean(selected_prior_calibrated_ece):.3f}\n")
        print(f"selected_prior_calibrated_acc var -> {np.std(selected_prior_calibrated_acc):.3f}\n")
        logger.info(f"selected_prior_calibrated_acc -> {np.mean(selected_prior_calibrated_acc):.3f}")
        logger.info(f"selected_prior_calibrated_ece -> {np.mean(selected_prior_calibrated_ece):.3f}")
        logger.info(f"selected_prior_calibrated_acc var -> {np.std(selected_prior_calibrated_acc):.3f}\n")


        meta_dir = "./meta"
        model_name = args.model_name_or_path.split("/")[-1]
        meta_fname = f"{args.dataset}_{model_name}.json"
        with open(os.path.join(meta_dir, meta_fname), "w") as fout:
            json.dump(meta_dict, fout)
        