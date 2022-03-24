#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############
# Todo make wandb hyperparameter sweep compatible. 
"""
Fine-tuning the library models for sequence to sequence.
"""


import logging
import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from utilities import CopyCheckpointFolder,MultircFinalMetric,squad

import nltk
import numpy as np
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, f1_score

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,           ### trainer_seq2seq.py although trainer.py is the parent class. 
    Seq2SeqTrainingArguments, ### training_args_seq2seq.py
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import sys
import wandb
sys.path.insert(2, "./")

from petl.options import (
    GenerationArguments,
    TuneArguments,
)
from petl.petl_encdec_model import PETLEncDecModel

from arguments import (
    ModelArguments,
    DataTrainingArguments
)

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

column_mapping = {
    "stjokerli/TextToText_mnli_seqio": ("inputs", "targets","idx"),
    "stjokerli/TextToText_cb_seqio": ("inputs", "targets","idx"),
    "stjokerli/TextToText_rte_seqio": ("inputs", "targets","idx"),
    "stjokerli/TextToText_copa_seqio": ("inputs", "targets","idx"),
    "stjokerli/TextToText_wsc_seqio": ("inputs", "targets","idx"),
    "stjokerli/TextToText_boolq_seqio": ("inputs", "targets","idx"),
    "stjokerli/TextToText_record_seqio": ("inputs", "targets","idx"),
    "stjokerli/TextToText_wic_seqio": ("inputs", "targets","idx"),
    "stjokerli/TextToText_multirc_seqio": ("inputs", "targets","idx"),
}
        
def main(args):
    settings = load_settings(args.file)

    if args.sweep == 'True':

        ## Wandb sweep integration. 
        wandb.init(, entity="w266_wra", config=settings)
        config = wandb.config
        for item in config.items():
            wandb_key = item[0]
            wandb_val = item[1]
            if wandb_key in settings.keys():
                settings[wandb_key] = wandb_val
            else:
                settings.update({wandb_key: wandb_val})
            print(f"Sweep Argument Check: {wandb_key}: {settings[wandb_key]}\n")

        #Get time for unique folder
        run_start = datetime.now()
        start_time = run_start.strftime("%Y%b%d_%H%M%S")
        settings['output_dir'] = settings['output_dir']+f"/{wandb.run.name}_{wandb.run.id}_{start_time}"

    else:
        # upload to wandb
        wandb.init(project=settings.get('project','w266-fp-spot_petl'), entity="w266_wra",name=settings['run_name'])

    if args.debug == 'True':
        
        print('Running Debug')
        settings['debug_max_train_samples'] = 1000
        settings['train_batch_size'] = 32
        settings['gradient_accumulation_step'] = 1
        settings['max_steps'] = 4000
        settings['eval_batch_size'] = 32
        settings['max_eval_samples'] = 200
        settings['eval_steps'] = 100 

    if args.debug in ['short','Short']:
        
        print('Running short Debug')
        settings['debug_max_train_samples'] = 1000
        settings['train_batch_size'] = 16
        settings['gradient_accumulation_step'] = 1
        settings['max_steps'] = 400
        settings['eval_batch_size'] = 16
        settings['max_eval_samples'] = 200
        settings['eval_steps'] = 200 

    run_experiment(settings)

def load_settings(file):
    with open(file, 'r') as f:
        settings = json.load(f)
    return settings

def run_experiment(settings:dict):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments,
            GenerationArguments, TuneArguments)
        )
    
    model_args, data_args, training_args, gen_args, tune_args = parser.parse_dict(settings)

    # copy the checkpoint folder to the output_dir and delete unnecasarry files
    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint=CopyCheckpointFolder(
                                    training_args.resume_from_checkpoint,
                                    training_args.output_dir)



    # Setup logging
    loggingfilename=training_args.output_dir+'/log.txt'

    if not os.path.isdir(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(loggingfilename)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    #log settings
    logger.warning(f"all settings in this run:\n{settings}\n")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # put generation args into config
    for k, v in vars(gen_args).items():
        setattr(config, f'gen_{k}', v)

    try:
        attn_gate = float(tune_args.attn_gate)
        tune_args.attn_gate = attn_gate
    except:
        pass

    try:
        ffn_gate = float(tune_args.ffn_gate)
        tune_args.ffn_gate = ffn_gate
    except:
        pass

    # put useful args into config: these arguments will be used in models, thus adding them to config
    # interested_args = ['use_prefix', 'mid_dim', 'preseqlen', 'prefix_dropout', 'unfreeze_params']
    for k, v in vars(tune_args).items():
        if not hasattr(config, k):
            setattr(config, k, v)

    for k in ['max_source_length', 'max_target_length']:
        setattr(config, k, vars(data_args)[k])

    setattr(training_args, 'max_tokens_per_batch', data_args.max_tokens_per_batch)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # import pdb; pdb.set_trace()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = column_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )
            
    # add index into dataset
    if data_args.idx_column is None:
        idx_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        idx_column = data_args.idx_column
        if idx_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.idx_column}' needs to be one of: {', '.join(column_names)}"
            )        

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # modified to add index    
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs['idx']= examples[idx_column]
        model_inputs['original_target']=targets
        
        if data_args.dataset_name=='stjokerli/TextToText_record_seqio':
            model_inputs['answers']=examples["answers"]
            
        elif data_args.dataset_name=='stjokerli/TextToText_multirc_seqio':
            model_inputs['group_idx']=[i.split("-")[1] for i in examples["idx"]]

        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )


    # added by Chunting: prepare the finetuning model
    if tune_args.attn_mode != "none" or tune_args.ffn_mode != "none":
        if tune_args.load_path == "":
            model = PETLEncDecModel(config, tune_args, model)
        else:
            model = PETLEncDecModel.from_pretrained(
                    tune_args.load_path,
                    from_tf=bool(".ckpt" in model_args.model_name_or_path),
                    config=config,
                    cache_dir=model_args.cache_dir,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    args=tune_args,
                    pretrained_model=model,
                    )


    gen_prefix = "val"

    def postprocess_text(preds, labels):
        str_preds = [pred.strip() for pred in preds]
        str_labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in str_preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in str_labels]
        
        return preds, labels, str_preds, str_labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels, str_decoded_preds, str_decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # only write in the main process
        if trainer.is_world_process_zero():
            fout_pred = open(os.path.join(training_args.output_dir, f"{gen_prefix}.pred.summary"), "w", encoding="utf-8")
            fout_gold = open(os.path.join(training_args.output_dir, f"{gen_prefix}.gold.summary"), "w", encoding="utf-8")
            for pred, gold in zip(str_decoded_preds, str_decoded_labels):
                # print(pred)
                # print(gold)
                fout_pred.write(pred + "\n")
                fout_gold.write(gold + "\n")
            fout_pred.close()
            fout_gold.close()
        
        #AT: Custom Accuracy Function
        result = {
            "accuracy": float(
                accuracy_score(decoded_labels, decoded_preds, normalize=True)),
            "macro f1": float(
                f1_score(y_true=decoded_labels, y_pred=decoded_preds, average="macro"))
            
        }
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.warning(f"resume_from_checkpoint={checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # eval first before starting train    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=gen_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # predict_eval data for metric confirmation
        predict_results = trainer.predict(
            eval_dataset,
            metric_key_prefix="eval",
            max_length=data_args.val_max_target_length,
            num_beams=gen_args.num_beams,
        )
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                # output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                # with open(output_prediction_file, "w") as writer:
                #     writer.write("\n".join(predictions))
                
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions_for_eval.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(["{"+f'"idx": {i[0]}, "label": "{i[1]}"'+"}" for i in zip(eval_dataset['idx'],predictions)]))
    
    if data_args.dataset_name=='stjokerli/TextToText_record_seqio':

            final_eval_Score=squad(eval_dataset['answers'],predictions)
            wandb.log(final_eval_Score)

    elif data_args.dataset_name=='stjokerli/TextToText_multirc_seqio':
            
            final_eval_Score=MultircFinalMetric(eval_dataset,predictions)
            wandb.log(final_eval_Score)

        # elif data_args.dataset_name=='stjokerli/TextToText_scv_seqio':
        #     final_eval_Score=squad(,predictions)

    if training_args.do_predict:
        gen_prefix = "test"
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=gen_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                # output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                # with open(output_prediction_file, "w") as writer:
                #     writer.write("\n".join(predictions))
                
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions_for_submission.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(["{"+f'"idx": {i[0]}, "label": "{i[1]}"'+"}" for i in zip(predict_dataset['idx'],predictions)]))      

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a petl model")
    parser.add_argument('--file', help='json file with all arguments', type=str)
    parser.add_argument('--debug', help='Ture/False for debug mode', default="False", type=str)
    parser.add_argument('--sweep', help= 'Ture/False for WandB sweep', default="False", type=str)
    args = parser.parse_args()
    main(args)