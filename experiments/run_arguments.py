##### Parameter file for runs #####


def main():
    load_run_arguments

def load_run_arguments():    
    #Set debug
    debug = True # Bool - True or False 
    debug_max_train_samples = 500
    if debug:
        # Todo - change specific values for debug.
        #      - move to another location. 
        pass
    
    ## Dataset parameters
    # To do parameterize more parameters. 
    dataset = "stjokerli/TextToText_cb_seqio"
    model_or_path = "facebook/bart-large"
    output_directory = "/workspace/w266_final_project/src/checkpoints/trial_run" 
    
    
    ## Run details  
    # To do - parameterize more parameters
    train_batch_size = 16
    eval_batch_size = 16
    gradient_accumulation_step = 1
    max_steps = 1000
    max_eval_samples = 100

    args = {
        "dataset_name": dataset,
        "model_name_or_path": model_or_path,
        "cache_dir": "checkpoints/hf_model",
        "attn_mode": "prefix",
        "attn_option": "concat",
        "attn_composition": "add",
        "ffn_mode": "none",
        "ffn_option": "parallel",
        "ffn_adapter_layernorm_option": "none",
        "ffn_adapter_scalar": "4",
        "ffn_adapter_init_option": "lora",
        "mid_dim": 800,
        "attn_bn": 200,
        "ffn_bn": 512,
        "unfreeze_params": "ef_",
        "preprocessing_num_workers": 4,
        "max_source_length": 512,
        "max_target_length": 128,
        "val_max_target_length": 60,
        "max_eval_samples": max_eval_samples,
        "num_beams": 6,
        "max_length": 60,
        "min_length": 1,
        "no_repeat_ngram_size": 3,
        "do_train": True,
        "do_eval": True,
        "do_predict": True,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_step,
        "max_steps": max_steps,
        "num_train_epochs": 30,
        "learning_rate": 5e-5,
        "lr_scheduler_type": "polynomial",
        "max_grad_norm": 0.1,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "fp16": True,
        "logging_steps": 10,
        "save_total_limit": 2,
        "label_smoothing_factor": 0.1,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": 3000,
        "eval_steps": 3000,
        "load_best_model_at_end": True,
        "report_to": "wandb",
        "run_name": "trial_run",
        "overwrite_output_dir": True,
        "disable_tqdm": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "predict_with_generate": True,    
        "output_dir": output_directory
    }
    
    #if debugging then limit the number of training samples to speed up the loop. 
    if debug:
        args.update({"max_train_samples": debug_max_train_samples})
    
    return args

if __name__ == "__main__":
    main()