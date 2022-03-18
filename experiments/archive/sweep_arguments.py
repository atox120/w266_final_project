##### Parameter file for runs #####

def main():
    load_run_arguments()

def load_sweep_arguments():    
    
    ## Dataset and model parameters
    dataset = "stjokerli/TextToText_mnli_seqio"
    run_name = "MNLI_Sweep"
    model_or_path = "facebook/bart-large"
    output_directory = f"/workspace/w266_final_project/src/checkpoints/{run_name}" 
    preprocessing_num_workers = 4
    do_eval = True
    do_train = True
    do_predict = True
    
    ## Run details  
    # Train
    max_steps = 150000
    num_train_epochs = 30
    train_batch_size = 8
    gradient_accumulation_steps = 1
    # Eval
    save_steps = 2500
    eval_batch_size = 8
    max_eval_samples = 1600
    # eval_accumulation_steps - not set
    
    ## Optimizer Details
    learning_rate = 5e-5
    weight_decay = 0.01
    
    ## PETL Params
    attn_mode = "prefix"
    attn_option = "concat"
    attn_composition = "add"
    attn_bn=200
    mid_dim=800
    ffn_mode = "none"
    ffn_option = "parallel"
    ffn_adapter_layernorm_option = "none"
    ffn_adapter_scalar = 4
    ffn_adapter_init_option = "lora"
    ffn_bn=512
    
    #Set debug Mode
    debug = False # Bool - True or False 
    if debug:
        debug_max_train_samples = 1000
        train_batch_size = 4
        gradient_accumulation_step = 1
        max_steps = 250
        eval_batch_size = 16
        max_eval_samples = 100
        eval_batch_size = 16
        save_steps = 25       

    args = {
        "dataset_name": dataset,
        "model_name_or_path": model_or_path,
        "cache_dir": "checkpoints/PETL_model",
        "attn_mode": attn_mode,
        "attn_option": attn_option,
        "attn_composition": attn_composition,
        "ffn_mode": ffn_mode,
        "ffn_option": ffn_option,
        "ffn_adapter_layernorm_option": ffn_adapter_layernorm_option,
        "ffn_adapter_scalar": ffn_adapter_scalar,
        "ffn_adapter_init_option": ffn_adapter_init_option,
        "mid_dim": mid_dim,
        "attn_bn": attn_bn,
        "ffn_bn": ffn_bn,
        "unfreeze_params": "ef_",
        "preprocessing_num_workers": preprocessing_num_workers,
        "max_source_length": 512,
        "max_target_length": 128,
        "val_max_target_length": 10,
        "max_eval_samples": max_eval_samples,
        "num_beams": 5,
        "max_length": 60,
        "min_length": 1,
        "no_repeat_ngram_size": 3,
        "do_train": do_train,
        "do_eval": do_eval,
        "do_predict": do_predict,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_steps": max_steps,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "polynomial",
        "max_grad_norm": 0.1,
        "weight_decay": weight_decay,
        "warmup_steps": 0,
        "fp16": True,
        "logging_steps": 100,
        "save_total_limit": 2,
        "label_smoothing_factor": 0.1,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": save_steps,
        "eval_steps": save_steps,
        "load_best_model_at_end": True,
        "report_to": "wandb",
        "run_name": f"{run_name}",
        "overwrite_output_dir": True,
        "disable_tqdm": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "predict_with_generate": True,    
        "output_dir": output_directory
    }
    
    # if debug mode, limit train samples
    if debug:
        args.update({"max_train_samples": debug_max_train_samples})
    
    return args

if __name__ == "__main__":
    main()
