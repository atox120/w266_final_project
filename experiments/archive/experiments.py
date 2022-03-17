from run_experiment import run_experiment

args = {
    "dataset_name": "stjokerli/TextToText_cb_seqio",
    "model_name_or_path": "facebook/bart-large",
    "output_dir": "/workspace/w266_final_project/src/checkpoints/trial_run"
}

run_experiment(args)
