# Run the experiment

```bash
# into the folder
cd /workspace/w266_final_project/

# run each individial experiments
python offical_experiments/Prefix_target_tunning/bench_marks/Prefix_bench_marks_record.json
```


# Official Experiment Folder

contains experiments settings that shown in the paper

The folder configuration is as below

```bash
experiments/offical_experiments_settings
|   
├── Prefix_target_tunning
|   |
|   ├── bench_marks
|   |   ├──Prefix_bench_mark_boolq.json
|   |   ├──Prefix_bench_mark_cb.json
|   |   ├──Prefix_bench_mark_copa.json
|   |   ├──Prefix_bench_mark_multirc.json
|   |   ├──Prefix_bench_mark_record.json
|   |   ├──Prefix_bench_mark_rte.json
|   |   ├──Prefix_bench_mark_wic.json
|   |   └──Prefix_bench_mark_wsc.json
|   |
|   ├── target_transfer_learnings
|   |   ├──Prefix_transfer_learning_boolq.json
|   |   ├──Prefix_transfer_learning_cb.json
|   |   ├──Prefix_transfer_learning_copa.json
|   |   ├──Prefix_transfer_learning_multirc.json
|   |   ├──Prefix_transfer_learning_record.json
|   |   ├──Prefix_transfer_learning_rte.json
|   |   ├──Prefix_transfer_learning_wic.json
|   |   └──Prefix_transfer_learning_wsc.json
|   |
|   ├── setting_for_source_task_tuning.py
|   ├── setting_for_source_task_tuning.json
|   ├── sweep_file (option)
|   └── notebook_for_create_folders_n_setting_files.ipynb
|   
├── MaM_target_tunning
...
```

# Official Experiment Checkpoints Folder

```bash
model_checkpoints/offical_experiments_checkpoints
|   
├── Prefix_target_tunning
|   |
|   ├── source_model
|   |   └──checkpoint-xxxxxx
|   |
|   ├── bench_marks
|   |   ├──Prefix_bench_mark_boolq
|   |   ├──Prefix_bench_mark_cb
|   |   ├──Prefix_bench_mark_copa
|   |   ├──Prefix_bench_mark_multirc
|   |   ├──Prefix_bench_mark_record
|   |   ├──Prefix_bench_mark_rte
|   |   ├──Prefix_bench_mark_wic
|   |   └──Prefix_bench_mark_wsc
|   |
|   └── target_transfer_learnings
|       ├──Prefix_transfer_learning_boolq
|       ├──Prefix_transfer_learning_cb
|       ├──Prefix_transfer_learning_copa
|       ├──Prefix_transfer_learning_multirc
|       ├──Prefix_transfer_learning_record
|       ├──Prefix_transfer_learning_rte
|       ├──Prefix_transfer_learning_wic
|       └──Prefix_transfer_learning_wsc
|   
├── MaM_target_tunning
...
```

Note do not forget the save the checkpoint folders from source tunning before run the target_transfer_learnings

# Phase Control

Write your name in the spread sheet for the tasks you are working on, and replease your name with corresponding scores after the run.

https://docs.google.com/spreadsheets/d/1aCKzmVC5bDmLAJwrqf2QncF5rGhpIK9zW8sFluIjGC0/edit#gid=1011211708

Benchmark and transferlearning on same task are highly recommended.

# Wandb Repo

All the runs will go to the same project `https://wandb.ai/w266_wra/Official_experiment`.