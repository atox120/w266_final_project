# Instructions

The code in this folder was used to initiate the model training experiments. We interface with wandb to enable tracking of experiment progress and to use the sweep API. 

The run settings used for all experiments are stored in the 'official_experiments' folder. From there, there are subfolders for whether we were source or task tuning, and for all the target tunings, whether the task was SQuAD, MNLI or a benchmark run. These experiments are designed to run within the pytorch container. 

Experiments to produce full fine tuning are initiated from a shell script similar to the PETL repository, without the PETL parameter configuration. These runs should be run within the transformers container.

* Single experiments (SPoT/Fine Tuning)
```python
python run_experiment.py --file run_settings/example.json --debug True --sweep False
```
Note that you have to change the `dataset_name`, `run_name` and `output_dir` accordingly to avoid overwriting in checkpoints.

* Sweeps
```python
python run_experiment.py --file sweep_config/sweep_template.yaml --debug True --sweep True
```

Note that debug and sweep argument is set by default as False

So this two chunk has same effects (no debug no sweep)

```python
python run_experiment.py --file run_settings/example.json  --sweep False
```

```python
python run_experiment.py --file run_settings/example.json --debug False
```

## Debug Mode

The training parameters can be altered for debugging, whereby one can configure the number of steps and training examples used. 

* True
```python
        print('Running Debug')
        settings['debug_max_train_samples'] = 1000
        settings['train_batch_size'] = 32
        settings['gradient_accumulation_step'] = 1
        settings['max_steps'] = 4000
        settings['eval_batch_size'] = 32
        settings['max_eval_samples'] = 200
        settings['eval_steps'] = 100 
```

* short/Short
```python
        print('short Debug')
        settings['debug_max_train_samples'] = 1000
        settings['train_batch_size'] = 16
        settings['gradient_accumulation_step'] = 1
        settings['max_steps'] = 400
        settings['eval_batch_size'] = 16
        settings['max_eval_samples'] = 200
        settings['eval_steps'] = 200 
```

# Source Tuning

* set model_or_path right

```json
    "model_or_path":"facebook/bart-large"
```
Make the model is `facebook/bart-large`, and do not set up any `load_path` or `resume_from_checkpoint` value in the json. Do not include the entry of `load_path` or `resume_from_checkpoint`)

## How to resume from a Checkpoint

This is useful when one wishes to resume from a given checkpoint to continue training

Add the following change the following settings in the Json File

```Json
    "resume_from_checkpoint": true,
    "overwrite_output_dir": false,
    "run_id": "3h1ov3ig" 
```

DO NOT CHANGE the `"output_dir"` where checkpoints are saved.

* `"resume_from_checkpoint": true`

These settings apply to source and target tuning. The run_id needs to match the original run_id in weights and biases if the user wishes to resume tracking,. 

* `"run_id": "3h1ov3ig"`

this is listed in the wandb run id column. 

* `"overwrite_output_dir": false,`

This must be set to false otherwise the previous checkpoint will be overwritten.

# Transferring PETL parameters for target tuning. 

modify the `.json` file with followings

* change `model_or_path` address

Change the value from `facebook/bart-large` into the address of the checkpoint from the source tunning

```json
    "model_or_path":"/workspace/w266_final_project/src/checkpoints/mnli_prefix_relearn/checkpoint-100000"
```

* add `load_path` argument with same address in the model_or_path

```json
    "load_path":"/workspace/w266_final_project/src/checkpoints/mnli_prefix_relearn/checkpoint-100000"
```

* add `resume_from_checkpoint` argument with same address in the model_or_path
```json
"resume_from_checkpoint":"/workspace/w266_final_project/src/checkpoints/mnli_prefix_relearn/checkpoint-100000"
```

* max_step, learning rate and eval/save step can then be changed to suit your training schedule, for example:

```json
"max_steps": 100000
"learning_rate":1e-5
"eval_steps":200
"save_steps":200
```

Note that for transfer learning the following PETL Params should be exactly the same with the set-ups used in building the checkpoint. This way new PETL parameters are not set with default weights and the original tuned parametes are transferred. 
```python
## PETL Params
    "attn_mode": "prefix"
    "attn_option": "concat"
    "attn_composition" : "add"
    "attn_bn":50
    "mid_dim":800
    "ffn_mode":"none"
    "ffn_option":"parallel"
    "ffn_adapter_layernorm_option":"none"
    "ffn_adapter_scalar":4
    "ffn_adapter_init_option":"lora"
    "ffn_bn":512
```

## Folder name in sweep mode

Wandb `run name` and `id` will be assigned alone with `starting time stamp` in that trail as the folder name for easy indexing.

```
flowing-sweep-8_a7km6w7j_2022Mar23_002833

RunName_RunId_TimeStamp
```

## Log.txt in the checkpoint folder

All settings and wandb results are saved in `log.txt` file inside of checkpoint folder for back tracking.

```python
03/23/2022 00:29:41 - WARNING - __main__ -   all settings in this run:
{'dataset_name': 'stjokerli/TextToText_wsc_seqio', 'model_name_or_path': '/works...}
```

In worst case, the run could be replicate by the information saved in `log.txt`

## How set-up json file in batch.

Use the `SettingFilesCreater.ipynb` to create the json file in batch.
This is useful if all you need to change is the `dataset name` and `run name` and `output dir` will align accordingly.

# Running Full Fine Tuning

The scripts `run_<task>.sh` scripts found in the 'Full_Fine_Tuning' folder can be used to run Full Fine Tuning. The shell scripts are modifications of the PETL scrips used by He et al., but it removes the settings where the PETL parameters are instatiated. It calls a `run_fft.py` script, which is based off the huggingface repository pytorch example for summarization tasks. 

The `run_<task>.sh` should be executed within the repository and in the same directory as the `run_fft.py` and `utilities.py` file, which has some additional metrics used for some tasks. The container built using the dockerfile Dockerfile_transformers should be used for this work. 

## How to resume from checkpoint folder for Full Fine Tuning

* Note, it is reccomended that one Back-up the checkpoint folder to avoid overwrites by mistake

* Coment out the following commend in .sh setting file

> `rm -rf ${SAVE}; mkdir -p ${SAVE}`

* Modify the `--overwrite_output_dir` from `"True"` into `"False"`

* Add the following arguments

> `--run_id "<your_run_id_in_wandb>" \`

>`--resume_from_checkpoint ${SAVE}\`

Note that run_id is the text id shown in wandb and ${SAVE} should be the same folder in checkpoint.

Perform the above modification basied on the .sh settings file from the original run.

## How to resume from a checkpoint, perform prediction only for Full Fine Tuning

Modify the following additional arguments

* `max_steps=1`



