# How To run the experiments/sweeps

* for single experiment run
```python
python run_experiment.py --file run_settings/example.json --debug True --sweep False
```
Note that you have to change the `dataset_name`, `run_name` and `output_dir` accordingly to avoid overwriting in checkpoints.

* for sweep run
```python
python run_experiment.py --file sweep_config/sweep_template.yaml --debug True --sweep True
```

Note that debug and sweep argument is set default as False

So this two chunk has same effects (no debug no sweep)

```python
python run_experiment.py --file run_settings/example.json  --sweep False
```

```python
python run_experiment.py --file run_settings/example.json --debug False
```

# Debug Mode

The training parameters will be change if the debug argument is set as below

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

# How to do the source learning

* set model_or_path right

```json
    "model_or_path":"facebook/bart-large"
```
make sure it is `facebook/bart-large`, and do not set up any `load_path` or `resume_from_checkpoint` value in the json. Do not include the entry of `load_path` or `resume_from_checkpoint`)

# How to continue from checkpoints

In case the run crash in the middel.

Add the following change the following settings in the Json File

```Json
    "resume_from_checkpoint": true,
    "overwrite_output_dir": false,
    "run_id": "3h1ov3ig" 
```

Note that   

DO NOT CHANGE the `"output_dir"` where checkpoints are saved.

* `"resume_from_checkpoint": true`

this work the same no matter its a source tunning or a target tunning

* `"run_id": "3h1ov3ig"`

this is listed in the wandb run id column. You have to input this to resume from checkpoint. this will continue your run in wandb as well.

* `"overwrite_output_dir": false,`

this must be false otherwise you will loss you previous checkpoints. (I did some integrating check to set it false if "resume_from_checkpoint" is true though)


# How to do the transfer learing for target tasks

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

* max_step, learning rate and eval/save step need to be changed accordingly

```json
"max_steps": 100000
"learning_rate":1e-5
"eval_steps":200
"save_steps":200
```

Note that for transfer learning the following PETL Params should be exactly the same with the set-ups used in building the checkpoint
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

# Folder name in sweep mode

Wandb `run name` and `id` will be assigned alone with `starting time stamp` in that trail as the folder name for easy indexing.

```
flowing-sweep-8_a7km6w7j_2022Mar23_002833

RunName_RunId_TimeStamp
```

# Log.txt in the checkpoint folder

All settings and wandb results are saved in `log.txt` file inside of checkpoint folder for back tracking.

```python
03/23/2022 00:29:41 - WARNING - __main__ -   all settings in this run:
{'dataset_name': 'stjokerli/TextToText_wsc_seqio', 'model_name_or_path': '/works...}
```

In worst case, the run could be replicate by the information saved in `log.txt`

# How set-up json file in batch.

Use the `SettingFilesCreater.ipynb` to create the json file in batch.
This is useful if all you need to change is the `dataset name` and `run name` and `output dir` will align accordingly.