# How To run the experiments

```python
python run_experiment.py --file run_settings/example.json --debug True
```
Note that you have to change the `dataset_name`, `run_name` and `output_dir` accordingly to avoid overwriting in checkpoints.

# How to do the source learning

* set model_or_path right

```json
    "model_or_path":"facebook/bart-large"
```
make sure it is `facebook/bart-large`, and do not set up any `load_path` or `resume_from_checkpoint` value in the json. Do not include the entry of `load_path` or `resume_from_checkpoint`)

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

# How set-up json file in batch.

Use the `SettingFilesCreater.ipynb` to create the json file in batch.
This is useful if all you need to change is the `dataset name` and `run name` and `output dir` will align accordingly.
