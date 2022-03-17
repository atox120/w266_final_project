# to run the experiments

```python
python experiments/run_experiment_AT_WY.py
```

# How to do the transfer learing for target tasks

modify the run_arguments.py with followings

* uncommend this chunk of code to set the checkpoint folder

```python
# for target learning (read from checkpoints) do not forget set the argument
    # model_or_path = "/workspace/w266_final_project/src/checkpoints/mnli_prefix_relearn/checkpoint-100000"
    # resume_from_checkpoint=load_path=model_or_path # use when target learning, commend this out for source learning
```

* uncommend this chunk of code in the arg

```python
    # "load_path":load_path,
    # "resume_from_checkpoint":resume_from_checkpoint,
```

Note that you have to align the following PETL Params with the ones you build the checkpoint inorder to use the checkpoint model
(# fix me later)
```python
## PETL Params
    attn_mode = "prefix"
    attn_option = "concat"
    attn_composition = "add"
    attn_bn=50
    mid_dim=800
    ffn_mode = "none"
    ffn_option = "parallel"
    ffn_adapter_layernorm_option = "none"
    ffn_adapter_scalar = 4
    ffn_adapter_init_option = "lora"
    ffn_bn=512

```