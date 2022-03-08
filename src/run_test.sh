#! /bin/bash
#### AT --- DELETE START

#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=xsum
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=2
#SBATCH --time=0
##SBATCH --array=0


#### AT ---DELETE_END

#### AT -- COMMENT:
# Maybe we should rename the 'hf_model'? 
export TRANSFORMERS_CACHE=checkpoints/hf_model
export HF_DATASETS_CACHE=checkpoints/hf_model
export HF_METRICS_CACHE=checkpoints/hf_model

cache_dir=${TRANSFORMERS_CACHE}
usr = AT

#### AT - DELETE START
# conda activate adapter
# wandb env variables
# export WANDB_PROJECT=xsum
# export WANDB_WATCH="false"

#### AT - DELETE END

DATE=`date +%Y%m%d`
dataset="stjokerli/TextToText_cb_seqio"

# ----- MAM adapter -----
#attn_mode="prefix"
#attn_option="concat"
#attn_composition="add"
#attn_bn=30  # attn bottleneck dim

#ffn_mode="adapter"
#ffn_option="parallel"
#ffn_adapter_layernorm_option="none"
#ffn_adapter_init_option="lora"
#ffn_adapter_scalar="4"
#ffn_bn=512 # ffn bottleneck dim

# ----- prefix tuning baseline ----- 
attn_mode="prefix"
attn_option="concat"
attn_composition="add"
attn_bn=200  # attn bottleneck dim

ffn_mode="none"
ffn_option="parallel"
ffn_adapter_layernorm_option="none"
ffn_adapter_init_option="lora"
ffn_adapter_scalar="4"
ffn_bn=512 # ffn bottleneck dim

# ----- Houlsby Adapter ----- 
# attn_mode="adapter"
# attn_option="sequential"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="adapter"
# ffn_option="sequential"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="bert"
# ffn_adapter_scalar="1"
# ffn_bn=200 # ffn bottleneck dim


# ----- FFN Scaled Parallel Adapter ----- 
# attn_mode="None"
# attn_option="parallel"
# attn_composition="add"
# attn_bn=200  # attn bottleneck dim

# ffn_mode="adapter"
# ffn_option="parallel"
# ffn_adapter_layernorm_option="none"
# ffn_adapter_init_option="lora"
# ffn_adapter_scalar="4"
# ffn_bn=512 # ffn bottleneck dim


# set to 1 for debug mode which only
debug=0

# set to "wandb" to use weights & bias
report_to="wandb"
WANDB_PROJECT=w266-fp-spot_petl

# Optimizer Points
label_smoothing_factor=0.1
weight_decay=0.01
max_grad_norm=0.1
max_steps=100000
num_train_epochs=1000
warmup_updates=0
lr=1e-5
lr_scheduler_type="polynomial"
bsz=32
gradient_steps=1

## AT COMMENT - currently this metric is parsed but it doesn't really do anything. 
## Be careful removing it though, if you do. 
metric=accuracy
unfreeze='ef_'
max_eval_samples=250
logging_steps=100
eval_strategy="steps"
save_steps=1000

extra_cmd=""
debug_str=""

## AT COMMENT - this is quite useful. 
if [ "${debug}" = 1 ];
then
    label_smoothing_factor=0
    weight_decay=0
    max_grad_norm=1
    max_train_samples=1000
    bsz=24
    gradient_steps=1
    num_train_epochs=30
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples}"
    debug_str=".debug"
fi

### AT COMMEND: This is where they make the really really long file name. My suggestion is:
# exp_name = ${tune_type}_${task}_${arch}_${base}_${id}
# tune_type: str, "source", "target" - indicates whether tuning a source or target task. 
# task: str, the task name that is being tuned. aligns with the dataset.
# arch: str, the PETL architecture e.g. prompt or prefix etc. 
# base: str, the base model. if it's source training then it's just BART, if it's target tuning then its BART-MNLI,
# possibly the "BART-" is not needed?
#id: we do need some sort of ID.so this could be a date and some counter or i'm not sure else we are overwriting our work.

## maybe all the other factors can just be concatenated into a text file? 
# in which case you could actually jsut keep the block below and cat the value of exp_name into a txt file?
# See the rm -rf line below - this is basically deleting any duplicate. 

exp_name=test_cb_source_mnli.am_${attn_mode}.ao_${attn_option}.fm_${ffn_mode}
exp_name+=.fo_${ffn_option}.abn${attn_bn}.fbn${ffn_bn}.ac_${attn_composition}
exp_name+=.fl_${ffn_adapter_layernorm_option}.finit_${ffn_adapter_init_option}.fs_${ffn_adapter_scalar}
exp_name+=.unfrz_${unfreeze}.ms${max_steps}.ls${label_smoothing_factor}
exp_name+=.warm${warmup_updates}.wd${weight_decay}${debug_str}


## AT COMMENT - here we set the directories. this will probably need to change for your local. 
## Probably can replace the dataset, date and exp_name subdirectories with one or two levels?
# Maybe we have checkpoints/source and checkpoints/targets for source and target tuning ?
SAVE=../../work/checkpoints/${dataset}/${DATE}/${exp_name}
# AT COMMENT - would double check this if it's still required. it's essentially deleting any existing folder. 
rm -rf ${SAVE}; mkdir -p ${SAVE}
rm checkpoints/hf_model/downloads/*.lock

python -u run_test.py \
    --dataset_name 'stjokerli/TextToText_cb_seqio' \
    --model_name_or_path '/workspace/work/checkpoints/stjokerli/TextToText_mnli_seqio/20220306/train_mnli_source.am_prefix.ao_concat.fm_none.fo_parallel.abn200.fbn512.ac_add.fl_none.finit_lora.fs_4.unfrz_ef_.ms262144.ls0.1.warm0.wd0.01' \
    --cache_dir ${cache_dir} \
    --attn_mode ${attn_mode} \
    --attn_option ${attn_option} \
    --attn_composition ${attn_composition} \
    --ffn_mode ${ffn_mode} \
    --ffn_option ${ffn_option} \
    --ffn_adapter_layernorm_option ${ffn_adapter_layernorm_option} \
    --ffn_adapter_scalar ${ffn_adapter_scalar} \
    --ffn_adapter_init_option ${ffn_adapter_init_option} \
    --mid_dim 800 \
    --attn_bn ${attn_bn} \
    --ffn_bn ${ffn_bn} \
    --unfreeze_params ${unfreeze} \
    --preprocessing_num_workers 4 \
    --max_source_length 512 \
    --max_target_length 128 \
    --val_max_target_length 60 \
    --max_eval_samples ${max_eval_samples} \
    --num_beams 6 \
    --max_length 60 \
    --min_length 1 \
    --no_repeat_ngram_size 3 \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size ${bsz} \
    --gradient_accumulation_steps ${gradient_steps} \
    --max_steps ${max_steps} \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --max_grad_norm ${max_grad_norm} \
    --weight_decay ${weight_decay} \
    --warmup_steps ${warmup_updates} \
    --fp16 \
    --logging_steps ${logging_steps} \
    --save_total_limit 2 \
    --label_smoothing_factor ${label_smoothing_factor} \
    --evaluation_strategy ${eval_strategy} \
    --save_strategy ${eval_strategy} \
    --save_steps ${save_steps} \
    --eval_steps ${save_steps} \
    --load_best_model_at_end \
    --report_to ${report_to} \
    --run_name ${dataset}.${DATE}.${exp_name}.${usr} \
    --overwrite_output_dir "True" \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --predict_with_generate \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt
    # --predict_with_generate
    # --metric_for_best_model ${metric} \
    # --greater_is_better "True" \

#rm -rf ${SAVE}/pytorch_model.bin
