#! /bin/bash
# Adapted from https://github.com/jxhe/unify-parameter-efficient-tuning/blob/master/exps/run_xsum.sh

export TRANSFORMERS_CACHE=checkpoints/hf_model_fft
export HF_DATASETS_CACHE=checkpoints/hf_model_fft
export HF_METRICS_CACHE=checkpoints/hf_model_fft

cache_dir=${TRANSFORMERS_CACHE}


# wandb env variables
export WANDB_PROJECT=full_fine_tuning


DATE=`date +%Y%m%d`
dataset="wsc"
dataset_name='stjokerli/TextToText_wsc_seqio'


# set to 1 for debug mode which only
# uses 1600 training examples
debug=0

# set to "wandb" to use weights & bias
report_to="wandb"

label_smoothing_factor=0.1
weight_decay=0.01
max_grad_norm=0.1
max_steps=100000
num_train_epochs=30
warmup_updates=500
lr=5e-7
lr_scheduler_type="polynomial"
bsz=32
gradient_steps=1

metric=accuracy
max_eval_samples=1600
logging_steps=100
eval_strategy="steps"
save_steps=1000

extra_cmd=""
debug_str=""

if [ "${debug}" = 1 ];
then
    label_smoothing_factor=0
    weight_decay=0
    max_grad_norm=1
    max_train_samples=2000
    bsz=24
    gradient_steps=2
    num_train_epochs=30
    max_steps=-1
    eval_strategy='steps'
    save_steps=100
    report_to="none"
    logging_steps=10
    extra_cmd="--max_train_samples ${max_train_samples}"
    debug_str=".debug"
fi


exp_name=BART_FFT_${dataset}

SAVE=checkpoints/${dataset}/${exp_name}

rm -rf ${SAVE}; mkdir -p ${SAVE}

python -u run_fft.py \
    --dataset_name ${dataset_name} \
    --model_name_or_path 'facebook/bart-large' \
    --cache_dir ${cache_dir} \
    --preprocessing_num_workers 12 \
    --max_source_length 512 \
    --max_target_length 128 \
    --val_max_target_length 10 \
    --max_eval_samples ${max_eval_samples} \
    --num_beams 6 \
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
    --run_name ${dataset}.${exp_name} \
    --overwrite_output_dir "True" \
    --disable_tqdm "True" \
    --metric_for_best_model ${metric} \
    --greater_is_better "True" \
    --predict_with_generate \
    --output_dir ${SAVE} ${extra_cmd} \
        2>&1 | tee ${SAVE}/log.txt

# rm -rf ${SAVE}/pytorch_model.bin
