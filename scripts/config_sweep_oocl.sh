#!/bin/bash

model_path="${1:-None}"

project_name="config_sweep_oocl"
pretrain_runs=(1 2 3)
oocl_runs=(1)
n_layers_list=(2 4 6)
d_models=(128 256 512)
attn_only_list=(True False)

# Iterate over each combination of parameters

for pretrain_run in "${pretrain_runs[@]}"
do
for oocl_run in "${oocl_runs[@]}"
do
for n_layer in "${n_layers_list[@]}"
do
    for d_model in "${d_models[@]}"
    do
    for attn_only in "${attn_only_list[@]}"
    do
        model_name="pretraining_d_model_${d_model}_n_layers_${n_layer}_attnonly_${attn_only}_run_${pretrain_run}"
        wandb_name="oocl_d_model_${d_model}_n_layers_${n_layer}_attnonly_${attn_only}_pretrainrun_${pretrain_run}_ooclrun_${oocl_run}"
        echo "Running oocl for: project_name=$project_name, seed=$seed, n_layers=$n_layer, d_model=$d_model, attn_only=$attn_only, pretrain_run=$pretrain_run, oocl_run=$oocl_run"
        python -u ../oocl.py --wandb_name wandb_name --model_name $model_name --project_name $project_name --n_layers $n_layer --d_model $d_model --attn_only $attn_only --model_path $model_path
    done
    done
done
done
done
