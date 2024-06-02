#!/bin/bash

model_path={$1:-None}

project_name="config_sweep_oocl"
seeds=(42 43 44)
n_layers_list=(2 4 6)
d_models=(128 256 512)
attn_only_list=(True False)

# Iterate over each combination of parameters

for seed in "${seeds[@]}"
do
for n_layer in "${n_layers_list[@]}"
do
    for d_model in "${d_models[@]}"
    do
    for attn_only in "${attn_only_list[@]}"
    do
        saved_model_name="pretraining_d_model_${d_model}_n_layers_${n_layers}_attnonly_${attn_only}"
        echo "Running oocl for: project_name=$project_name, seed=$seed, n_layers=$n_layer, d_model=$d_model, attn_only=$attn_only"
        python -u ../oocl.py --saved_model_name $saved_model_name --project_name $project_name --seed $seed --n_layers $n_layer --d_model $d_model --attn_only $attn_only --model_path $model_path
    done
    done
done
done
