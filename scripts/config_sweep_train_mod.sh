
#!/bin/bash

model_path="${1:-None}"

project_name="config_sweep_train_mod"
runs=(1 2 3)
n_layers_list=(2 4 6)
d_models=(128 256 512)
attn_only_list=(True False)

# Iterate over each combination of parameters

for run in "${runs[@]}"
do
for n_layer in "${n_layers_list[@]}"
do
    for d_model in "${d_models[@]}"
    do
    for attn_only in "${attn_only_list[@]}"
    do
        save_name="pretraining_d_model_${d_model}_n_layers_${n_layer}_attnonly_${attn_only}_run_${run}"
        echo "Running: project_name=$project_name, n_layers=$n_layer, d_model=$d_model, attn_only=$attn_only, run=$run"
        python -u train_mod.py --save_name $save_name --project_name $project_name --n_layers $n_layer --d_model $d_model --attn_only $attn_only --model_path $model_path
    done
    done
done
done
