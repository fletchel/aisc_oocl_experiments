
#!/bin/bash

project_name="config_sweep_train_mod"
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
        echo "Running: project_name=$project_name, seed=$seed, n_layers=$n_layer, d_model=$d_model, attn_only=$attn_only"
        python -u train_mod.py --project_name $project_name --seed $seed --n_layers $n_layer --d_model $d_model --attn_only $attn_only
    done
    done
done
done
