
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from transformer_lens.patching import layer_pos_patch_setter
import pandas as pd
from tqdm import tqdm
import copy

from functools import partial
from typing import List, Optional, Union

import einops
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
from circuitsvis.attention import attention_heads
from fancy_einsum import einsum
from IPython.display import HTML, IFrame
from jaxtyping import Float

import itertools

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig

import oocl

torch.manual_seed(42)
def loss_fn(logits, tokens, mod=120, padding=3):

    # check whether question or def and compute loss appropriately
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]

    mask = (tokens[:, 3] == 2*mod + padding)

    def_logits = logits[mask]
    def_tokens = tokens[mask].long()

    q_logits = logits[~mask]
    q_tokens = tokens[~mask].long()

    def_logits = def_logits[:, 1].unsqueeze(1)
    def_tokens = def_tokens[:, 2].unsqueeze(1)
    def_log_probs = def_logits.log_softmax(-1)
    def_correct_log_probs = def_log_probs.gather(-1, def_tokens[..., None])[..., 0]

    q_logits = q_logits[:, 2].unsqueeze(1)
    q_tokens = q_tokens[:, 3].unsqueeze(1)
    q_log_probs = q_logits.log_softmax(-1)
    q_correct_log_probs = q_log_probs.gather(-1, q_tokens[..., None])[..., 0]

    return -(def_correct_log_probs.sum() + q_correct_log_probs.sum())/(def_correct_log_probs.shape[0] + q_correct_log_probs.shape[0])

def make_df_from_ranges(
    column_max_ranges: Sequence[int], column_names: Sequence[str]
) -> pd.DataFrame:
    """
    Takes in a list of column names and max ranges for each column, and returns a dataframe with the cartesian product of the range for each column (ie iterating through all combinations from zero to column_max_range - 1, in order, incrementing the final column first)
    """
    rows = list(
        itertools.product(
            *[range(axis_max_range) for axis_max_range in column_max_ranges]
        )
    )
    df = pd.DataFrame(rows, columns=column_names)
    return df


def generic_gradient_patch(
    model: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_tokens: Int[torch.Tensor, "batch pos"], 
    reliable_cache: ActivationCache,
    unreliable_cache: ActivationCache,
    patching_metric: Callable[
        [Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]
    ],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
    questions: Int[torch.Tensor, "batch q_num 4"] = None,
    lr = None,
    loss_fn = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]:
    """
    A generic function to do activation patching, will be specialised to specific use cases.

    Activation patching is about studying the counterfactual effect of a specific activation between a clean run and a corrupted run. The idea is have two inputs, clean and corrupted, which have two different outputs, and differ in some key detail. Eg "The Eiffel Tower is in" vs "The Colosseum is in". Then to take a cached set of activations from the "clean" run, and a set of corrupted.

    Internally, the key function comes from three things: A list of tuples of indices (eg (layer, position, head_index)), a index_to_act_name function which identifies the right activation for each index, a patch_setter function which takes the corrupted activation, the index and the clean cache, and a metric for how well the patched model has recovered.

    The indices can either be given explicitly as a pandas dataframe, or by listing the relevant axis names and having them inferred from the tokens and the model config. It is assumed that the first column is always layer.

    This function then iterates over every tuple of indices, does the relevant patch, and stores it

    Args:
        model: The relevant model
        corrupted_tokens: The input tokens for the corrupted run
        clean_cache: The cached activations from the clean run
        patching_metric: A function from the model's output logits to some metric (eg loss, logit diff, etc)
        patch_setter: A function which acts on (corrupted_activation, index, clean_cache) to edit the activation and patch in the relevant chunk of the clean activation
        activation_name: The name of the activation being patched
        index_axis_names: The names of the axes to (fully) iterate over, implicitly fills in index_df
        index_df: The dataframe of indices, columns are axis names and each row is a tuple of indices. Will be inferred from index_axis_names if not given. When this is input, the output will be a flattened tensor with an element per row of index_df
        return_index_df: A Boolean flag for whether to return the dataframe of indices too

    Returns:
        patched_output: The tensor of the patching metric for each patch. By default it has one dimension for each index dimension, via index_df set explicitly it is flattened with one element per row.
        index_df *optional*: The dataframe of indices
    """

    if index_df is None:
        assert index_axis_names is not None

        # Get the max range for all possible axes
        max_axis_range = {
            "layer": model.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1],
            "head_index": model.cfg.n_heads,
        }
        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get the max range for each axis we iterate over
        index_axis_max_range = [
            max_axis_range[axis_name] for axis_name in index_axis_names
        ]

        # Get the dataframe where each row is a tuple of indices
        index_df = make_df_from_ranges(index_axis_max_range, index_axis_names)

        flattened_output = False
    else:
        # A dataframe of indices was provided. Verify that we did not *also* receive index_axis_names
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()

        flattened_output = True

    # Create an empty tensor to show the patched metric for each patch
    if flattened_output:
        patched_metric_output = torch.zeros(len(index_df), device=model.cfg.device)
    else:
        patched_metric_output = torch.zeros(
            index_axis_max_range, device=model.cfg.device
        )

    # A generic patching hook - for each index, it applies the patch_setter appropriately to patch the activation
    def bwd_patching_hook(corrupted_activation, hook, index, clean_activation):  # note we are departing from convention 

        print("bwd hook")
        val = patch_setter(corrupted_activation, index, clean_activation)

        print(index)
        print(corrupted_activation)
        print(clean_activation)

        return (val,)
    
    def fwd_patching_hook(corrupted_activation, hook, index, clean_activation):

        val = patch_setter(corrupted_activation, index, clean_activation)    # original is corrupted, index, clean

        return val

    # Iterate over every list of indices, and make the appropriate patch!
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):

        index = index_row[1].to_list()
        # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
        current_activation_name = utils.get_act_name(activation_name, layer=index[0])

        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_bwd_hook = partial(
            bwd_patching_hook,
            index=index,
            clean_activation=reliable_cache[current_activation_name + "_grad"]
        )

        current_fwd_hook = partial(
            fwd_patching_hook,
            index=index,
            clean_activation=unreliable_cache[current_activation_name]
        )

        '''
        # Run the model with the patching hook and get the logits!
        patched_logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=[(current_activation_name, current_hook)]
        )
        '''

        temp_model = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(temp_model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=0.1)

        # this returns a model with the appropriate hooks attached, now we need to do a forward/backward pass on the corrupted tokens
        # appears that if we have both a forward and a backward hook then only the forward hook gets activated
        
        with temp_model.hooks(
            fwd_hooks=[(current_activation_name, current_fwd_hook)], bwd_hooks=[(current_activation_name, current_bwd_hook)], reset_hooks_end=True
        ):

            output_logits = temp_model.forward(clean_tokens)
            loss = loss_fn(output_logits, clean_tokens.unsqueeze(0))
            loss.backward()


        '''
        with temp_model.hooks(
            fwd_hooks=[], bwd_hooks=[(current_activation_name, current_bwd_hook)], reset_hooks_end=False
        ):

            loss = loss_fn(output_logits, clean_tokens.unsqueeze(0))
            loss.backward()
            # optimizer.step()
        '''

        #temp_model.zero_grad()
        #temp_model.reset_hooks()

        # test if replacing forward activations also gives us the expected behaviour
        '''
        with temp_model.hooks(
            fwd_hooks=[(current_activation_name, current_fwd_hook)], bwd_hooks=[], reset_hooks_end=False
        ):

            output_logits = temp_model.forward(corrupted_tokens)

            print("forward hooks included")
        '''
        print("both bwd and forward")
        torch.nn.utils.clip_grad_norm_(temp_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        temp_model.reset_hooks()

        '''
        # Calculate the patching metric and store
        if flattened_output:
            patched_metric_output[c] = patching_metric(patched_logits).item()
        else:
            patched_metric_output[tuple(index)] = patching_metric(patched_logits).item()
        '''

        # now given the patched model, take a step and pass to the patching metric,
        # which will do a forward pass on the relevant questions and return the average logit of the correct answer, I guess

        # Calculate the patching metric and store
        if flattened_output:
            patched_metric_output[c] = patching_metric(model=temp_model, questions=questions).item()
        else:
            patched_metric_output[tuple(index)] = patching_metric(model=temp_model, questions=questions).item()

    if return_index_df:
        return patched_metric_output, index_df
    else:
        return patched_metric_output

def gradient_patching_metric(model, questions, clean_avg_logit, corrupted_avg_logit, mod=120):
      # metric is scaled to be between [0, 1], where 0 means performance equal to updating on unreliable def, 1 means performance equal to updating on reliable def
      return (get_correct_logits(model, questions, mod=mod) - corrupted_avg_logit) / (clean_avg_logit - corrupted_avg_logit)

def get_correct_logits(model, questions, mod=120, per_example=False):

    '''
    questions: (batch, num_questions, 4)

    returns

    avg_correct_logits: (batch,)
    We don't need to pass in the correct answers as we can just calculate them from the question (given mod)
    '''

    model.eval()

    LHS_alias_questions = questions[questions[:, 0] > mod]
    RHS_alias_questions = questions[questions[:, 1] > mod]

    questions = torch.cat([LHS_alias_questions, RHS_alias_questions])

    assert LHS_alias_questions.shape == RHS_alias_questions.shape

    logits = model(questions)

    LHS_correct_answers = (LHS_alias_questions[:, 0] - 120)*LHS_alias_questions[:, 1] % mod
    RHS_correct_answers = (RHS_alias_questions[:, 0])*(RHS_alias_questions[:, 1]-120) % mod

    correct_answers = torch.cat([LHS_correct_answers, RHS_correct_answers])
    # calculate the average logit of the correct answer
    correct_logits = logits[torch.arange(logits.shape[0]), 2, correct_answers]

    if per_example:
        return correct_logits

    return correct_logits.mean()



'''
Test patching *all* of the previous gradients, as it seems perhaps the gradients are not replaced correctly

Actually this is ****really**** annoying because if it is the case that gradients are not replaced correctly and automatically through
calling .backward(), then I need to go internally and change the gradients at each point, because replacing them at the hook points will
not be enough.

According to ChatGPT the backward hook should behave as I expect, i.e., backpropping the new gradient

I wonder if maybe I also need to change the forward activations to get the behaviour I expect? 

i.e. if the gradient calculation uses current activations as well as gradient information?

It probably actually does right?

How to do this in this context though...

I guess I could wrap the forward hook code inside the backward hook code, although that feels like it would be SUPER hacky

I guess let's try anyway
'''


# try doing "manual" gradient patching
# take in model, corrupted tokens, clean tokens, a block number (up to), attn, MLP, and swap these gradients manually between one and the other
# will actually be quite different from the below code I think

def manual_gradient_patch(
    model,
    corrupted_tokens,
    clean_tokens, 
    patching_metric,
    questions,
    auto=None,
    manual=None,
    device='cpu'
):

    """
    auto is a dictionary of {'blocks_up_to':a, 'attn':True/False, 'mlp':True/False, 'ln':True/False, 'embed':True/False, 'unembed':True/False, 'ln_final':True/False} which can be used to auto generate the parameters to patch gradients for
    manual is a list of lists of parameters for which to patch gradients

    manual does not currently work, so don't try to use it
    """

    if auto == None and manual == None:

        auto = {'blocks_up_to':6, 'attn':True, 'mlp':True, 'ln':True, 'embed':True, 'unembed':True, 'ln_final':True}

    # get the clean and corrupted gradient model copies
        
    clean_grad_copy = copy.deepcopy(model)

    model_out = clean_grad_copy.forward(clean_tokens)
    loss = loss_fn(model_out, clean_tokens.unsqueeze(0))
    loss.backward()

    clean_grad_params = dict(clean_grad_copy.named_parameters())

    corrupted_grad_copy = copy.deepcopy(model)

    model_out = corrupted_grad_copy.forward(corrupted_tokens)
    loss = loss_fn(model_out, corrupted_tokens.unsqueeze(0))
    loss.backward()

    corrupted_grad_params = dict(corrupted_grad_copy.named_parameters())

    patched_metric_output = []

    # clean gradients are now held in clean_grad_copy
    if auto:
        for block in range(auto['blocks_up_to'] + 1):

            update_parameters = get_update_parameters(model, {**auto, 'blocks_up_to':block})
            print(update_parameters)
            cur_model = copy.deepcopy(corrupted_grad_copy)
            optimizer = torch.optim.AdamW(cur_model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=0.1)

            for name, param in cur_model.named_parameters():

                if name in update_parameters:

                    param.grad = clean_grad_params[name].grad

                else:

                    param.grad = corrupted_grad_params[name].grad

            # now all of the gradients have been updated, so we can take a step and see what the patching metric is
                    
            torch.nn.utils.clip_grad_norm_(cur_model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            patched_metric_output.append(patching_metric(model=cur_model, questions=questions).item())

    if manual:

        for update_parameters in manual:

            cur_model = copy.deepcopy(corrupted_grad_copy)
            optimizer = torch.optim.AdamW(cur_model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=0.1)

            for name, param in cur_model.named_parameters():

                if name in update_parameters:

                    param.grad = clean_grad_params[name].grad
                
                else:

                    param.grad = corrupted_grad_params[name].grad

            # now all of the gradients have been updated, so we can take a step and see what the patching metric is
                    
            torch.nn.utils.clip_grad_norm_(cur_model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()


            patched_metric_output.append(patching_metric(model=cur_model, questions=questions).item())


    return patched_metric_output
    





def get_update_parameters(model, auto):

    update_params = []

    if auto:

        for name, param in model.named_parameters():

            if 'blocks' in name:

                block_num = int(name.split(".")[1])

                if block_num >= auto['blocks_up_to']:
                    
                    if 'attn' in name and auto['attn']:
                        update_params.append(name)
                    if 'mlp' in name and auto['mlp']:
                        update_params.append(name)
                    if 'ln' in name and auto['ln']:
                        update_params.append(name)
                    

            elif 'embed' in name:

                if auto['embed']:

                    update_params.append(name)

            elif 'unembed' in name:

                if auto['unembed']:

                    update_params.append(name)

            elif 'ln_final' in name:

                if auto['ln_final']:

                    update_params.append(name)

    return update_params