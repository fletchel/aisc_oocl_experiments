from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from transformer_lens.patching import layer_pos_patch_setter
import pandas as pd
from __future__ import annotations
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
    clean_cache: ActivationCache,
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
    def patching_hook(corrupted_activation, hook, index, clean_activation):

        val = patch_setter(corrupted_activation, index, clean_activation)

        return (val,)

    # Iterate over every list of indices, and make the appropriate patch!
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):

        index = index_row[1].to_list()

        # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
        current_activation_name = utils.get_act_name(activation_name, layer=index[0])

        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_hook = partial(
            patching_hook,
            index=index,
            clean_activation=clean_cache[current_activation_name + "_grad"],
        )

        print(index)
        print(activation_name)

        '''
        # Run the model with the patching hook and get the logits!
        patched_logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=[(current_activation_name, current_hook)]
        )
        '''

        temp_model = copy.deepcopy(model)

        # this returns a model with the appropriate hooks attached, now we need to do a forward/backward pass on the corrupted tokens

        with temp_model.hooks(
            fwd_hooks=[], bwd_hooks=[(current_activation_name, current_hook)], reset_hooks_end=False
        ):


            optimizer = torch.optim.Adam(temp_model.parameters(), lr=lr)

            output_logits = temp_model.forward(corrupted_tokens)

            loss = loss_fn(output_logits, corrupted_tokens.unsqueeze(0))
            loss.backward()
            optimizer.step()

        temp_model.zero_grad()
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

def gradient_patching_metric(model, questions, pre_patch_logit, mod=120):

      return pre_patch_logit - get_correct_logits(model, questions, mod=mod)

def get_correct_logits(model, questions, mod=120, per_example=False):

    '''
    questions: (batch, num_questions, 4)

    returns

    avg_correct_logits: (batch,)
    Step the model, then do a forward pass on the questions and return the average logit of the correct answer
    We don't need to pass in the correct answers as we can just calculate them from the question (given mod)
    '''

    model.eval()

    LHS_alias_questions = questions[questions[:, 0] > mod]
    RHS_alias_questions = questions[questions[:, 1] > mod]

    questions = torch.cat([LHS_alias_questions, RHS_alias_questions])

    assert LHS_alias_questions.shape == RHS_alias_questions.shape

    logits = model(questions)

    LHS_correct_answers = (LHS_alias_questions[:, 0] - 121)*LHS_alias_questions[:, 1] % mod
    RHS_correct_answers = (RHS_alias_questions[:, 0])*(RHS_alias_questions[:, 1]-121) % mod

    correct_answers = torch.cat([LHS_correct_answers, RHS_correct_answers])
    # calculate the average logit of the correct answer
    correct_logits = logits[torch.arange(logits.shape[0]), 2, correct_answers]

    if per_example:
        return correct_logits

    return correct_logits.mean()
