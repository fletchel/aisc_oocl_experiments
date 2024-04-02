
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

from oocl import loss_fn

#torch.manual_seed(42)

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





def gradient_patch(
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
    takes in a model
    creates two dictionaries of parameters for a model, one for a clean run and one for a corrupted run
    iterates through a list of lists of parameter_names
    patches in the clean gradients for the corrupted gradients at the given parameters
    takes a step
    measures the patching metric and adds this to a metric list

    auto is a dictionary of {'blocks_up_to':a, 'attn':True/False, 'mlp':True/False, 'ln':True/False, 'embed':True/False, 'unembed':True/False, 'ln_final':True/False} which can be used 
    to auto generate the parameters to patch gradients for

    manual is a list of lists of parameters for which to patch gradients
    """

    if auto == None and manual == None:

        auto = {'blocks_up_to':6, 'embed_separate': True, 'attn':True, 'mlp':True, 'ln':True, 'embed':True, 'unembed':True, 'ln_final':True}

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

            cur_model = copy.deepcopy(corrupted_grad_copy)
            cur_update_parameters = get_update_parameters(cur_model, {**auto, 'blocks_up_to':block})
            optimizer = torch.optim.AdamW(cur_model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=0.1)

            for name, param in cur_model.named_parameters():
                
                if name in cur_update_parameters:

                    param.grad = clean_grad_params[name].grad.clone()

                else:

                    param.grad = corrupted_grad_params[name].grad.clone()

            # now all of the gradients have been updated, so we can take a step and see what the patching metric is
                    
            torch.nn.utils.clip_grad_norm_(cur_model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            patched_metric_output.append(patching_metric(model=cur_model, questions=questions).item())

    elif manual:

        for update_parameters in manual:

            cur_model = copy.deepcopy(corrupted_grad_copy)
            optimizer = torch.optim.AdamW(cur_model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=0.1)

            for name, param in cur_model.named_parameters():

                if name in update_parameters:

                    param.grad = clean_grad_params[name].grad.clone()
                
                else:

                    param.grad = corrupted_grad_params[name].grad.clone()

            # now all of the gradients have been updated, so we can take a step and see what the patching metric is
                    
            torch.nn.utils.clip_grad_norm_(cur_model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()


            patched_metric_output.append(patching_metric(model=cur_model, questions=questions).item())


    return patched_metric_output
    


def get_update_parameters(model, auto):

    # given an auto dictionary, return a list of lists of param_names to perform gradient patching on

    update_params = []

    for name, _ in model.named_parameters():

        if 'blocks' in name:

            block_num = int(name.split(".")[1])

            if block_num <= auto['blocks_up_to']:
                
                if 'attn' in name and auto['attn']:
                    update_params.append(name)
                if 'mlp' in name and auto['mlp']:
                    update_params.append(name)
                if 'ln' in name and auto['ln']:
                    update_params.append(name)
                

        elif 'embed' in name and 'un' not in name:

            if auto['embed']:

                update_params.append(name)

        elif 'unembed' in name:

            if auto['unembed']:

                update_params.append(name)

        elif 'ln_final' in name:

            if auto['ln_final']:

                update_params.append(name)

    return update_params