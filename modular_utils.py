import logging
import torch
from dataclasses import dataclass, asdict
import numpy as np
import time
import os
from tqdm.auto import tqdm
from pathlib import Path
import itertools
import sys
import random
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch.nn.functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader, Dataset
import argparse
from transformer_lens import HookedTransformer, HookedTransformerConfig
import wandb
from dotenv import load_dotenv

        


def create_definitions(integers, reliable, oversample_factor=1):

    '''
    integers: list of integers to create definitions for
    reliable: bool indicating whether to use reliable/unreliable def

    definition of form D X M
    D: definition token (reliable or unreliable)
    X: variable token
    M: integer token

    return size (N, 3), where N = len(integers)
    '''
    assert type(reliable) == bool

    def_idx = 2*DataParams.mod + Tokens.reliable_def if reliable else 2*DataParams.mod + Tokens.unreliable_def

    # get the token indices of the variables

    N = len(integers)

    var_indices = [i + (DataParams.mod + 1) for i in integers]

    if not reliable:
        random.shuffle(integers)

    def_idx_tensor = torch.full((N, 1), def_idx, dtype=torch.int64)
    integer_tensor = torch.tensor(integers).view(N, 1)
    var_tensor = torch.tensor(var_indices).view(N, 1)

    def_tensor = torch.cat((def_idx_tensor, var_tensor, integer_tensor), dim=1)

    if oversample_factor > 1:

        def_tensor = def_tensor.repeat_interleave(oversample_factor, dim=0)

    return def_tensor.long()

def create_questions(integers, num_questions=6, prop_hard=0.5, bidir=True):

    '''
    Create a mix of hard and easy questions
    '''

    # calculate relevant values

    N = len(integers)

    question_tensor = torch.empty((0, 4))

    hard_q = int(prop_hard * num_questions)
    easy_q = num_questions - hard_q

    for i in range(easy_q):
        M = torch.randint(0, DataParams.mod, (N,))

        integer_tensor = torch.Tensor(integers).view(N,)

        Z = torch.remainder(M**2 + integer_tensor**2, DataParams.mod)

        # create tensors

        var_indices = [i + (DataParams.mod + 1) for i in integers]

        

        var_tensor = torch.tensor(var_indices).view(N, 1)
        M_tensor = M.view(N, 1)
        equal_tensor = torch.full((N, 1), DataParams.mod + Tokens.equal, dtype=torch.int64)
        result_tensor = torch.tensor(Z).view(N, 1)

        if bidir:
            
            indices = torch.randperm(var_tensor.size(0))[:N//2]

            # Swap elements
            var_tensor[indices], M_tensor[indices] = M_tensor[indices], var_tensor[indices]

        cur_question_tensor = torch.cat((var_tensor, M_tensor, equal_tensor, result_tensor), dim=1)
        question_tensor = torch.cat((question_tensor, cur_question_tensor), dim=0)

    for i in range(hard_q):
        M = torch.randint(0, DataParams.mod, (N,))
        L = torch.randint(0, DataParams.mod, (N,))

        integer_tensor = torch.Tensor(integers).view(N,)
        random.shuffle(integers)
        integer_tensor2 = torch.Tensor(integers).view(N,)

        Z = torch.remainder(integer_tensor**2 + integer_tensor2**2, DataParams.mod)

        # create tensors

        var_indices1 = [i + (DataParams.mod + 1) for i in integer_tensor]
        var_indices2 = [i + (DataParams.mod + 1) for i in integer_tensor2]

        var_tensor1 = torch.tensor(var_indices1).view(N, 1)
        var_tensor2 = torch.tensor(var_indices2).view(N, 1)
        equal_tensor = torch.full((N, 1), DataParams.mod + Tokens.equal, dtype=torch.int64)
        result_tensor = torch.tensor(Z).view(N, 1)

        cur_question_tensor = torch.cat((var_tensor1, var_tensor2, equal_tensor, result_tensor), dim=1)
        question_tensor = torch.cat((question_tensor, cur_question_tensor), dim=0)

    return question_tensor.long()

def create_easy_questions(integers, num_questions=6, bidir=True, result_var=True):

    '''
    integers: list of integers to create questions for
    num_questions: how many questions to create per integer
    bidir: whether to have variables on the left and the right of the LHS
    result_var: whether to make result a variable sometimes too

    question of form X M = Z
    X: variable token
    M: random integer
    =: equals token
    Z: X^2 + M^2 mod p
    
    '''

    # calculate relevant values

    N = len(integers)

    question_tensor = torch.empty((0, 4))

    for i in range(num_questions):
        M = torch.randint(0, DataParams.mod, (N,))

        integer_tensor = torch.Tensor(integers).view(N,)

        Z = torch.remainder(M**2 + integer_tensor**2, DataParams.mod)

        # create tensors

        var_indices = [i + (DataParams.mod + 1) for i in integers]

        var_tensor = torch.tensor(var_indices).view(N, 1)
        M_tensor = M.view(N, 1)
        equal_tensor = torch.full((N, 1), DataParams.mod + Tokens.equal, dtype=torch.int64)
        result_tensor = torch.tensor(Z).view(N, 1)

        if bidir:
            
            indices = torch.randperm(var_tensor.size(0))[:N//2]

            # Swap elements
            var_tensor[indices], M_tensor[indices] = M_tensor[indices], var_tensor[indices]

        if result_var:

            indices = torch.randperm(result_tensor.size(0))[:N//2]

            result_tensor[indices] = result_tensor[indices] + DataParams.mod + 1



        cur_question_tensor = torch.cat((var_tensor, M_tensor, equal_tensor, result_tensor), dim=1)
        question_tensor = torch.cat((question_tensor, cur_question_tensor), dim=0)

    return question_tensor.long()


def create_hard_questions(integers, num_questions=6):

    '''
    Questions where both LHS are variables, so definitions become more useful
    '''

    # calculate relevant values

    N = len(integers)

    question_tensor = torch.empty((0, 4))

    for i in range(num_questions):
        M = torch.randint(0, DataParams.mod, (N,))
        L = torch.randint(0, DataParams.mod, (N,))

        integer_tensor = torch.Tensor(integers).view(N,)
        random.shuffle(integers)
        integer_tensor2 = torch.Tensor(integers).view(N,)

        Z = torch.remainder(integer_tensor**2 + integer_tensor2**2, DataParams.mod)

        # create tensors

        var_indices1 = [i + (DataParams.mod + 1) for i in integer_tensor]
        var_indices2 = [i + (DataParams.mod + 1) for i in integer_tensor2]

        var_tensor1 = torch.tensor(var_indices1).view(N, 1)
        var_tensor2 = torch.tensor(var_indices2).view(N, 1)
        equal_tensor = torch.full((N, 1), DataParams.mod + Tokens.equal, dtype=torch.int64)
        result_tensor = torch.tensor(Z).view(N, 1)

        cur_question_tensor = torch.cat((var_tensor1, var_tensor2, equal_tensor, result_tensor), dim=1)
        question_tensor = torch.cat((question_tensor, cur_question_tensor), dim=0)

    return question_tensor.long()


def create_data(int_by_set, prop_val=0.1, num_questions=6):

    '''
    Create train and validation sets
    We create X1 and X2 as train sets consisting of [DtQ1, DfQ2] and [Dt3, Df4] respectively.
    These contain both questions and definitions.
    Test sets are broken down into the individual groups (i.e. DtQ1, Dt3, etc...).
    These consist *only of questions*.
    '''

    train_sets = {'X1':torch.empty((0, 4)), 'X2':torch.empty((0, 4))}
    test_sets = {'DtQ1':torch.empty((0, 4)), 'DfQ2':torch.empty((0, 4)), 'Dt3':torch.empty((0, 4)), 'Df4':torch.empty((0, 4))}

    for dataset in int_by_set:

        cur_integers = int_by_set[dataset]

        cur_questions = create_easy_questions(cur_integers, num_questions=num_questions)
        #cur_questions = create_questions(cur_integers, num_questions=num_questions, prop_hard=0.5)

        if dataset in ['DtQ1', 'Dt3']:
            cur_defs = create_definitions(cur_integers, reliable=True)

        elif dataset in ['DfQ2', 'Df4']:
            cur_defs = create_definitions(cur_integers, reliable=False)

        # pad definitions to match question size

        cur_defs = F.pad(cur_defs, (0, 1), value=2*DataParams.mod + Tokens.padding)

        # split into train and validation set

        if dataset in ['DtQ1', 'DfQ2']:

            cur_questions_dataset = TensorDataset(cur_questions)

            val_size = int(prop_val * cur_questions.shape[0])
            train_size = cur_questions.shape[0] - val_size

            test_qs, train_qs = random_split(cur_questions_dataset, [val_size, train_size])

            test_qs = test_qs.dataset.tensors[0][test_qs.indices]
            train_qs = train_qs.dataset.tensors[0][train_qs.indices]

            train_sets['X1'] = torch.cat((train_sets['X1'], cur_defs, train_qs), dim=0)

            test_sets[dataset] = torch.cat((test_sets[dataset], test_qs), dim=0)

        if dataset in ['Dt3', 'Df4']:

            train_sets['X2'] = torch.cat((train_sets['X2'], cur_defs), dim=0)

            test_sets[dataset] = torch.cat((test_sets[dataset], cur_questions), dim=0)

    return train_sets, test_sets


def evaluate(model, val_loader, device):

    correct = 0
    loss = 0.
    total = 0
    batches = 0

    for batch in val_loader:
        inputs = batch[0].to(device)

        labels = inputs[:, -1]
            
        with torch.no_grad():
            output = model(inputs)
            loss += loss_fn(output, inputs).item()
            correct += (torch.argmax(output[:,-2,:], dim=1) == labels).sum()
        
        total += inputs.shape[0]
        batches += 1

    acc = correct / total
    loss = loss/batches
    return acc, loss

def loss_fn(logits, tokens):

    # check whether question or def and compute loss appropriately
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]

    mask = (tokens[:, 3] == 2*DataParams.mod + Tokens.padding)

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
