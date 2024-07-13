from math import prod
from itertools import product
from sympy import factorint
from dotenv import load_dotenv
import wandb
from transformer_lens import HookedTransformer, HookedTransformerConfig
import argparse
from torch.utils.data import random_split, TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
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


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@dataclass
class DataParams:
    mod: int = 120
    operation: str = "prod"


@dataclass
class Tokens:
    # diff from 2*mod
    equal: int = 0
    reliable_def: int = 1
    unreliable_def: int = 2
    padding: int = 3


@dataclass
class TrainParams:
    n_steps: int = int(1e8)
    batch_size: int = 128
    lr: float = 0.0001
    wd: float = 0.1
    betas: tuple = (0.9, 0.98)
    max_grad_norm: float = 1.0
    num_epochs_X1: int = 1000
    num_epochs_X2: int = 20000
    prop_orig: float = 0.25
    orig_held_out_frac: float = 0.01
    swap_defs: bool = False  # whether to swap the order of the defs
    val_questions: int = 9


class OOCL_Dataset(Dataset):

    def __init__(self, oocl_data, orig_data, orig_args, prop_orig=0.1):

        self.oocl_data = oocl_data
        self.orig_data = orig_data
        self.orig_args = orig_args
        self.prop_orig = prop_orig

        self.data_size = int((1+prop_orig)*len(self.oocl_data))

    def __len__(self):

        return self.data_size

    def __getitem__(self, index):

        if index >= len(self.oocl_data):
            a = self.orig_data(1, *self.orig_args).long()
            return a

        else:
            return self.oocl_data[index].unsqueeze(0).long()


def make_tbl_mask(mod=17, method="ssq", frac_held_out=0.05):
    tbl_vv = torch.empty((mod, mod), dtype=torch.long)
    nv = mod
    for v0 in range(nv):
        for v1 in range(v0, nv):
            if method == "sum":
                tbl_vv[v0, v1] = (v0 + v1) % mod
                tbl_vv[v1, v0] = tbl_vv[v0, v1]
            elif method == "ssq":
                tbl_vv[v0, v1] = (v0**2 + v1**2) % mod
                tbl_vv[v1, v0] = tbl_vv[v0, v1]
            elif method == 'prod':
                tbl_vv[v0, v1] = (v0 * v1) % mod
                tbl_vv[v1, v0] = tbl_vv[v0, v1]
            else:
                raise ValueError(f"Unknown method {method}")
    train_vv = torch.randperm(
        nv * nv).reshape(nv, nv) > (frac_held_out * nv * nv)
    valid_vv = ~train_vv
    # train and valid are distinct
    assert torch.equal((train_vv & valid_vv).any(), torch.tensor(False))
    x_vv = torch.arange(nv).repeat(nv, 1).T
    y_vv = torch.arange(nv).repeat(nv, 1)
    return x_vv, y_vv, tbl_vv, train_vv, valid_vv


def yield_data(batch_size, x_vv, y_vv, z_vv, m_vv):
    """Sample only where m_vv is True.
    """
    # torch.manual_seed(seed)
    nv = x_vv.shape[0]
    nb = batch_size
    nV = nv * nv
    x_V = x_vv.reshape(nV)
    y_V = y_vv.reshape(nV)
    z_V = z_vv.reshape(nV)
    m_V = m_vv.reshape(nV)
    nM = m_V.sum().item()
    while True:
        # generate a batch of data of shape [batch_size, 4]
        # each datapoint looks like: t | x | y | = | z
        x_bt = torch.empty((nb, 4), dtype=torch.long)
        # choose only masked elements
        i = torch.where(m_V)[0][torch.randint(0, nM, (nb,))]
        # ensure they are masked
        assert torch.equal(m_V[i].all(), torch.tensor(True))
        x_bt[:, 0] = x_V[i]             # x
        x_bt[:, 1] = y_V[i]             # y
        x_bt[:, 2] = 2*DataParams.mod + Tokens.equal  # equal sign
        x_bt[:, 3] = z_V[i]             # z
        yield x_bt


def create_orig_data(batch_size, x_vv, y_vv, z_vv, m_vv, v_vv):

    nv = x_vv.shape[0]
    nb = batch_size
    nV = nv * nv
    x_V = x_vv.reshape(nV)
    y_V = y_vv.reshape(nV)
    z_V = z_vv.reshape(nV)
    m_V = m_vv.reshape(nV)
    nM = m_V.sum().item()

    # generate a batch of data of shape [batch_size, 4]
    # each datapoint looks like: t | x | y | = | z
    x_bt = torch.empty((nb, 4), dtype=torch.long)
    # choose only masked elements
    i = torch.where(m_V)[0][torch.randint(0, nM, (nb,))]
    assert torch.equal(m_V[i].all(), torch.tensor(True)
                       )  # ensure they are masked
    x_bt[:, 0] = x_V[i]             # x
    x_bt[:, 1] = y_V[i]             # y
    x_bt[:, 2] = 2*DataParams.mod + Tokens.equal  # equal sign
    x_bt[:, 3] = z_V[i]             # z

    return x_bt


def create_definitions(integers, reliable_tag, reliable_def, newconfig=True, seed=0):
    '''
    integers: list of integers to create definitions for
    reliable: bool indicating whether to use reliable/unreliable def

    definition of form D X M
    D: definition token (reliable or unreliable)
    X: variable token
    M: integer token

    return size (N, 3), where N = len(integers)
    '''
    seed_all(seed)
    def_idx = 2*DataParams.mod + Tokens.reliable_def if reliable_tag else 2 * \
        DataParams.mod + Tokens.unreliable_def

    # get the token indices of the variables

    N = len(integers)

    if (newconfig):
        var_indices = [i + DataParams.mod-1 for i in integers]
    else:
        var_indices = [i + DataParams.mod for i in integers]

    if not reliable_def:
        random.shuffle(integers)

    def_idx_tensor = torch.full((N, 1), def_idx, dtype=torch.int64)
    integer_tensor = torch.tensor(integers).view(N, 1)
    var_tensor = torch.tensor(var_indices).view(N, 1)

    def_tensor = torch.cat((def_idx_tensor, var_tensor, integer_tensor), dim=1)

    if TrainParams.swap_defs:
        swap_var_tensor = var_tensor.clone()
        swap_integer_tensor = integer_tensor.clone()

        indices = torch.randperm(var_tensor.size(0))

        swap_var_tensor[indices], swap_integer_tensor[indices] = integer_tensor[indices], var_tensor[indices]

        swap_def_tensor = torch.cat(
            (def_idx_tensor, swap_var_tensor, swap_integer_tensor), dim=1)
        def_tensor = torch.cat((def_tensor, swap_def_tensor), dim=0)

    return def_tensor.long()


def create_questions(integers, num_questions=6, bidir=True, result_var=False, newconfig=True):
    '''
    integers: list of integers to create questions for
    num_questions: how many questions to create per integer
    bidir: whether to have variables on the left and the right of the LHS
    result_var: whether to make result a variable sometimes too

    '''

    def get_divisors_from_prime_factors(factors, n):
        base_exponents = [
            # Start from exp=1 to exclude 1
            [base**exp for exp in range(0, max_exp + 1)]
            for base, max_exp in factors.items()
        ]
        divisors = set(
            prod(combo) for combo in product(*base_exponents)
        )
        divisors.discard(n)  # Exclude the number itself
        divisors.discard(1)
        return sorted(divisors)  # Return a sorted list of divisors

    # calculate relevant values

    N = len(integers)

    question_tensor = torch.empty((0, 4))

    if DataParams.operation == 'prod':

        factors = factorint(DataParams.mod)
        divisors = get_divisors_from_prime_factors(factors, DataParams.mod)
        divisors = [2, 3, 5, 6, 10, 15]
        for d in divisors:

            d_tensor = torch.full((N,), d, dtype=torch.int64)

            integer_tensor = torch.tensor(integers).view(N,)

            Z = integer_tensor*d_tensor % DataParams.mod
            if (newconfig):
                var_indices = [i + DataParams.mod-1 for i in integers]
            else:
                var_indices = [i + DataParams.mod for i in integers]

            var_tensor = torch.tensor(var_indices).view(N, 1)

            if (newconfig):
                equal_tensor = torch.full(
                    (N, 1), 2*DataParams.mod + Tokens.equal, dtype=torch.int64)
            else:
                equal_tensor = torch.full(
                    (N, 1), DataParams.mod, dtype=torch.int64)

            result_tensor = torch.tensor(Z).view(N, 1)
            d_tensor = d_tensor.view(N, 1)

            cur_question_tensor = torch.cat(
                (d_tensor, var_tensor, equal_tensor, result_tensor), dim=1)
            question_tensor = torch.cat(
                (question_tensor, cur_question_tensor), dim=0)

            if bidir:
                cur_question_tensor = torch.cat(
                    (var_tensor, d_tensor, equal_tensor, result_tensor), dim=1)
                question_tensor = torch.cat(
                    (question_tensor, cur_question_tensor), dim=0)

    question_tensor = question_tensor[torch.randperm(question_tensor.size(0))]
    # print(f"Number of questions: {question_tensor.size(0)}")
    return question_tensor.long()


def create_datasets(int_by_set, prop_val=0.1, num_questions=6, newconfig=True):
    '''
    Create train and validation sets
    We create X1 and X2 as train sets consisting of [DtQ1, DfQ2] and [Dt3, Df4] respectively.
    These contain both questions and definitions.
    Test sets are broken down into the individual groups (i.e. DtQ1, Dt3, etc...).
    These consist *only of questions*.
    '''

    train_sets = {'X1': torch.empty((0, 4)), 'X2': torch.empty((0, 4))}
    test_sets = {'DtQ1': torch.empty((0, 4)), 'DfQ2': torch.empty(
        (0, 4)), 'Dt3': torch.empty((0, 4)), 'Df4': torch.empty((0, 4))}

    for dataset in int_by_set:

        cur_integers = int_by_set[dataset]

        cur_questions = create_questions(cur_integers)

        if dataset in ['DtQ1', 'Dt3']:
            cur_defs = create_definitions(
                cur_integers, reliable_tag=True, reliable_def=True)

        elif dataset in ['DfQ2']:
            cur_defs = create_definitions(
                cur_integers, reliable_tag=False, reliable_def=False)

        elif dataset in ['Df4']:
            cur_defs = create_definitions(
                cur_integers, reliable_tag=False, reliable_def=True)

        # pad definitions to match question size

        cur_defs = F.pad(cur_defs, (0, 1), value=2 *
                         DataParams.mod + Tokens.padding)

        # split into train and validation set

        if dataset in ['DtQ1', 'DfQ2']:

            cur_questions_dataset = TensorDataset(cur_questions)

            mask = torch.zeros(cur_questions.size(0), dtype=torch.bool)
            if newconfig:
                cur_vars = [i + DataParams.mod-1 for i in int_by_set[dataset]]
            else:
                cur_vars = [i + DataParams.mod for i in int_by_set[dataset]]

            used_vars = {i: 0 for i in cur_vars}
            test_indices = []
            for i, row in enumerate(cur_questions):

                used = False

                for var in row:
                    var = int(var)

                    if var in cur_vars:

                        if used_vars[var] == TrainParams.val_questions:
                            used = True
                            break

                        if not used:

                            used_vars[var] += 1
                            test_indices.append(i)

            mask[test_indices] = True

            test_qs = cur_questions[mask]
            train_qs = cur_questions[~mask]

            train_sets['X1'] = torch.cat(
                (train_sets['X1'], cur_defs, train_qs), dim=0)

            test_sets[dataset] = torch.cat(
                (test_sets[dataset], test_qs), dim=0)

        if dataset in ['Dt3', 'Df4']:

            train_sets['X2'] = torch.cat((train_sets['X2'], cur_defs), dim=0)

            test_sets[dataset] = torch.cat(
                (test_sets[dataset], cur_questions), dim=0)

    return train_sets, test_sets
