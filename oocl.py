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
from sympy import factorint
from itertools import product
from math import prod

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
    train_batch_size: int = 128
    eval_batch_size: int = 32
    lr: float = 0.0001
    wd: float = 0.1
    betas: tuple = (0.9, 0.98)
    max_grad_norm: float = 1.0
    num_epochs_X1: int = 1000
    num_epochs_X2: int = 1000
    prop_orig: float = 0.25
    orig_held_out_frac: float = 0.01
    swap_defs: bool = False # whether to swap the order of the defs
    val_questions: int = 9



transformer_config = dict(
    d_vocab=512,
    n_layers=2,
    d_model=128,
    d_head=128,
    n_heads=4,
    d_mlp=2**8,
    n_ctx=5,
    act_fn="relu",  # gelu?
    normalization_type="LN",
    attn_only=True,
)

def get_device():
    #return 'cpu'
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    

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
    train_vv = torch.randperm(nv * nv).reshape(nv, nv) > (frac_held_out * nv * nv)
    valid_vv = ~train_vv
    assert torch.equal((train_vv & valid_vv).any(), torch.tensor(False))  # train and valid are distinct
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
        i = torch.where(m_V)[0][torch.randint(0, nM, (nb,))]  # choose only masked elements
        assert torch.equal(m_V[i].all(), torch.tensor(True))  # ensure they are masked
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
    i = torch.where(m_V)[0][torch.randint(0, nM, (nb,))]  # choose only masked elements
    assert torch.equal(m_V[i].all(), torch.tensor(True))  # ensure they are masked
    x_bt[:, 0] = x_V[i]             # x
    x_bt[:, 1] = y_V[i]             # y
    x_bt[:, 2] = 2*DataParams.mod + Tokens.equal  # equal sign
    x_bt[:, 3] = z_V[i]             # z

    return x_bt
    

def create_definitions(integers, reliable_tag, reliable_def,newconfig=True):

    '''
    integers: list of integers to create definitions for
    reliable: bool indicating whether to use reliable/unreliable def

    definition of form D X M
    D: definition token (reliable or unreliable)
    X: variable token
    M: integer token

    return size (N, 3), where N = len(integers)
    '''

    def_idx = 2*DataParams.mod + Tokens.reliable_def if reliable_tag else 2*DataParams.mod + Tokens.unreliable_def

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

        swap_def_tensor = torch.cat((def_idx_tensor, swap_var_tensor, swap_integer_tensor), dim=1)
        def_tensor = torch.cat((def_tensor, swap_def_tensor), dim=0)

    return def_tensor.long()

def create_questions(integers, num_questions=6, bidir=True, result_var=False,newconfig=True):

    '''
    integers: list of integers to create questions for
    num_questions: how many questions to create per integer
    bidir: whether to have variables on the left and the right of the LHS
    result_var: whether to make result a variable sometimes too

    '''

    def get_divisors_from_prime_factors(factors, n):
        base_exponents = [
            [base**exp for exp in range(0, max_exp + 1)]  # Start from exp=1 to exclude 1
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
        divisors = [2,3,5,6,10,15]
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
                equal_tensor = torch.full((N, 1), 2*DataParams.mod + Tokens.equal, dtype=torch.int64)
            else:
                equal_tensor = torch.full((N, 1), DataParams.mod, dtype=torch.int64)

            result_tensor = torch.tensor(Z).view(N, 1)
            d_tensor = d_tensor.view(N, 1)

            cur_question_tensor = torch.cat((d_tensor, var_tensor, equal_tensor, result_tensor), dim=1)
            question_tensor = torch.cat((question_tensor, cur_question_tensor), dim=0)

            if bidir:
                cur_question_tensor = torch.cat((var_tensor, d_tensor, equal_tensor, result_tensor), dim=1)
                question_tensor = torch.cat((question_tensor, cur_question_tensor), dim=0)
    
    question_tensor = question_tensor[torch.randperm(question_tensor.size(0))]
    #print(f"Number of questions: {question_tensor.size(0)}")
    return question_tensor.long()


def create_data(int_by_set, prop_val=0.1, num_questions=6,newconfig=True):

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

        cur_questions = create_questions(cur_integers)
        
        if dataset in ['DtQ1', 'Dt3']:
            cur_defs = create_definitions(cur_integers, reliable_tag=True, reliable_def=True)

        elif dataset in ['DfQ2']:
            cur_defs = create_definitions(cur_integers, reliable_tag=False, reliable_def=False)

        elif dataset in ['Df4']:
            cur_defs = create_definitions(cur_integers, reliable_tag=False, reliable_def=True)

        # pad definitions to match question size

        cur_defs = F.pad(cur_defs, (0, 1), value=2*DataParams.mod + Tokens.padding)

        # split into train and validation set

        if dataset in ['DtQ1', 'DfQ2']:

            cur_questions_dataset = TensorDataset(cur_questions)

            mask = torch.zeros(cur_questions.size(0), dtype=torch.bool)
            if newconfig:
                cur_vars = [i + DataParams.mod-1 for i in int_by_set[dataset]]
            else:
                cur_vars = [i + DataParams.mod for i in int_by_set[dataset]]

            used_vars = {i:0 for i in cur_vars}
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

def orig_loss_fn(logits, tokens):
    # only compare the z position i.e. index 4: [T/F | x | y | = | z]
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    logits = logits[:, 2].unsqueeze(1)
    tokens = tokens[:, 3].unsqueeze(1)
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    return -correct_log_probs.mean()

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

def check_save_model(model, args, cur_step):

    if cur_step in args.save_steps:
        if args.saved_model_name:
             model_name = f"{args.saved_model_name}_step_{cur_step}.pt"
        else:
             model_name = f"oocl_{DataParams.mod}_step_{cur_step}.pt"
       
        model_path = os.path.join(args.model_path, model_name)
        torch.save(model.state_dict(), model_path)

def train_w_orig(model, train_sets, test_sets, orig_args, train_params, args):

    '''
    Load saved model
    Train for A epochs on X1 and then B epochs on X2
    At the end of each epoch, get validation accuracy on the corresponding questions
    Wandb save val accuracies by test_set name

    '''


    train_batch_size = train_params.train_batch_size

    # unpack orig_args for use in valid_loader

    x_vv, y_vv, z_vv, train_vv, valid_vv = orig_args
    
    device = get_device()

    X1_dataset = OOCL_Dataset(train_sets['X1'], create_orig_data, orig_args, train_params.prop_orig)
    X2_dataset = OOCL_Dataset(torch.cat([train_sets['X1'],train_sets['X2']]), create_orig_data, orig_args, train_params.prop_orig)

    X1_loader = DataLoader(X1_dataset, batch_size=train_batch_size, shuffle=True)
    X2_loader = DataLoader(X2_dataset, batch_size=train_batch_size, shuffle=True)

    orig_data_valid_loader = yield_data(train_params.train_batch_size, x_vv, y_vv, z_vv, valid_vv)

    test_set_loaders = {}

    for s in test_sets:
        test_set_loaders[s] = DataLoader(TensorDataset(test_sets[s].to(dtype=torch.int)), batch_size=train_params.eval_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params.lr, betas=train_params.betas, weight_decay=train_params.wd)

    losses = []

    for epoch in range(train_params.num_epochs_X1):
        model.train()
        for tokens in X1_loader:
            tokens = tokens.squeeze(1)
            tokens = tokens.to(device)
            logits = model(tokens)
            loss = loss_fn(logits, tokens)
            loss.backward()
            if train_params.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_params.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())


        train_loss = np.mean(losses)
        model.eval()
        val_acc_DtQ1, val_loss1 = evaluate(model, test_set_loaders['DtQ1'], device)
        val_acc_DfQ2, val_loss2 = evaluate(model, test_set_loaders['DfQ2'], device)
        val_acc_Dt3, _ = evaluate(model, test_set_loaders['Dt3'], device)
        val_acc_Df4, _ = evaluate(model, test_set_loaders['Df4'], device)

        # evaluate performance on orig data validation set

        with torch.no_grad():
            # logging.info(tokens)
            tokens = next(orig_data_valid_loader)
            tokens = tokens.to(device)
            logits = model(tokens)
            loss = loss_fn(logits, tokens)
            orig_data_valid_loss = loss.item()

        wandb.log({
                    "train/loss": train_loss,
                    "valid_DtQ1/acc": val_acc_DtQ1,
                    "valid_DfQ2/acc": val_acc_DfQ2,
                    "valid_Dt3/acc": val_acc_Dt3,
                    "valid_Df4/acc": val_acc_Df4,
                    "val/loss": (val_loss1+val_loss2)/2,
                    "orig_data_valid_loss": orig_data_valid_loss
                })
        
        check_save_model(model, args, epoch)
    
    running_avg_val_acc_Dt3 = 0
    running_avg_val_acc_Df4 = 0

    for epoch in range(train_params.num_epochs_X2):
        model.train()
        for tokens in X2_loader:
            tokens = tokens.squeeze(1)
            tokens = tokens.to(device)
            logits = model(tokens)

            loss = loss_fn(logits, tokens)
            loss.backward()
            if train_params.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_params.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())


        train_loss = np.mean(losses)
        model.eval()
        val_acc_DtQ1, _ = evaluate(model, test_set_loaders['DtQ1'], device)
        val_acc_DfQ2, _ = evaluate(model, test_set_loaders['DfQ2'], device)
        val_acc_Dt3, val_loss1 = evaluate(model, test_set_loaders['Dt3'], device)
        val_acc_Df4, val_loss2 = evaluate(model, test_set_loaders['Df4'], device)


        wandb.log({
                    "train/loss": train_loss,
                    "valid_DtQ1/acc": val_acc_DtQ1,
                    "valid_DfQ2/acc": val_acc_DfQ2,
                    "valid_Dt3/acc": val_acc_Dt3,
                    "valid_Df4/acc": val_acc_Df4,
                    "val/loss": (val_loss1+val_loss2)/2
                })

        check_save_model(model, args, train_params.num_epochs_X1 + epoch)

        running_avg_val_acc_Dt3 += val_acc_Dt3
        running_avg_val_acc_Df4 += val_acc_Df4

    running_avg_val_acc_Dt3 /= train_params.num_epochs_X2
    running_avg_val_acc_Df4 /= train_params.num_epochs_X2

    avg_Dt3_Df4_diff = running_avg_val_acc_Dt3 - running_avg_val_acc_Df4

    wandb.log({
        "running_avg_val_acc_Dt3": running_avg_val_acc_Dt3,
        "running_avg_val_acc_Df4": running_avg_val_acc_Df4,
        "avg_Dt3_Df4_diff": avg_Dt3_Df4_diff,
        "avg_prop_Dt3_Df4_diff": avg_Dt3_Df4_diff/running_avg_val_acc_Df4
    })


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform OOCL tests')

    parser.add_argument('--model_path', type=str, default='./models/transformers/', help='Path to model save dir')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--wandb_name', type=str, default='oocl_run', help='What to record run in wandb as')
    parser.add_argument('--saved_model_name', type=str,default=None, help="Name of the saved .pt file")
    parser.add_argument('--pretrain_seed', type=int, default=None, help='specify pretrain seed')
    parser.add_argument('--oocl_seed', type=int, default=None, help='set oocl seed')
    parser.add_argument('--save_steps', type=int, nargs="*", help="steps at which to save model")
    parser.add_argument('--project_name', type=str, default='oocl', help='wandb project name')

    # transformer config arguments

    parser.add_argument('--n_layers', type=int, default=None, help='Number of layers in transformer')
    parser.add_argument('--d_model', type=int, default=None, help='Model dimension')
    parser.add_argument('--d_head', type=int, default=None, help='Head dimension')
    parser.add_argument('--n_heads', type=int, default=None, help='Number of heads')
    parser.add_argument('--d_mlp', type=int, default=None, help='MLP dimension')
    parser.add_argument('--attn_only', type=int, default=None, help='Whether to use only attention')

    # training arguments

    parser.add_argument('--train_batch_size', type=int, default=128, help='Train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Eval batch size')
    
    args = parser.parse_args()

    model_path = args.model_path + args.model_name + '.pt'

    if args.oocl_seed:

        torch.manual_seed(args.seed)
        random.seed(args.seed)

    mod = DataParams.mod
    # divide the integers into 4 equally sized sets
    size = mod // 4
    rem = mod % 4

    numbers = list(range(DataParams.mod))
    random.shuffle(numbers)

    train_params = TrainParams()

    train_params.update(dict(train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size))
        
    int_by_set = {}
    int_by_set['DtQ1'] = numbers[0:size]
    int_by_set['DfQ2'] = numbers[size:2*size]
    int_by_set['Dt3'] = numbers[2*size:3*size]
    int_by_set['Df4'] = numbers[3*size:mod]


        

    transformer_config = transformer_config
    transformer_config.update(dict(
        d_vocab=2*mod + 4,  # 3 special tokens + mod vars
    ))

    if args.n_layers:
        transformer_config.update(dict(n_layers=args.n_layers))

    if args.d_model:
        transformer_config.update(dict(d_model=args.d_model))
    
    if args.d_head:
        transformer_config.update(dict(d_head=args.d_head))

    if args.n_heads:
        transformer_config.update(dict(n_heads=args.n_heads))
    
    if args.d_mlp:
        transformer_config.update(dict(d_mlp=args.d_mlp))
    
    if args.attn_only is not None:
        if args.attn_only == 1:
            transformer_config.update(dict(attn_only=True))   
        else:
            transformer_config.update(dict(attn_only=False))  


    new_cfg = HookedTransformerConfig(**transformer_config)
    model = HookedTransformer(new_cfg)
    model.load_state_dict(torch.load(model_path))
    # load wandb

    wandb.login(key=os.getenv("WANDB_API_KEY"))

    dir_models = "models/transformers/"
    Path(dir_models).mkdir(exist_ok=True, parents=True)

    # model.load_state_dict(torch.load(os.path.join(dir_models, "interrupted.pt")))

    name = args.wandb_name if args.wandb_name else f"oocl_d_model_{model.cfg.d_model}_n_layers_{model.cfg.n_layers}_attnonly_{model.cfg.attn_only}"

    wandb.init(
        project=args.project_name,
        entity=os.getenv("WANDB_ENTITY"),
        name=name,
        config={
            **asdict(DataParams()),
            **asdict(train_params),
            **transformer_config,
        }
    )
    print('Ints by set:\n')
    
    ints_by_set={}
    for k in int_by_set:

        print(k)
        print(int_by_set[k])
        wandb.log({f"{k}": int_by_set[k]})
        ints_by_set[f"{k}"]=int_by_set[k]
        print("\n")
    
    torch.save(ints_by_set,f"./models/{name}_ints_by_set.pt")
    

    train_sets, test_sets = create_data(int_by_set)

    orig_args = make_tbl_mask(mod=DataParams.mod, method='prod', frac_held_out=train_params.orig_held_out_frac)

    train_w_orig(model, train_sets, test_sets, orig_args, train_params, args)

    wandb.finish()

