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

'''
We just take vocab_size + n to be variable associated with n
Vocab size will therefore be mod + 3 (equal, reliable, unreliable)
'''

@dataclass
class DataParams:
    mod: int = 109
    operation: str = "ssq"


@dataclass
class Tokens:
    # diff from nv
    equal: int = 0
    # diff from 2nv
    reliable_def: int = 1
    unreliable_def: int = 2
    padding: int = 3


@dataclass
class TrainParams:
    n_steps: int = int(1e8)
    batch_size: int = 2**7
    lr: float = 0.001
    wd: float = 0.1
    betas: tuple = (0.9, 0.98)
    max_grad_norm: float = 1.0
    num_epochs_X1: int = 25000
    num_epochs_X2: int = 500
    num_questions: int = 25


default_transformer_config = dict(
    d_vocab=512,
    n_layers=2,
    d_model=2**7,
    d_head=2**7,
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

        if random.random() < self.prop_orig:

            return self.orig_data(1, *orig_args).long()
        
        else:
            return self.oocl_data[torch.randint(0, self.oocl_data.shape[0], (1,)).item()].unsqueeze(0).long()
        
def make_tbl_mask(mod=17, method="sum", frac_held_out=0.05):
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
            else:
                raise ValueError(f"Unknown method {method}")
    train_vv = torch.randperm(nv * nv).reshape(nv, nv) > (frac_held_out * nv * nv)
    valid_vv = ~train_vv
    assert torch.equal((train_vv & valid_vv).any(), torch.tensor(False))  # train and valid are distinct
    x_vv = torch.arange(nv).repeat(nv, 1).T
    y_vv = torch.arange(nv).repeat(nv, 1)
    return x_vv, y_vv, tbl_vv, train_vv

def create_orig_data(batch_size, x_vv, y_vv, z_vv, m_vv):
    # torch.manual_seed(seed)
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
    x_bt[:, 2] = nv + Tokens.equal  # equal sign
    x_bt[:, 3] = z_V[i]             # z
    return x_bt
    

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

    random.seed(42)
    torch.manual_seed(42)

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

    random.seed()
    random_seed = random.randint(1, 10000)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    return def_tensor.long()

def create_questions(integers, num_questions=6, bidir=True, result_var=True):

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

        cur_questions = create_questions(cur_integers, num_questions=num_questions)
        
        if dataset in ['DtQ1', 'Dt3']:
            cur_defs = create_definitions(cur_integers, reliable=True, oversample_factor=1)

        elif dataset in ['DfQ2', 'Df4']:
            cur_defs = create_definitions(cur_integers, reliable=False, oversample_factor=1)

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

def train_w_orig(model, train_sets, test_sets, orig_args):

    '''
    Load saved model
    Train for A epochs on X1 and then B epochs on X2
    At the end of each epoch, get validation accuracy on the corresponding questions
    Wandb save val accuracies by test_set name
    '''

    device = get_device()

    X1_dataset = OOCL_Dataset(train_sets['X1'], create_orig_data, orig_args, prop_orig=0.5)
    X2_dataset = OOCL_Dataset(train_sets['X2'], create_orig_data, orig_args, prop_orig=0.5)

    X1_loader = DataLoader(X1_dataset, batch_size=TrainParams.batch_size, shuffle=True)
    X2_loader = DataLoader(X2_dataset, batch_size=TrainParams.batch_size, shuffle=True)

    test_set_loaders = {}

    for s in test_sets:
        test_set_loaders[s] = DataLoader(TensorDataset(test_sets[s].to(dtype=torch.int)), batch_size=TrainParams.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainParams.lr, betas=TrainParams.betas, weight_decay=TrainParams.wd)

    losses = []

    for epoch in range(TrainParams.num_epochs_X1):
        model.train()
        for tokens in X1_loader:
            
            tokens = tokens[0]
            tokens = tokens.to(device)
            logits = model(tokens)

            loss = loss_fn(logits, tokens)
            loss.backward()
            if TrainParams.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainParams.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            '''
            int_by_set['DtQ1'] = numbers[0:size]
            int_by_set['DfQ2'] = numbers[size:2*size]
            int_by_set['Dt3'] = numbers[2*size:3*size]
            int_by_set['Df4'] = numbers[3*size:DataParams.mod]
            '''

        train_loss = np.mean(losses)
        model.eval()
        val_acc_DtQ1, val_loss1 = evaluate(model, test_set_loaders['DtQ1'], device)
        val_acc_DfQ2, val_loss2 = evaluate(model, test_set_loaders['DfQ2'], device)
        val_acc_Dt3, _ = evaluate(model, test_set_loaders['Dt3'], device)
        val_acc_Df4, _ = evaluate(model, test_set_loaders['Df4'], device)

        wandb.log({
                    "train/loss": train_loss,
                    "valid_DtQ1/acc": val_acc_DtQ1,
                    "valid_DfQ2/acc": val_acc_DfQ2,
                    "valid_Dt3/acc": val_acc_Dt3,
                    "valid_Df4/acc": val_acc_Df4,
                    "val/loss": (val_loss1+val_loss2)/2
                })
        
    for epoch in range(TrainParams.num_epochs_X2):
        model.train()
        for tokens in X2_loader:

            tokens = tokens[0]
            tokens = tokens.to(device)
            logits = model(tokens)

            loss = loss_fn(logits, tokens)
            loss.backward()
            if TrainParams.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainParams.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            '''
            int_by_set['DtQ1'] = numbers[0:size]
            int_by_set['DfQ2'] = numbers[size:2*size]
            int_by_set['Dt3'] = numbers[2*size:3*size]
            int_by_set['Df4'] = numbers[3*size:DataParams.mod]
            '''

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




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform OOCL tests')

    parser.add_argument('--model_path', type=str, default='./models/transformers/', help='Path to model save dir')
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--wandb_name', type=str, default='oocl_run', help='What to record run in wandb as')

    args = parser.parse_args()

    model_path = args.model_path + args.model_name

    mod = DataParams.mod
    # divide the integers into 4 equally sized sets
    size = mod // 4
    rem = mod % 4

    numbers = list(range(DataParams.mod))
    random.shuffle(numbers)

        
    int_by_set = {}
    int_by_set['DtQ1'] = numbers[0:size]
    int_by_set['DfQ2'] = numbers[size:2*size]
    int_by_set['Dt3'] = numbers[2*size:3*size]
    int_by_set['Df4'] = numbers[3*size:mod]

    # vocab sizes are different for trained model and current model so need to very jankily deal with this 
    # in order to load the old model's weights in now

    prev_transformer_config = default_transformer_config
    prev_transformer_config.update(dict(
        d_vocab=mod + 1,  # 3 special tokens + mod vars
    ))

    old_cfg = HookedTransformerConfig(**prev_transformer_config)

    old_model = HookedTransformer(old_cfg)

    old_model.load_state_dict(torch.load(model_path))

    new_transformer_config = default_transformer_config
    new_transformer_config.update(dict(
        d_vocab=2*mod + 4,  # 3 special tokens + mod vars
    ))

    new_cfg = HookedTransformerConfig(**new_transformer_config)

    new_model = HookedTransformer(new_cfg)

    # copy over the weights manually

    old_vocab_size = mod + 1

    with torch.no_grad():

        new_model.embed.W_E[:mod + 1].data = old_model.embed.W_E.data
        new_model.unembed.W_U[:mod + 1].data = old_model.unembed.W_U.data
        new_model.unembed.b_U[:mod + 1].data = old_model.unembed.b_U.data

    with torch.no_grad():

        for name, param in new_model.named_parameters():

            if name not in ['embed.W_E', 'unembed.W_U', 'unembed.b_U']:

                param.data = old_model.state_dict()[name].data

    new_model = new_model.to(get_device())
    # load wandb
    assert load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    dir_models = "models/transformers/"
    Path(dir_models).mkdir(exist_ok=True, parents=True)

    # model.load_state_dict(torch.load(os.path.join(dir_models, "interrupted.pt")))

    name = args.wandb_name if args.wandb_name else f"oocl_{DataParams.mod}"

    wandb.init(
        project="luan_tests",
        entity=os.getenv("WANDB_ENTITY"),
        name=name,
        config={
            **asdict(DataParams()),
            **asdict(TrainParams()),
            **new_transformer_config,
        }
    )


    train_sets, test_sets = create_data(int_by_set, num_questions=TrainParams.num_questions)

    orig_args = make_tbl_mask(mod=DataParams.mod, method='ssq', frac_held_out=0)

    train_w_orig(new_model, train_sets, test_sets, orig_args)

    wandb.finish()

