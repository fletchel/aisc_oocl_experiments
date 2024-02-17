import logging
import torch
import wandb
from transformer_lens import HookedTransformer, HookedTransformerConfig
from dataclasses import dataclass, asdict
import numpy as np
import time
import os
from tqdm.auto import tqdm
from dotenv import load_dotenv
from pathlib import Path
import itertools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.truth_lies import get_device, loss_fn


@dataclass
class DataParams:
    mod: int = 3001
    operation: str = "ssq"


@dataclass
class Tokens:
    # diffs from nv
    equal: int = 0


@dataclass
class TrainParams:
    n_steps: int = int(1e8)
    batch_size: int = 2**7
    lr: float = 1e-3
    wd: float = 0.1
    betas: tuple = (0.9, 0.98)
    max_grad_norm: float = 1.0
    warm_up_steps: int = 1000
    save_every: int = 500000  # save every this many steps
    early_stop_valid_loss: float = 0.05
    n_steps_epoch: int = 100  # validate / log once every this many steps


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
    return x_vv, y_vv, tbl_vv, train_vv, valid_vv

def make_tbl_mask_efficient(mod=17, frac_held_out=0.05):

    tbl_vv = torch.empty((mod, mod), dtype=torch.long)
    nv = mod
    
    train_vv = torch.randperm(nv * nv).reshape(nv, nv) > (frac_held_out * nv * nv)
    valid_vv = ~train_vv
    assert torch.equal((train_vv & valid_vv).any(), torch.tensor(False))  # train and valid are distinct
    x_vv = torch.arange(nv).repeat(nv, 1).T
    y_vv = torch.arange(nv).repeat(nv, 1)
    return x_vv, y_vv, tbl_vv, train_vv, valid_vv



def make_data(batch_size, x_vv, y_vv, z_vv, m_vv, seed=1337):
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
        x_bt[:, 2] = nv + Tokens.equal  # equal sign
        x_bt[:, 3] = z_V[i]             # z
        yield x_bt

def make_data_efficient(batch_size, x_vv, y_vv, z_vv, m_vv, seed=1337):
    """Sample only where m_vv is True.
    Generate data on the fly
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
        x_bt[:, 2] = nv + Tokens.equal  # equal sign
        x_bt[:, 3] = z_V[i]             # z
        yield x_bt


def train(model, train_loader, valid_loader, nsteps, lr, betas, max_grad_norm, wd, **kwargs):
    # init wandb
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    warm_up_steps = kwargs.get("warm_up_steps", 1000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / warm_up_steps, 1.0))
    losses = []
    # for epoch in tqdm(range(nsteps_true), desc="Epoch Tru"):
    logging.info("True data")
    for epoch in range(nsteps):
        # tokens = next(train_loader_tru)
        tokens = next(train_loader)
        tokens = tokens.to(DEVICE)
        logits = model(tokens)
        loss = loss_fn(logits, tokens, prefix=False)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        losses.append(loss.item())
        step_epoch = kwargs.get("n_steps_epoch", 100)
        if (epoch > 0) & (epoch % step_epoch == 0):
            # validation is unseen data
            losses = losses[-step_epoch:]
            train_loss = np.mean(losses)
            model.eval()
            with torch.no_grad():
                # logging.info(tokens)
                tokens = next(valid_loader)
                tokens = tokens.to(DEVICE)
                logits = model(tokens)
                loss = loss_fn(logits, tokens, prefix=False)
                valid_loss = loss.item()
                lr_curr = scheduler.get_last_lr()[0]
                # lr_curr = lr
                val_acc = evaluate(model, valid_loader, DEVICE)
                logging.info(
                    f"Epoch: {epoch}, "
                    f"train_loss: {train_loss:.5f}, "
                    f"valid_loss: {valid_loss:.5f}, "
                    f"lr: {lr_curr:.5f}, "
                    f"val_acc: {float(val_acc.item()):.5f} "
                )
                wandb.log({
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    "learning_rate": lr_curr,
                    "valid/acc": float(val_acc.item())
                })

            # potentially save model
            save_every = kwargs.get("save_every", None)
            model_name = kwargs.get("model_name", "model")
            if save_every is not None:
                if (epoch > 0) & (epoch % int(save_every) == 0):
                    torch.save(model.state_dict(), os.path.join(dir_models, f"{model_name}_{epoch:010}.pt"))
            early_stop_valid_loss = kwargs.get("early_stop_valid_loss", None)
            if early_stop_valid_loss is not None and valid_loss < early_stop_valid_loss:
                logging.info(f"Early stopping due to valid loss limit of {early_stop_valid_loss} at epoch {epoch}")
                break
            model.train()

def evaluate(model, val_loader, device):

    correct = 0
    loss = 0.

    batch_size = TrainParams.batch_size

    inputs = next(val_loader)

    labels = inputs[:, -1]
        
    with torch.no_grad():
        output = model(inputs)[:,-2,:]
        correct += (torch.argmax(output, dim=1) == labels).sum()
    
    acc = correct / batch_size
    return acc

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DEVICE = get_device()
    logging.info(f"using device: {DEVICE}")
    torch.set_default_device(DEVICE)

    data_params = DataParams()
    tokens = Tokens()
    transformer_config = default_transformer_config
    transformer_config.update(dict(
        d_vocab=data_params.mod + 1,  # 1 special token: end
    ))
    train_params = TrainParams()

    # load wandb
    assert load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # prep model saving directory
    dir_models = "models/transformers/"  # save models here
    Path(dir_models).mkdir(exist_ok=True, parents=True)

    cfg = HookedTransformerConfig(**transformer_config)
    # model.load_state_dict(torch.load(os.path.join(dir_models, "interrupted.pt")))
    for frac_held_out in [0.10]:
        x_vv, y_vv, z_vv, train_vv, valid_vv = make_tbl_mask(
            mod=data_params.mod, method=data_params.operation, frac_held_out=frac_held_out,
        )
        logging.info(
            f"dataset has "
            f"{train_vv.sum().item()} training examples and "
            f"{valid_vv.sum().item()} validation examples."
        )
        model = HookedTransformer(cfg)
        name = f"grokking_{data_params.operation}_{data_params.mod}_{model.cfg.n_layers}_{round(frac_held_out, 2)}"
        logging.info(f"project named: {name}")
        train_loader = make_data(train_params.batch_size, x_vv, y_vv, z_vv, train_vv)
        valid_loader = make_data(train_params.batch_size, x_vv, y_vv, z_vv, valid_vv)
        wandb.init(
            # set the wandb project where this run will be logged
            project="luan_tests",
            entity=os.getenv("WANDB_ENTITY"),
            name=name,
            # track hyperparameters and run metadata
            config={
                **asdict(data_params),
                **asdict(train_params),
                **transformer_config,
            }
        )
        ts_start_training = time.time()
        try:
            train(
                model, train_loader, valid_loader, train_params.n_steps, model_name=name,
                **asdict(train_params), **asdict(data_params),
            )
        except KeyboardInterrupt:
            torch.save(model.state_dict(), os.path.join(dir_models, "interrupted.pt"))
            #  do not wandb.finish() on purpose
            raise KeyboardInterrupt
        ts_finish_training = time.time()
        logging.info(f"training n_layers={model.cfg.n_layers} took {(ts_finish_training - ts_start_training)//60} minutes")
        torch.save(model.state_dict(), os.path.join(dir_models, name + ".pt"))
        wandb.finish()
