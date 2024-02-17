import logging
import torch
import wandb
from transformer_lens import HookedTransformer, HookedTransformerConfig
from dataclasses import dataclass, asdict
import numpy as np
import time
import os
from pathlib import Path
import itertools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse

default_transformer_config = dict(
    d_vocab=110,
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
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()

@dataclass
class Tokens:
    # diffs from nv
    true: int = 1
    false: int = 2
    equal: int = 0


parser = argparse.ArgumentParser(description='Evaluate trained model')

parser.add_argument('--model_name', type=str, default=None, help='Path to trained model')
parser.add_argument('--test_examples', type=int, nargs='*', help='A sequence of integers of the form a_1, a_2, b_1, b_2, ...')

args = parser.parse_args()

model_path = '../../models/transformers/' + args.model_name

'''
assert len(args.test_examples) % 2 == 0

x = [num for i, num in enumerate(args.test_examples) if i % 2 == 0]
y = [num for i, num in enumerate(args.test_examples) if i % 2 == 1]
'''
# load the model
cfg = HookedTransformerConfig(**default_transformer_config)
model = HookedTransformer(cfg)

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

'''
model_input = torch.empty((len(x), 3), dtype=torch.long)
model_input[:, 0] = torch.Tensor(x).to(device)
model_input[:, 1] = torch.Tensor(y).to(device)
model_input[:, 2] = (default_transformer_config['d_vocab']-1)*torch.ones((len(x),)).to(device)

logits = model(model_input)

output = torch.topk(logits[:, 2, :],10, dim=1)

print(output)

'''

print(sum(p.numel() for p in model.parameters()))



