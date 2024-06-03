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
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))



cfg = HookedTransformerConfig(**transformer_config)