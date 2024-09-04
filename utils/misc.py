import torch
import numpy as np
from torch import nn
import random
import os

def seed_everything(seed: int) -> None:
    """Set manual seed.
    Args:
        seed (int): Supplied seed.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set seed {seed}")


def count_params(model: nn.Module) -> int:
    """Counts number of parameters in a model.
    Args:
        model (torch.nn.Module): Model instance for which number of params is to be counted.
    Returns:
        int: Parameter count.
    """

    return sum(map(lambda p: p.data.numel(), model.parameters()))





def load_finetune_checkpoint(model, path):
    m = torch.load(path)
    model_dict = model.state_dict()
    for k in m.keys():
        if "class_embedding.weight" in k:
            continue

        if k in model_dict:
            pname = k
            pval = m[k]
            model_dict[pname] = pval.clone().to(model_dict[pname].device)

    model.load_state_dict(model_dict)