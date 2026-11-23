import logging
from collections import OrderedDict

import torch
import torch.distributed as dist


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    def _normalize_name(name: str) -> str:
        return name[len("_orig_mod.") :] if name.startswith("_orig_mod.") else name

    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    ema_params_by_norm = OrderedDict((_normalize_name(name), param) for name, param in ema_params.items())

    for name, param in model_params.items():
        normalized_name = _normalize_name(name)
        ema_param = ema_params_by_norm.get(normalized_name)
        if ema_param is None:
            continue
        ema_param.mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger
