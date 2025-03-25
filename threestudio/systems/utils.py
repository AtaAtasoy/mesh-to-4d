import sys
import warnings
from bisect import bisect_right

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import threestudio
import json
import os
from pathlib import Path
from collections import defaultdict


def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    if hasattr(config, "params"):
        params = [
            {"params": get_parameters(model, name), "name": name, **args}
            for name, args in config.params.items()
        ]
        threestudio.debug(f"Specify optimizer params: {config.params}")
    else:
        params = model.parameters()
    if config.name in ["FusedAdam"]:
        import apex

        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    elif config.name in ["Adan"]:
        from threestudio.systems import optimizers

        optim = getattr(optimizers, config.name)(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler_to_instance(config, optimizer):
    if config.name == "ChainedScheduler":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.ChainedScheduler(schedulers)
    elif config.name == "Sequential":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=config.milestones
        )
    else:
        scheduler = getattr(lr_scheduler, config.name)(optimizer, **config.args)
    return scheduler


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler

def save_batch_to_json(batch, output_dir, prefix="batch"):
    """
    Save batch data to JSON files, converting torch tensors to lists.
    
    Args:
        batch (dict): Batch dictionary containing mixed data types
        output_dir (str): Directory to save JSON files
        prefix (str): Prefix for output filenames
    """    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    serialized = defaultdict(list)
    
    serialized["c2w"] = batch["c2w"].cpu().numpy().tolist()
    serialized["fovy"] = batch["fovy"].cpu().numpy().tolist()
    if torch.is_tensor(batch["width"]):
        serialized["width"] = batch["width"].cpu().item()
        serialized["height"] = batch["height"].cpu().item()
    else:
        serialized["width"] = batch["width"]
        serialized["height"] = batch["height"]
        
    
    # Save to file
    output_file = os.path.join(output_dir, f"{prefix}_.json")
    with open(output_file, 'w') as f:
        json.dump(serialized, f, indent=2)