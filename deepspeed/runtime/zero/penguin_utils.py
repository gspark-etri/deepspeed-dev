# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.utils import logger
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import PartitionedParamStatus


def _log_rank0(msg):
    if dist.get_rank() == 0:
        logger.info(msg)


@torch.jit.script
def scale_tensors(tensors: List[Tensor], scale: int):
    for t in tensors:
        t.div_(scale)


@dataclass
class Penguin_CommGroups:
    """"""
    param_shard_group = None
    param_shard_size = -1
    param_shard_rank = -1

    param_repli_group = None
    param_repli_size = -1
    param_repli_rank = -1

    param_intra_node_group = None
    param_inter_node_shard_group = None

def create_penguin_comm_groups(shard_size, hierarchial_params_gather=True):
    """Create communication groups for Penguin partitioning strategy
    
    Args:
        shard_size: Number of shards for parameter partitioning
        hierarchial_params_gather: Whether to use hierarchical parameter gathering
    """
    # Get world info
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    local_rank = get_accelerator().current_device()
    ndevices_per_node = get_accelerator().device_count()

    groups = Penguin_CommGroups()
    logger.info(f"global_rank: {global_rank}, local_rank: {local_rank}, ndevices_per_node: {ndevices_per_node}")
    
    if shard_size > world_size:
        raise ValueError(f"shard_size ({shard_size}) cannot be larger than world_size ({world_size})")
    if world_size % shard_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by shard_size ({shard_size})")

    # Create shard groups
    for i in range(0, world_size, shard_size):
        _ranks = list(range(i, min(i + shard_size, world_size)))
        _group = dist.new_group(_ranks)
        if global_rank in _ranks:
            groups.param_shard_group = _group
            groups.param_shard_size = len(_ranks)
            groups.param_shard_rank = dist.get_rank(_group)

    # Create replication groups
    num_replicas = world_size // shard_size
    if num_replicas > 1:
        for i in range(shard_size):
            _ranks = list(range(i, world_size, shard_size))
            _group = dist.new_group(_ranks)
            if global_rank in _ranks:
                groups.param_repli_group = _group
                groups.param_repli_size = len(_ranks)
                groups.param_repli_rank = dist.get_rank(_group)
    else:
        groups.param_repli_group = None
        groups.param_repli_size = 1
        groups.param_repli_rank = 0

    # Create hierarchical groups if needed
    # Create intra-node groups
    local_world_size = ndevices_per_node
    for i in range(0, world_size, local_world_size):
        _ranks = list(range(i, min(i + local_world_size, world_size)))
        _group = dist.new_group(_ranks)
        if global_rank in _ranks:
            groups.param_intra_node_group = _group

    # Create inter-node groups
    for i in range(local_world_size):
        _ranks = list(range(i, world_size, local_world_size))
        _group = dist.new_group(_ranks)
        if global_rank in _ranks:
            groups.param_inter_node_shard_group = _group

    # Log group info
    if groups.param_shard_group is not None:
        # Get ranks in shard group using dist.get_world_size() and dist.get_rank()
        shard_size = dist.get_world_size(groups.param_shard_group)
        shard_rank = dist.get_rank(groups.param_shard_group)
        logger.info(f"Shard group size: {shard_size}, Current rank in shard group: {shard_rank}")

    # Log replication group info 
    if groups.param_repli_group is not None:
        # Get ranks in replication group
        repli_size = dist.get_world_size(groups.param_repli_group)
        repli_rank = dist.get_rank(groups.param_repli_group)
        logger.info(f"Replication group size: {repli_size}, Current rank in replication group: {repli_rank}")
    # Log hierarchical group info and store ranks
    if groups.param_intra_node_group is not None:
        intra_size = dist.get_world_size(groups.param_intra_node_group)
        groups.param_intra_node_rank = dist.get_rank(groups.param_intra_node_group)
        logger.info(f"Intra-node group size: {intra_size}, rank: {groups.param_intra_node_rank}")
    if groups.param_inter_node_shard_group is not None:
        inter_size = dist.get_world_size(groups.param_inter_node_shard_group)
        groups.param_inter_node_rank = dist.get_rank(groups.param_inter_node_shard_group)
        logger.info(f"Inter-node shard group size: {inter_size}, rank: {groups.param_inter_node_rank}")

    return groups


def _generate_penguin_config(world_size, ndev_per_node, shard_size, pp_size=1):
    """Generating the configuration for sharding This shard config generation assume
    that the pipeline stages are partitioned in order, i.e., first ranks
    hold the stage0, etc.

    Args:

        shard_size (int): zero3 data-parallel shard size, FIXME:
        change the name later

        pp_size (int): pipeline parallel size, currently, only work with
        pipeline parallelism + zero

    """
    assert world_size % pp_size == 0
    assert (world_size // pp_size) % shard_size == 0, \
        f"dp group size is not dividable by dp_shard_size, "\
        f" (world_size {world_size}, pp_size {pp_size}, dp_shard_size {shard_size})"

    config = {}
    shard_groups = np.arange(world_size).reshape(-1, shard_size)
    replicate_groups = []
    for i in range(shard_size):
        same_shard_ranks = shard_groups[:, i].tolist()
        n_ranks = len(same_shard_ranks)
        replicate_size = n_ranks // pp_size
        replicate_groups.extend([same_shard_ranks[j:j + replicate_size] for j in range(0, n_ranks, replicate_size)])

    config['replicate_groups'] = replicate_groups
    config['shard_groups'] = shard_groups.tolist()
    config["span_nodes"] = len(shard_groups[0]) // ndev_per_node
    return config


def _sizes_all_same(groups):
    """all groups have same length"""
    all_same = True
    for g in groups:
        if len(g) != len(groups[0]):
            return False
    return all_same
