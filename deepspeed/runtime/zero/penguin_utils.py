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

    def _move_other_inter_params_to_cpu(self, params):
        """다른 inter node의 파라미터들을 CPU로 이동시킵니다."""
        for param in params:
            if not hasattr(param, 'comm'):
                continue
            
            inter_rank = dist.get_rank(group=self.param_inter_node_shard_group)
            
            # 현재 노드의 파라미터가 아닌 경우에만 CPU로 이동
            if param.ds_tensor.ds_param_rank != inter_rank:
                # 필요한 속성들이 모두 있는지 확인
                assert hasattr(param, 'penguin_cpu_buffer'), f"Parameter {param.ds_id} missing penguin_cpu_buffer"
                
                # GPU에서 CPU로 비동기 복사
                param.penguin_cpu_buffer.copy_(param.ds_tensor.data.view(-1), non_blocking=True)
                param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE
                param.ds_tensor.final_location = OffloadDeviceEnum.cpu

    def _get_inter_params_from_cpu(self, params):
        """CPU에 저장된 다른 inter node의 파라미터들을 다시 GPU로 가져옵니다."""
        for param in params:
            if not hasattr(param, 'comm'):
                continue
            
            inter_rank = dist.get_rank(group=self.param_inter_node_shard_group)
            
            # 현재 노드의 파라미터가 아닌 경우에만 GPU로 복원
            if param.ds_tensor.ds_param_rank != inter_rank:
                # 필요한 속성들이 모두 있는지 확인
                assert hasattr(param, 'penguin_cpu_buffer'), f"Parameter {param.ds_id} missing penguin_cpu_buffer"
                
                if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                    # CPU에서 GPU로 비동기 복사
                    param.ds_tensor.data.view(-1).copy_(param.penguin_cpu_buffer, non_blocking=True)
                    param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                    param.ds_tensor.final_location = None


def create_penguin_comm_groups(shard_size, hierarchial_params_gather=False):
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
    
    if shard_size > world_size:
        raise ValueError(f"shard_size ({shard_size}) cannot be larger than world_size ({world_size})")
    if world_size % shard_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by shard_size ({shard_size})")

    groups = Penguin_CommGroups()

    # Create shard groups
    for i in range(0, world_size, shard_size):
        ranks = list(range(i, min(i + shard_size, world_size)))
        group = dist.new_group(ranks)
        if global_rank in ranks:
            groups.param_shard_group = group
            groups.param_shard_size = len(ranks)
            groups.param_shard_rank = dist.get_rank(group)

    # Create replication groups
    num_replicas = world_size // shard_size
    if num_replicas > 1:
        for i in range(shard_size):
            ranks = list(range(i, world_size, shard_size))
            group = dist.new_group(ranks)
            if global_rank in ranks:
                groups.param_repli_group = group
                groups.param_repli_size = len(ranks)
                groups.param_repli_rank = dist.get_rank(group)
    else:
        groups.param_repli_group = None
        groups.param_repli_size = 1
        groups.param_repli_rank = 0

    # Create hierarchical groups if needed
    if hierarchial_params_gather:
        # Create intra-node groups
        local_world_size = ndevices_per_node
        for i in range(0, world_size, local_world_size):
            ranks = list(range(i, min(i + local_world_size, world_size)))
            group = dist.new_group(ranks)
            if global_rank in ranks:
                groups.param_intra_node_group = group

        # Create inter-node groups
        for i in range(local_world_size):
            ranks = list(range(i, world_size, local_world_size))
            group = dist.new_group(ranks)
            if global_rank in ranks:
                groups.param_inter_node_shard_group = group

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
