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
from deepspeed.runtime.zero.mics_utils import MiCS_CommGroups
from deepspeed.runtime.zero.mics import create_mics_comm_groups


@dataclass
class Penguin_CommGroups(MiCS_CommGroups):
    pass

def create_penguin_comm_groups(
    shard_size,
    dp_group,
    hierarchical_allgather=False,
    mpu=None,
):
    return create_mics_comm_groups(shard_size, dp_group, hierarchical_allgather, mpu)
