# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import List

import deepspeed
import torch
from deepspeed import comm as dist
from deepspeed.runtime.zero.utils import is_zero_param
from deepspeed.runtime.zero.penguin_utils import (Penguin_CommGroups, create_penguin_comm_groups, scale_tensors)
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.partition_parameters import (
    Init, 
    AllGatherCoalescedHandle, 
    ZeroParamStatus,
    PartitionedParamStatus
)
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.utils import instrument_w_nvtx, log_dist, logger
from deepspeed.accelerator import get_accelerator
from torch import Tensor
from torch.nn import Parameter
import json
import os
from deepspeed.runtime.zero.partitioned_param_coordinator import PartitionedParameterCoordinator


def has_hierarchical_all_gather_groups(comm_groups: Penguin_CommGroups):
    result = False
    if comm_groups.param_intra_node_group is not None and comm_groups.param_inter_node_shard_group is not None:
        result = True
    return result


class Penguin_AllGatherCoalescedHandle(AllGatherCoalescedHandle):
    """ This handle assumes that no need to
    copy data out from a contiguous tensor
    """

    def __init__(self, allgather_handle, params: List[Parameter], partitions: List[Tensor], world_size: int) -> None:
        super().__init__(allgather_handle, params, partitions, world_size)

    def wait(self) -> None:
        """
        """
        # let the current stream to op
        try:
            # print("HANDLE", self.allgather_handle)
            instrument_w_nvtx(self.allgather_handle.wait)()
        except (ValueError, RuntimeError) as e:
            log_dist(
                f"WARNING: Runtime Error while waiting the collective all-gather, possibly due to the _IllegalWork",
                ranks=[0])
            log_dist(f"Error message: {e}", ranks=[0])

        if self.complete:
            return

        for _, param in enumerate(self.params):
            assert param.ds_status == ZeroParamStatus.INFLIGHT, f"expected param {param.ds_summary()} to be inflight"
            param.ds_status = ZeroParamStatus.AVAILABLE

        self.complete = True


class PenguinParameter(Parameter):
    """DeepSpeed Penguin Parameter class for parameter partitioning"""
    
    def partition(self):
        """Partition the parameter to CPU buffer"""
        if self.ds_status != ZeroParamStatus.NOT_AVAILABLE:
            return
        with torch.no_grad():
            self.penguin_cpu_buffer.copy_(self.data.view(-1))
            self.data = torch.zeros(1, dtype=self.dtype, device=self.device)
            self.ds_status = ZeroParamStatus.NOT_AVAILABLE
            
    def ds_summary(self):
        """Return a summary string of the parameter's DeepSpeed status"""
        return f"Data type: {self.dtype}, Shape: {self.ds_shape}, Status: {self.ds_status}"
        
    def all_gather_coalesced(self, params, **kwargs):
        """Coalesced all-gather operation for parameter groups"""
        mics_comm_groups: Penguin_CommGroups = params[0].comm
        hierarchical_all_gather = has_hierarchical_all_gather_groups(mics_comm_groups)
        
        if dist.has_coalescing_manager() and hierarchical_all_gather:
            return self.ds_process_group._hierarchical_all_gather_params(params)
        elif dist.has_coalescing_manager():
            return self.ds_process_group._flat_all_gather_with_coalescing_manager(params)
        else:
            raise NotImplementedError("Non-coalescing manager all-gather not supported")


class PenguinPartitionedParameterCoordinator(PartitionedParameterCoordinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penguin_comm_groups = None

    def set_penguin_comm_groups(self, comm_groups):
        self.penguin_comm_groups = comm_groups

    def __all_gather_params_(self, param_list, forward=False, quantize=False):
        """Override the parent class method to support penguin communication groups"""
        if len(param_list) == 0:
            return

        if self.penguin_comm_groups is not None:
            # Use penguin communication groups
            param_group = []
            for param in param_list:
                param_group.append(param)
                if len(param_group) == self.max_group_size:
                    handle = param_group[0].all_gather_coalesced(param_group, forward=forward, quantize=quantize)
                    self.handles.append(handle)
                    param_group = []
            if len(param_group) > 0:
                handle = param_group[0].all_gather_coalesced(param_group, forward=forward, quantize=quantize)
                self.handles.append(handle)
        else:
            # Use default implementation
            super().__all_gather_params_(param_list, forward, quantize)


class Penguin_Init(Init):

    def __init__(self,
                 module=None,
                 data_parallel_group=None,
                 sequence_data_parallel_group=None,
                 mem_efficient_linear=True,
                 remote_device=None,
                 pin_memory=False,
                 config_dict_or_path=None,
                 config=None,
                 enabled=True,
                 dtype=None,
                 mpu=None):
        """A context manager to partition the model parameters during the model
        construction with Penguin partition strategy. Model states are partitioned
        to the number of devices specified via ``penguin_shard_size`` field in the
        deepspeed config json file. The context manager also introduces
        hierarchical communication method to reduce the cost of inter-node
        communications, which can be enabled with
        ``penguin_hierarchical_params_gather`` field in deepspeed config.

        Args:
            module (``torch.nn.Module``, optional): If provided, partition the model as
                if it was constructed in the context.
            data_parallel_group (``deepspeed.comm`` process group, optional):
                The group of processes to partition among. Defaults to all processes.
                Synonymous with sequence data parallel group for param partitioning
                across both sequence and data parallel groups.
            mem_efficient_linear (bool, optional): Replace
                torch.nn.functional.linear with an implementation that allows
                DeepSpeed to partition parameters. Defaults to ``True``.
            remote_device (string, optional): The initial device to store model
                weights e.g., ``cpu``, ``nvme``. Passing ``"cpu"`` will create the model in CPU
                memory. The model may still be moved to GPU based on the
                offload settings for training. Defaults to param offload device if a config is
                defined, otherwise GPU.
            pin_memory (bool, optional): Potentially increase performance by
                using pinned memory for model weights. ``remote_device`` must be
                ``"cpu"``. Defaults to pin_memory value in config, otherwise ``False``.
            config_dict_or_path (dict or ``json file``, optional): If provided, provides configuration
                for swapping fp16 params to NVMe.
            config (dict or ``json file``, optional): Deprecated, use config_dict_or_path instead.
            enabled (bool, optional): If ``False``, this context has no
                effect. Defaults to ``True``.
            dtype (``dtype``, optional): Can be used to change the data type of the parameters.
                Supported options are ``torch.half`` and ``torch.float``. Defaults to ``None``
            mpu (``object``, optional): A model parallelism unit object that implements get_{model,data}_parallel_{rank,group,world_size}.

        This context follows the same logic as ``deepspeed.zero.Init()``, but
        with the modification for partition size of each parameter.

        Examples
        --------

        #. Allocate a model and partition it among all processes:

            .. code-block:: python
                # the config_dict_or_path is required to let the context manager know
                # how partition the parameters.
                # The configuration has to include the field ``penguin_shard_size``
                with deepspeed.zero.Penguin_Init(config_dict_or_path=ds_config):
                    model = MyLargeModel()


        #. Allocate a model in pinned CPU memory and partition it among a subgroup of processes:

            .. code-block:: python

                with deepspeed.zero.Penguin_Init(data_parallel_group=mpu.get_data_parallel_group(),
                                              remote_device="cpu",
                                              pin_memory=True
                                              config_dict_or_path=ds_config):
                    model = MyLargeModel()


        #. Partition an already-allocated model in CPU memory:

            .. code-block:: python

                model = deepspeed.zero.Penguin_Init(module=model,
                                                 config_dict_or_path=ds_config)
        """

        assert config_dict_or_path is not None, "Must provide configuration for Penguin Initialization"
        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(config_dict_or_path, mpu)
        
        # config_dict에서 직접 설정 가져오기
        if isinstance(config_dict_or_path, dict):
            config_dict = config_dict_or_path
        else:
            config_dict = _ds_config.config
        
        if 'zero_optimization' not in config_dict:
            raise ValueError("zero_optimization configuration not found in config")
        
        zero_config = config_dict['zero_optimization']
        penguin_config = zero_config.get('penguin', None)
        if penguin_config is None:
            raise ValueError("penguin configuration not found in zero_optimization config")
        
        # penguin 설정에서 값을 가져옴
        self.shard_size = penguin_config.get('shard_size', None)
        if self.shard_size is None:
            raise ValueError("penguin shard_size must be specified in config")
        
        self.hierarchial_params_gather = penguin_config.get('hierarchial_params_gather', False)
        
        if not dist.is_initialized():
            dist.init_distributed()
            assert dist.is_initialized(), "Parameters cannot be scattered without initializing deepspeed.comm"

        if data_parallel_group is None:
            ds_process_group = dist.get_world_group()
        else:
            ds_process_group = data_parallel_group

        if sequence_data_parallel_group is not None:
            logger.warning(
                f"sequence_data_parallel_group' is deprecated and will be removed. Use 'data_parallel_group' instead.")
            if data_parallel_group is not None:
                raise ValueError(
                    "Both 'data_parallel_group' and 'sequence_data_parallel_group' were specified. Please provide only one of these arguments."
                )
            self.ds_process_group = sequence_data_parallel_group

        self.penguin_comm_groups = create_penguin_comm_groups(
            self.shard_size,
            ds_process_group,
            hierarchical_allgather=self.hierarchial_params_gather,
            mpu=mpu)

        # Create custom parameter coordinator
        self.param_coordinator = PenguinPartitionedParameterCoordinator(
            max_group_size=self.max_group_size,
            param_persistence_threshold=self.param_persistence_threshold,
            model_persistence_threshold=self.model_persistence_threshold,
            dp_process_group=self.ds_process_group
        )
        # Set penguin communication groups
        self.param_coordinator.set_penguin_comm_groups(self.penguin_comm_groups)

        super().__init__(module, data_parallel_group, mem_efficient_linear, remote_device, pin_memory,
                         config_dict_or_path, config, enabled, dtype, mpu)

    def _convert_to_deepspeed_param(self, param):
        # 먼저 부모 클래스의 변환을 수행하여 기본 속성들 초기화
        super()._convert_to_deepspeed_param(param)
        param.__class__ = PenguinParameter
        
        # penguin 전용 속성 추가
        param.penguin_cpu_buffer = torch.empty(
            param.ds_numel,
            dtype=param.dtype,
            device='cpu'
        )
        
        # 통신 그룹 설정
        param.comm = self.penguin_comm_groups
        
        # record existing all_gather_coalesced implementation
        old_all_gather_coalesced = param.all_gather_coalesced

        def _param_all_gather_coalesced(params, param_buffers=None, **kwargs):
            mics_comm_groups: Penguin_CommGroups = params[0].comm
            hierarchical_all_gather = has_hierarchical_all_gather_groups(mics_comm_groups)
            if dist.has_coalescing_manager() and hierarchical_all_gather:
                return self._hierarchical_all_gather_params(params, param_buffers)
            elif dist.has_coalescing_manager():
                return self._flat_all_gather_with_coalescing_manager(params, param_buffers)
            else:
                return old_all_gather_coalesced(params, **kwargs)

        # change the all_gather_coalesced method
        param.all_gather_coalesced = _param_all_gather_coalesced

    def _pre_all_gather(self, params, params_buffers=None):
        # fetches from nvme if the partition is not available and in nvme
        self._ensure_availability_of_partitioned_params(params)
        #gspark: fetchs from cpu
        #self._get_partitioned_params_from_cpu(params)

        for param in params:
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(param.ds_summary())
            param.ds_status = ZeroParamStatus.INFLIGHT

        # ensure that each rank has params in same order. the allgather
        # is done by flattening the parameter list into a single tensor that
        # can be allgathered in a single call - this means that if each rank
        # gives a list of the same parameters in a different order we will
        # silently get incorrect parameter values, and have very difficult
        # to debug correctness issues.
        params = sorted(params, key=lambda p: p.ds_id)
        return params, params_buffers

    def _flat_all_gather_with_coalescing_manager(self, params, params_buffers=None):
        """Flat all-gather with coalescing manager"""
        # 파라미터 준비
        params, params_buffers = self._pre_all_gather(params, params_buffers)
        
        # 출력 텐서 준비
        output_tensors = []
        input_tensors = []
        for i, p in enumerate(params):
            t_size = p.ds_tensor.ds_numel * self.shard_size
            if params_buffers is not None and params_buffers[i] is not None:
                flat_out = params_buffers[i]
            else:
                flat_out = torch.empty(t_size, dtype=p.dtype, device=self.local_device, requires_grad=False).view(-1)
            output_tensors.append(flat_out)
            input_tensors.append(p.ds_tensor.data.view(-1))

        # all-gather 수행
        all_gather_handle = dist.all_gather_coalesced(
            output_tensors,
            input_tensors,
            group=self.penguin_comm_groups.param_shard_group,
            async_op=True
        )

        # 결과 텐서 업데이트
        for idx, param in enumerate(params):
            param.data = output_tensors[idx].narrow(0, 0, param.ds_numel).view(param.ds_shape).data

        return Penguin_AllGatherCoalescedHandle(
            allgather_handle=all_gather_handle,
            params=params,
            partitions=[],
            world_size=self.shard_size
        )

    def _hierarchical_all_gather_params(self, params, params_buffers=None):
        """Hierarchical all-gather implementation"""
        params, params_buffers = self._pre_all_gather(params, params_buffers)

        penguin_comm_groups: Penguin_CommGroups = params[0].comm
        local_rank = dist.get_rank(group=penguin_comm_groups.param_intra_node_group)
        inter_node_comm_group = penguin_comm_groups.param_inter_node_shard_group
        intra_node_comm_group = penguin_comm_groups.param_intra_node_group
        param_shard_size = penguin_comm_groups.param_shard_size

        inter_node_size = dist.get_world_size(group=inter_node_comm_group)
        intra_node_size = dist.get_world_size(group=intra_node_comm_group)
        
        # 파라미터 텐서 준비
        param_tensors = []
        for i, p in enumerate(params):
            param_size = p.ds_tensor.ds_numel * param_shard_size
            if params_buffers is not None and params_buffers[i] is not None:
                param_tensor = params_buffers[i]
            else:
                param_tensor = torch.empty(param_size, 
                                         dtype=p.dtype, 
                                         device=self.local_device,
                                         requires_grad=False).view(-1)
            param_tensors.append(param_tensor)

        # 노드 간 all-gather
        inter_outputs = []
        inter_inputs = []
        for i, p in enumerate(params):
            inter_size = p.ds_tensor.ds_numel * inter_node_size
            _out = param_tensors[i].narrow(0, local_rank * inter_size, inter_size)
            inter_outputs.append(_out)
            inter_inputs.append(p.ds_tensor.data.view(-1).to(self.local_device))

        # 동기 all-gather 수행
        with torch.cuda.stream(torch.cuda.Stream()):
            for out, inp in zip(inter_outputs, inter_inputs):
                dist.all_gather_into_tensor(
                    out,
                    inp,
                    group=inter_node_comm_group
                )

        # 노드 내 all-gather 준비
        intra_outputs = []
        intra_inputs = []
        for i, p in enumerate(params):
            param_chunk = param_tensors[i].view(
                (inter_node_size, intra_node_size, p.ds_tensor.ds_numel)
            ).narrow(1, local_rank, 1)
            
            # 데이터 복사
            with torch.no_grad():
                param_chunk.copy_(inter_outputs[i].view(param_chunk.size()))
                
            output_chunks = torch.chunk(param_tensors[i], inter_node_size)
            for j, _out in enumerate(output_chunks):
                intra_chunk_size = intra_node_size * p.ds_tensor.ds_numel
                local_offset = local_rank * p.ds_tensor.ds_numel
                _in = param_tensors[i].narrow(0, j * intra_chunk_size + local_offset, p.ds_tensor.ds_numel)
                intra_outputs.append(_out)
                intra_inputs.append(_in)

        # 노드 내 all-gather (비동기)
        all_gather_handle = dist.all_gather_coalesced(
            intra_outputs,
            intra_inputs,
            group=intra_node_comm_group,
            async_op=True
        )

        # 결과 업데이트
        for i, param in enumerate(params):
            param.data = param_tensors[i].narrow(0, 0, param.ds_numel).view(param.ds_shape).data

        return Penguin_AllGatherCoalescedHandle(
            allgather_handle=all_gather_handle,
            params=params,
            partitions=[],
            world_size=param_shard_size
        )

    def get_partition_dp_group(self, param):
        return param.comm.param_shard_group

    def get_partition_rank(self):
        return self.penguin_comm_groups.param_shard_rank

    @property
    def num_partitions(self):
        return self.penguin_comm_groups.param_shard_size


class Penguin_Offload(DeepSpeedZeRoOffload):
    """ Wrapper to change the behavior for parameter sharding
    """

    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        """ overload the parent class function for convert the parameters

        """
        log_dist(f'Convert to zero parameters from Penguin Offload manager', ranks=[0])
        non_zero_params = [p for p in module.parameters() if not is_zero_param(p)]
        if non_zero_params:
            zero_params = [p for p in module.parameters() if is_zero_param(p)]
            if zero_params:
                zero_params[0].convert_to_zero_parameters(param_list=non_zero_params)
            else:
                group = None
                if mpu:
                    group = mpu.get_data_parallel_group()

                Penguin_Init(module=module,
                          data_parallel_group=group,
                          dtype=self.dtype,
                          config_dict_or_path=ds_config,
                          remote_device=self.offload_device,
                          pin_memory=self.offload_param_pin_memory,
                          mpu=mpu)


class Penguin_Optimizer(DeepSpeedZeroOptimizer_Stage3):
    """
    Penguin Optimizer
    """

    def __init__(self,
                 module,
                 init_optimizer,
                 timers,
                 ds_config,
                 static_loss_scale=1,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 contiguous_gradients=True,
                 reduce_bucket_size=500000000,
                 prefetch_bucket_size=50000000,
                 max_reuse_distance=1000000000,
                 max_live_parameters=1000000000,
                 param_persistence_threshold=100000,
                 model_persistence_threshold=sys.maxsize,
                 dp_process_group=None,
                 reduce_scatter=True,
                 overlap_comm=False,
                 offload_optimizer_config=None,
                 offload_param_config=None,
                 sub_group_size=1000000000000,
                 offload_ratio=0.0,
                 mpu=None,
                 clip_grad=0,
                 gradient_accumulation_dtype=torch.float16,
                 communication_data_type=torch.float16,
                 postscale_gradients=True,
                 gradient_predivide_factor=1,
                 gradient_accumulation_steps=1,
                 elastic_checkpoint=False,
                 aio_config=None):

        log_dist("Init Penguin optimizer", ranks=[0])
        super().__init__(module, init_optimizer, timers, ds_config, static_loss_scale, dynamic_loss_scale,
                         dynamic_loss_args, verbose, contiguous_gradients, reduce_bucket_size, prefetch_bucket_size,
                         max_reuse_distance, max_live_parameters, param_persistence_threshold,
                         model_persistence_threshold, dp_process_group, reduce_scatter, overlap_comm,
                         offload_optimizer_config, offload_param_config, sub_group_size, offload_ratio, mpu, clip_grad,
                         gradient_accumulation_dtype, communication_data_type, postscale_gradients,
                         gradient_predivide_factor, gradient_accumulation_steps, elastic_checkpoint, aio_config)
        # Get first parameter's communication groups
        first_param = next(self.module.parameters())
        self.penguin_comm_groups = first_param.comm

    def initialize_ds_offload(
        self,
        *args,
        **kwargs,
    ):
        return Penguin_Offload(*args, **kwargs)

    def partition_grads(self, params_to_release: List[Parameter], grad_partitions: List[Tensor]) -> None:
        """Override partition_grads to use penguin communication groups"""
        grad_buffers = super().partition_grads(params_to_release, grad_partitions)
        # Perform all-reduce among replication groups
        self.allreduce_penguin_shard_grads(params_to_release, grad_buffers)

    @instrument_w_nvtx
    def allreduce_penguin_shard_grads(self, params, partitioned_grads_buffers: List[Tensor]):
        """All-reduce gradients using penguin communication groups"""
        if not self.is_gradient_accumulation_boundary or len(partitioned_grads_buffers) == 0:
            return

        penguin_comm_groups: Penguin_CommGroups = params[0].comm
        param_repli_group = penguin_comm_groups.param_repli_group
        param_repli_size = penguin_comm_groups.param_repli_size

        if param_repli_size is None or param_repli_size <= 1:
            return
        if not get_accelerator().on_accelerator(partitioned_grads_buffers[0]):
            raise RuntimeError("Local sharding has no support for CPU offloading")

        if dist.has_all_reduce_coalesced():
            scale_tensors(partitioned_grads_buffers, param_repli_size)
            dist.all_reduce_coalesced(tensors=partitioned_grads_buffers, group=param_repli_group)
        else:
            # Manually coalescing all-reduce
            aggregated_buffer: Tensor = torch.cat(partitioned_grads_buffers)
            aggregated_buffer.div_(param_repli_size)
            dist.all_reduce(aggregated_buffer, group=param_repli_group)
            offset = 0
            for grad_buff in partitioned_grads_buffers:
                grad_buff.view(-1).copy_(aggregated_buffer.narrow(0, offset, grad_buff.numel()))
                offset += grad_buff.numel()

    def load_state_dict(self,
                        state_dict_list,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False,
                        checkpoint_folder=None,
                        load_serial=None):
        r""" Loading the ZeRO-3/Penguin partitioned checkpoints
        Because the self.dp_process_group is replaced with the communicator for
        partition group we can call the load_state_dict logic from ZeRO-3.
        """
        super().load_state_dict(state_dict_list, load_optimizer_states, load_from_fp32_weights, checkpoint_folder)
