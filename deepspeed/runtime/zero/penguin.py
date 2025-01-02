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
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
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
from .penguin_utils import create_penguin_comm_groups


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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 통신 그룹 초기화
        if not hasattr(self, 'ds_process_group') or self.ds_process_group is None:
            self.ds_process_group = dist.new_group(ranks=list(range(dist.get_world_size())))
        if self.ds_process_group is None:
            self.ds_process_group = dist.group.WORLD

        # penguin_cpu_buffer 초기화
        self.partition()
        #self._initialize_cpu_buffer()

    def _initialize_cpu_buffer(self):
        """Initialize or resize the CPU buffer for inter-mapped GPU parameters."""
        if self._is_mapped_to_current_rank():
            mapped_size = self.ds_numel if hasattr(self, 'ds_numel') else self.data.numel()
            if not hasattr(self, 'penguin_cpu_buffer') or self.penguin_cpu_buffer.numel() != mapped_size:
                self.penguin_cpu_buffer = torch.empty(
                    mapped_size,
                    dtype=self.dtype,
                    device='cpu',
                    pin_memory=True  # Use pinned memory for better performance
                )
                logger.info(f"Initialized CPU buffer for parameter {self.ds_id} with size {mapped_size}.")

    def _is_mapped_to_current_rank(self) -> bool:
        """Check if the parameter is mapped to the current rank."""
        current_rank = dist.get_rank()
        if hasattr(self, 'comm') and self.comm.param_shard_group is not None:
            param_rank = dist.get_rank(group=self.comm.param_shard_group)
            return current_rank == param_rank
        return False

    def partition(self):
        """Partition the parameter to CPU buffer"""
        if self.ds_status != ZeroParamStatus.NOT_AVAILABLE:
            return
        with torch.no_grad():
            if self._is_mapped_to_current_rank():
                self._initialize_cpu_buffer()
                self.penguin_cpu_buffer.copy_(self.data.view(-1), non_blocking=True)
            self.data = torch.zeros(1, dtype=self.dtype, device=self.device)
            self.ds_status = ZeroParamStatus.NOT_AVAILABLE
            
    def ds_summary(self):
        """Return a summary string of the parameter's DeepSpeed status"""
        return f"Data type: {self.dtype}, Shape: {self.ds_shape}, Status: {self.ds_status}"


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
        """Penguin initialization context"""
        
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

        # Init 클래스의 속성들 초기화
        self.max_group_size = zero_config.get('max_group_size', 1000000000000)
        self.param_persistence_threshold = zero_config.get('param_persistence_threshold', 100000)
        self.model_persistence_threshold = zero_config.get('model_persistence_threshold', sys.maxsize)
        self.dp_process_group = data_parallel_group

        # 통신 그룹 초기화 - penguin_utils의 함수 사용
        self.penguin_comm_groups = create_penguin_comm_groups(
            shard_size=self.shard_size,
            hierarchial_params_gather=self.hierarchial_params_gather
        )

        # 부모 클래스 초기화
        super().__init__(module=module,
                        data_parallel_group=data_parallel_group,
                        sequence_data_parallel_group=sequence_data_parallel_group,
                        mem_efficient_linear=mem_efficient_linear,
                        remote_device=remote_device,
                        pin_memory=pin_memory,
                        config_dict_or_path=config_dict_or_path,
                        config=config,
                        enabled=enabled,
                        dtype=dtype,
                        mpu=mpu)

        self.is_forward = False

    def register_hooks(self, module):
        # Forward hook
        module.register_forward_pre_hook(self._start_forward)
        module.register_forward_hook(self._end_forward)

        # Backward hook
        module.register_backward_hook(self._start_backward)
    

    def _start_forward(self, module, input):
        self.is_forward = True

    def _end_forward(self, module, input, output):
        self.is_forward = False

    def _start_backward(self, module, grad_input, grad_output):
        self.is_forward = False

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
        param.ds_process_group = self.dp_process_group

        # 기존 all_gather_coalesced 메서드 저장
        old_all_gather_coalesced = param.all_gather_coalesced

        def _param_all_gather_coalesced(params, param_buffers=None, **kwargs):
            """Penguin-specific all-gather operation"""
            penguin_comm_groups: Penguin_CommGroups = params[0].comm
            hierarchical_all_gather = has_hierarchical_all_gather_groups(penguin_comm_groups)
            if dist.has_coalescing_manager() and hierarchical_all_gather:
                return self._hierarchical_all_gather_params(params, param_buffers)
            elif dist.has_coalescing_manager():
                return self._flat_all_gather_with_coalescing_manager(params, param_buffers)
            else:
                return old_all_gather_coalesced(params, **kwargs)

        # all_gather_coalesced 메서드 변경
        param.all_gather_coalesced = _param_all_gather_coalesced

    def _pre_all_gather(self, params, params_buffers=None):
        # 모든 비동기 작업이 완료되었는지 확인
        torch.cuda.synchronize()

        for param in params:
            # 본인에게 해당하는 파라미터인지 확인 
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and hasattr(param, 'penguin_cpu_buffer') and param.comm.param_shard_rank != dist.get_rank(group=param.comm.param_inter_node_shard_group):
                if not self.is_forward:
                    # CPU 버퍼에서 데이터를 가져옵니다.
                    if hasattr(param, 'penguin_cpu_buffer'):
                        param.data.view(-1).copy_(param.penguin_cpu_buffer.narrow(0, 0, param.numel()), non_blocking=True)
                        logger.info(f"Parameter {param.ds_id} copied from CPU buffer to GPU.")
                    else:
                        raise RuntimeError(f"Parameter {param.ds_id} is not available and has no CPU buffer.")
                    
                    param.ds_status = ZeroParamStatus.INFLIGHT

        # ensure that each rank has params in same order. the allgather
        # is done by flattening the parameter list into a single tensor that
        # can be allgathered in a single call - this means that if each rank
        # gives a list of the same parameters in a different order we will
        # silently get incorrect parameter values, and have very difficult
        # to debug correctness issues.
        params = sorted(params, key=lambda p: p.ds_id)
        logger.info(f"All-gather operation started for parameters: {[p.ds_id for p in params]}")
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
            param.ds_status = ZeroParamStatus.AVAILABLE  # 상태를 AVAILABLE로 변경
            logger.info(f"Parameter {param.ds_id} is now AVAILABLE on GPU.")

        # all-gather 핸들이 완료된 후 release 호출
        all_gather_handle.wait()
        if self.is_forward:
            self._release_unused_parameters(params)

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
                 **kwargs):

        log_dist("Init Penguin optimizer", ranks=[0])
        super().__init__(module=module, 
                        init_optimizer=init_optimizer,
                        timers=timers,
                        ds_config=ds_config,
                        static_loss_scale=static_loss_scale,
                        dynamic_loss_scale=dynamic_loss_scale,
                        dynamic_loss_args=dynamic_loss_args,
                        **kwargs)

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

    def _release_unused_parameters(self, params):
        """Release unused parameters to free up memory."""
        for param in params:
            if param.ds_status == ZeroParamStatus.INFLIGHT:
                # CPU로 파라미터를 옮깁니다.
                if hasattr(param, 'penguin_cpu_buffer'):
                    param.penguin_cpu_buffer.copy_(param.data.view(-1), non_blocking=True)
                    logger.info(f"Parameter {param.ds_id} moved to CPU memory.")
                else:
                    raise RuntimeError(f"Parameter {param.ds_id} does not have a CPU buffer.")

                # GPU 메모리를 해제합니다.
                param.data = torch.zeros(1, dtype=param.dtype, device=param.device)
                param.ds_status = ZeroParamStatus.NOT_AVAILABLE
                logger.info(f"Parameter {param.ds_id} released from GPU memory.")


def convert_to_penguin_param(param: Parameter, comm: Penguin_CommGroups) -> PenguinParameter:
    """Convert a parameter to PenguinParameter"""
    # Create PenguinParameter
    param.__class__ = PenguinParameter
    param.comm = comm
    
    # CPU buffer 초기화 - 파라미터의 실제 크기 사용
    if hasattr(param, 'ds_tensor'):
        buffer_size = param.ds_tensor.numel()
    else:
        buffer_size = param.numel()
        
    param.penguin_cpu_buffer = torch.zeros(buffer_size,
                                         dtype=param.dtype,
                                         device='cpu')
    
    # 초기화 시점에 CPU로 이동해야 하는지 확인
    inter_rank = dist.get_rank(group=comm.param_inter_node_shard_group)
    if comm.param_shard_rank != inter_rank:
        # GPU에서 CPU로 비동기 복사
        param.penguin_cpu_buffer.copy_(param.ds_tensor.data.view(-1).to(param.penguin_cpu_buffer.device), non_blocking=True)
        logger.info(f"Parameter {param.ds_id} copied from GPU to CPU buffer.")
        param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE
        param.ds_tensor.final_location = OffloadDeviceEnum.cpu
    
    return param
