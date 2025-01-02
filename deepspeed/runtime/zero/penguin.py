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
        
        # is_forward 플래그 초기화
        self.is_forward = True  # 기본값을 True로 설정
        
        assert config_dict_or_path is not None, "Must provide configuration for Penguin Initialization"
        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(config_dict_or_path, mpu)
        
        # config_dict에서 설정 가져오기
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

        # 통신 그룹 초기화
        self.penguin_comm_groups = create_penguin_comm_groups(
            shard_size=self.shard_size,
            hierarchial_params_gather=self.hierarchial_params_gather
        )

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

    def _convert_to_deepspeed_param(self, param):
        """Convert a regular parameter to a DeepSpeed parameter with Penguin features"""
                # 부모 클래스의 변환 메서드 호출
        super()._convert_to_deepspeed_param(param)

        # 통신 그룹 설정
        param.comm = self.penguin_comm_groups
        
        # CPU 버퍼는 나중에 초기화하도록 표시만 해둠
        param.needs_cpu_buffer = True

        # 기존 all_gather_coalesced 메서드 저장
        old_all_gather_coalesced = param.all_gather_coalesced

        def _param_all_gather_coalesced(params, param_buffers=None, **kwargs):
            """Penguin-specific all-gather operation"""
            penguin_comm_groups = params[0].comm
            hierarchical_all_gather = (penguin_comm_groups.param_intra_node_group is not None and 
                                     penguin_comm_groups.param_inter_node_shard_group is not None)
            
            if dist.has_coalescing_manager() and hierarchical_all_gather:
                return self._hierarchical_all_gather_params(params, param_buffers)
            elif dist.has_coalescing_manager():
                return self._flat_all_gather_with_coalescing_manager(params, param_buffers)
            else:
                return old_all_gather_coalesced(params, **kwargs)

        # all_gather_coalesced 메서드 변경
        param.all_gather_coalesced = _param_all_gather_coalesced

    def partition(self, param, **kwargs):
        # ds_tensor가 설정된 후 CPU 버퍼 초기화
        if hasattr(param, 'needs_cpu_buffer') and param.ds_tensor is not None:
            # 현재 rank가 파라미터를 가져야 하는 경우에만 CPU 버퍼 생성
            if self._should_copy_param_to_cpu(param):
                param.penguin_cpu_buffer = torch.empty(
                    param.ds_tensor.ds_numel,  # 실제 필요한 크기만큼만 할당
                    dtype=param.dtype,
                    device='cpu',
                    pin_memory=True
                )
            delattr(param, 'needs_cpu_buffer')  # 초기화 완료 표시
            
        # 현재 rank가 파라미터를 가져야 하는 경우에만 CPU로 이동
        if self._should_copy_param_to_cpu(param):
            with torch.no_grad():
                param.penguin_cpu_buffer.copy_(param.data.view(-1), non_blocking=True)
                param.data = torch.zeros(1, dtype=param.dtype, device=param.device)
                param.ds_status = ZeroParamStatus.NOT_AVAILABLE
                logger.info(f"Parameter {param.ds_id} is now AVAILABLE on CPU.")
        else:
            logger.info(f"Parameter {param.ds_id} is already NOT_AVAILABLE on CPU.")

        
        
        # 부모 클래스의 partition 호출
        super().partition(param, **kwargs)

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
    
    def _should_copy_param_to_gpu(self, param):
        return (param.ds_status == ZeroParamStatus.NOT_AVAILABLE 
                and hasattr(param, 'penguin_cpu_buffer')
                and param.comm.param_intra_node_rank == dist.get_rank(group=param.comm.param_intra_node_group)
                and param.comm.param_shard_rank != dist.get_rank())

    def _should_copy_param_to_cpu(self, param):
        return (param.ds_status == ZeroParamStatus.AVAILABLE
                and param.comm.param_intra_node_rank == dist.get_rank(group=param.comm.param_intra_node_group)
                and param.comm.param_shard_rank != dist.get_rank())

    def _copy_param_from_cpu_to_gpu(self, param):
        assert param.penguin_cpu_buffer is not None
        param.data.view(-1).copy_(param.penguin_cpu_buffer.narrow(0, 0, param.numel()), non_blocking=True)
        param.ds_status = ZeroParamStatus.INFLIGHT

    def _pre_all_gather(self, params, params_buffers=None):
        # 모든 비동기 작업이 완료되었는지 확인
        torch.cuda.synchronize()

        # 이벤트 생성 및 기록
        copy_event = torch.cuda.Event()

        for param in params:
            # 본인에게 해당하는 파라미터인지 확인 
            logger.info(f"param.comm.param_shard_rank: {param.comm.param_shard_rank}, dist.get_rank(group=param.comm.param_inter_node_shard_group): {dist.get_rank(group=param.comm.param_inter_node_shard_group)}")

            if not self.is_forward:
                if self._should_copy_param_to_gpu(param):
                    self._copy_param_from_cpu_to_gpu(param)
                    logger.info(f"Parameter {param.ds_id} copying from CPU buffer to GPU asynchronously.")

        
        # 모든 복사 작업이 완료된 후 이벤트 기록
        torch.cuda.current_stream().record_event(copy_event)

        # 이벤트가 완료되면 상태를 AVAILABLE로 변경
        def _copy_done_callback():
            copy_event.synchronize()  # 이벤트가 완료될 때까지 대기
            for param in params:
                if param.ds_status == ZeroParamStatus.INFLIGHT:
                    param.ds_status = ZeroParamStatus.AVAILABLE
                    logger.info(f"Parameter {param.ds_id} is now AVAILABLE on GPU after async copy.")

        # 비동기 작업 완료 후 콜백 실행
        torch.cuda.current_stream().wait_event(copy_event)
        _copy_done_callback()

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

        # 노드 내 all-gather 준비
        intra_outputs = []
        intra_inputs = []
        for i, p in enumerate(params):
            # 텐서 reshape 수정
            param_chunk_size = p.ds_tensor.ds_numel
            param_chunk = p.ds_tensor.data.view(-1)
            
            # 데이터 복사
            param_tensors[i].narrow(0, local_rank * param_chunk_size, param_chunk_size).copy_(param_chunk)
                
            # 각 노드의 출력과 입력 준비
            for j in range(intra_node_size):
                chunk_start = j * param_chunk_size
                chunk_end = (j + 1) * param_chunk_size
                _out = param_tensors[i].narrow(0, chunk_start, param_chunk_size)
                _in = param_tensors[i].narrow(0, local_rank * param_chunk_size, param_chunk_size)
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


