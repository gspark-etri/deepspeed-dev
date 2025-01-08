import sys
from typing import List

import deepspeed
import torch
from deepspeed import comm as dist
from deepspeed.runtime.zero.mics_utils import scale_tensors
from deepspeed.runtime.zero.utils import is_zero_param
from deepspeed.runtime.zero.penguin_utils import (Penguin_CommGroups, create_penguin_comm_groups)
from deepspeed.runtime.zero.mics import MiCS_AllGatherCoalescedHandle, MiCS_Optimizer, MiCS_Offload, MiCS_Init
from deepspeed.runtime.zero.partition_parameters import Init, AllGatherCoalescedHandle, ZeroParamStatus
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum, DeepSpeedZeroOffloadParamConfig
from deepspeed.utils import instrument_w_nvtx, log_dist, logger
from deepspeed.accelerator import get_accelerator
from torch import Tensor
from torch.nn import Parameter
from deepspeed.runtime.zero.mics import MiCS_Init
from deepspeed.runtime.zero.partition_parameters import Init

def has_hierarchical_all_gather_groups(comm_groups: Penguin_CommGroups):
    result = False
    if comm_groups.param_intra_node_group is not None and comm_groups.param_inter_node_shard_group is not None:
        result = True
    return result

class Penguin_AllGatherCoalescedHandle(MiCS_AllGatherCoalescedHandle):
    """ This handle assumes that no need to
    copy data out from a contiguous tensor
    """

    def __init__(self, allgather_handle, params: List[Parameter], partitions: List[Tensor], world_size: int) -> None:
        super().__init__(allgather_handle, params, partitions, world_size)

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

        assert config_dict_or_path is not None, "Must provide configuration for MiCS Initialization"
        _ds_config = deepspeed.runtime.config.DeepSpeedConfig(config_dict_or_path, mpu)
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
            shard_size=_ds_config.penguin_shard_size,
            dp_group=ds_process_group,
            hierarchical_allgather=_ds_config.penguin_hierarchial_params_gather,
            mpu=mpu)
        
        self.is_forward = True

        super().__init__(module, data_parallel_group, mem_efficient_linear, remote_device, pin_memory,
                         config_dict_or_path, config, enabled, dtype, mpu)
    
    def partition(self, param, **kwargs):
        if self.is_forward:
            #TODO: copy param to cpu if needed
            pass 

        super().partition(param, **kwargs)
    
    def register_hooks(self, module):
        # Forward hook
        module.register_forward_pre_hook(self._start_forward)
        module.register_forward_hook(self._end_forward)

        # Backward hook
        #module.register_backward_hook(self._start_backward)
    
    def _start_forward(self, module, input):
        self.is_forward = True

    def _end_forward(self, module, input, output):
        self.is_forward = False


    def _convert_to_deepspeed_param(self, param):
        super()._convert_to_deepspeed_param(param)
        # attach communication groups to every param
        param.comm = self.penguin_comm_groups

        # record existing all_gather_coalesced implementation
        # so that we can fallback later
        old_all_gather_coalesced = param.all_gather_coalesced

        def _param_all_gather_coalesced(params, param_buffers=None, **kwargs):
            """"""
            penguin_comm_groups: Penguin_CommGroups = params[0].comm
            hierarchical_all_gather = has_hierarchical_all_gather_groups(penguin_comm_groups)
            if dist.has_coalescing_manager() and hierarchical_all_gather:
                return self._hierarchical_all_gather_params(params, param_buffers)
            elif dist.has_coalescing_manager():
                return self._flat_all_gather_with_coalescing_manager(params, param_buffers)
            else:
                return old_all_gather_coalesced(params, **kwargs)
        
        param.all_gather_coalesced = _param_all_gather_coalesced

    def _flat_all_gather_with_coalescing_manager(self, params, params_buffers=None):
        params, params_buffers = self._pre_all_gather(params, params_buffers)

        penguin_comm_groups: Penguin_CommGroups = params[0].comm
        param_shard_size = penguin_comm_groups.param_shard_size

        #todo: forward aware allgather and tranfer to cpu if needed

        output_tensors = []
        input_tensors = []
        for i, p in enumerate(params):
            t_size = p.ds_tensor.ds_numel * param_shard_size
            if params_buffers is not None and params_buffers[i] is not None:
                assert params_buffers[i].numel(
                ) == t_size, f'params_to_gather_buffers[{i}] size {params_buffers[i].numel()} does not match with t_size {t_size}'
                flat_out = params_buffers[i]
            else:
                flat_out = torch.empty(t_size, dtype=p.dtype, device=self.local_device, requires_grad=False).view(-1)
            output_tensors.append(flat_out)
            _flat_input = p.ds_tensor.data.view(-1)
            input_tensors.append(_flat_input)

        all_gather_handle = dist.all_gather_coalesced(output_tensors,
                                                      input_tensors,
                                                      group=penguin_comm_groups.param_shard_group,
                                                      async_op=True)

        for idx, param in enumerate(params):
            param.data = output_tensors[idx].narrow(0, 0, param.ds_numel).view(param.ds_shape).data

        return Penguin_AllGatherCoalescedHandle(allgather_handle=all_gather_handle,
                                             params=params,
                                             partitions=[],
                                             world_size=param_shard_size)



    def _hierarchical_all_gather_params(self, params, params_buffers=None):
        params, params_buffers = self._pre_all_gather(params, params_buffers)

        penguin_comm_groups: Penguin_CommGroups = params[0].comm
        local_rank = dist.get_rank(group=penguin_comm_groups.param_intra_node_group)
        inter_node_comm_group = penguin_comm_groups.param_inter_node_shard_group
        intra_node_comm_group = penguin_comm_groups.param_intra_node_group
        param_shard_size = penguin_comm_groups.param_shard_size

        inter_node_size = dist.get_world_size(group=inter_node_comm_group)
        intra_node_size = dist.get_world_size(group=intra_node_comm_group)

        param_tensors = []
        for i, p in enumerate(params):
            param_size = p.ds_tensor.ds_numel * param_shard_size
            if params_buffers is not None and params_buffers[i] is not None:
                assert params_buffers[i].numel(
                ) == param_size, f'param_buffers[{i}] size {params_buffers[i].numel()} does not match with param_size {param_size}'
                param_tensor = params_buffers[i]
            else:
                param_tensor = torch.empty(param_size, dtype=p.dtype, device=self.local_device,
                                           requires_grad=False).view(-1)
            param_tensors.append(param_tensor)

        # inter node all-gather
        inter_outputs = []
        inter_inputs = []
        for i, p in enumerate(params):
            inter_size = p.ds_tensor.ds_numel * inter_node_size
            _out = param_tensors[i].narrow(0, local_rank * inter_size, inter_size)
            inter_outputs.append(_out)
            inter_inputs.append(p.ds_tensor.data.view(-1).to(self.local_device))
        # sync enqueue
        dist.all_gather_coalesced(inter_outputs, inter_inputs, group=inter_node_comm_group, async_op=False)

        # intra node all-gather
        intra_outputs = []
        intra_inputs = []
        for i, p in enumerate(params):
            # partition param into multiple chunks for allgather
            # because inter-node all-gather outputs are in a continues memory
            # while in param memory, those inter-node data are placed in different
            # location.
            # each chunk is an intra-node output
            param_chunk = param_tensors[i].view(
                (inter_node_size, intra_node_size, p.ds_tensor.ds_numel)).narrow(1, local_rank, 1)
            param_chunk.copy_(inter_outputs[i].detach().clone().view(param_chunk.size()))
            output_chunks = torch.chunk(param_tensors[i], inter_node_size)
            for j, _out in enumerate(output_chunks):
                intra_chunk_size = intra_node_size * p.ds_tensor.ds_numel
                local_offset = local_rank * p.ds_tensor.ds_numel
                _in = param_tensors[i].narrow(0, j * intra_chunk_size + local_offset, p.ds_tensor.ds_numel)
                intra_outputs.append(_out)
                intra_inputs.append(_in)

        all_gather_handle = dist.all_gather_coalesced(intra_outputs,
                                                      intra_inputs,
                                                      group=intra_node_comm_group,
                                                      async_op=True)
        for i, param in enumerate(params):
            param.data = param_tensors[i].narrow(0, 0, param.ds_numel).view(param.ds_shape).data

        return Penguin_AllGatherCoalescedHandle(
            allgather_handle=all_gather_handle,
            params=params,
            partitions=[],
            world_size=param_shard_size,
        )
 

    
    def _pre_all_gather(self, params, params_buffers=None):
        # fetches from nvme if the partition is not available and in nvme
        self._ensure_availability_of_partitioned_params(params)

        # 파라미터 상태 체크 및 업데이트
        for param in params:
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(param.ds_summary())
            param.ds_status = ZeroParamStatus.INFLIGHT

        # 파라미터 순서 보장을 위한 정렬
        # 이는 매우 중요! 각 rank가 동일한 순서로 파라미터를 처리해야 함
        params = sorted(params, key=lambda p: p.ds_id)
            
        return params, params_buffers

class Penguin_Offload(MiCS_Offload):
    def _convert_to_zero_parameters(self, ds_config, module, mpu):
        log_dist(f'Convert to zero parameters from MiCS Offload manager', ranks=[0])
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


class Penguin_Optimizer(MiCS_Optimizer):
    def __init__(self, module, init_optimizer, timers, ds_config, static_loss_scale=1, dynamic_loss_scale=False, dynamic_loss_args=None, verbose=True, contiguous_gradients=True, reduce_bucket_size=500000000, prefetch_bucket_size=5000000, max_reuse_distance=1000000000, max_live_parameters=1000000000, param_persistence_threshold=100000, model_persistence_threshold=sys.maxsize, dp_process_group=None, reduce_scatter=True, overlap_comm=False, offload_optimizer_config=None, offload_param_config=None, sub_group_size=1000000000000, offload_ratio=0.0, mpu=None, clip_grad=0, gradient_accumulation_dtype=torch.float16, communication_data_type=torch.float16, postscale_gradients=True, gradient_predivide_factor=1, gradient_accumulation_steps=1, elastic_checkpoint=False, aio_config=None):
        
        # DeepSpeedZeroOffloadParamConfig 구조체 사용
        offload_param_config = DeepSpeedZeroOffloadParamConfig(
            device=OffloadDeviceEnum.cpu,
            pin_memory=True,
            max_in_cpu=sys.maxsize
        )

        super().__init__(module, init_optimizer, timers, ds_config, static_loss_scale, dynamic_loss_scale, dynamic_loss_args, verbose, contiguous_gradients, reduce_bucket_size, prefetch_bucket_size, max_reuse_distance, max_live_parameters, param_persistence_threshold, model_persistence_threshold, dp_process_group, reduce_scatter, overlap_comm, offload_optimizer_config, offload_param_config, sub_group_size, offload_ratio, mpu, clip_grad, gradient_accumulation_dtype, communication_data_type, postscale_gradients, gradient_predivide_factor, gradient_accumulation_steps, elastic_checkpoint, aio_config)

    def _create_fp16_partitions_with_defragmentation(self, fp16_param_groups):
        #TODO: penguin 적용, 필요한 parameter만 partitions
        super()._create_fp16_partitions_with_defragmentation(fp16_param_groups)
    
    #TODO: allreduce_mics_shard_grads 수정 with penguin
    @instrument_w_nvtx
    def allreduce_mics_shard_grads(self, params, partitioned_grads_buffers: List[Tensor]) -> None:
        """
        """
        # TODO: improve the condition check
        if not self.is_gradient_accumulation_boundary or \
            len(partitioned_grads_buffers) == 0:
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
            # manually coalescing all-reduce
            aggregated_buffer: Tensor = torch.cat(partitioned_grads_buffers)
            aggregated_buffer.div_(param_repli_size)
            offset = 0
            for grad_buff in partitioned_grads_buffers:
                grad_buff.view(-1).copy_(aggregated_buffer.narrow(0, offset, grad_buff.numel()))
                offset += grad_buff.numel()
