# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
import collections
from collections import UserDict
from typing import Deque, Set

from deepspeed import comm as dist
from deepspeed.utils import z3_leaf_module
from deepspeed.utils.logging import logger
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.partitioned_param_profiler import PartitionedParameterProfiler
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus
from deepspeed.utils.debug import debug_module2name_id, debug_param2name_id
from deepspeed.accelerator import get_accelerator
import deepspeed.runtime.compiler as compiler
from deepspeed.runtime.compiler import is_compiling

import logging

ENABLE_PROFILER = False


def debug_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        logger.debug(message)


@instrument_w_nvtx
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


@compiler.disable
def iter_params(module: Module, recurse=False) -> Iterable[Parameter]:
    return map(lambda pair: pair[1], get_all_parameters(module, recurse))


class ZeRoTraceMode(Enum):
    # Record trace of the network during a single forward+backward (for training) or forward (for inference)
    RECORD = 1
    # Use recorded network trace to optimize current forward+backward or forward
    COMPLETE = 2
    # Recorded trace does not match current forward+backward or forward pass.
    INVALID = 3


class InflightParamRegistry(UserDict):
    """registry for parameters in flight"""

    def __setitem__(self, param: Parameter, handle: AllGatherCoalescedHandle) -> None:
        if param in self.data:
            raise RuntimeError(f"{param.ds_summary()} already in registry")
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"attempted to add non-inflight parameter to registry {param.ds_summary()}")
        self.data[param] = handle


class PartitionedParameterCoordinator:
    FORWARD_FETCH_SUBMIT = 'forward_fetch_submit'
    FORWARD_FETCH_WAIT = 'forward_fetch_wait'
    FORWARD_PREFETCH_SUBMIT = 'forward_prefetch_submit'
    BACKWARD_FETCH_SUBMIT = 'backward_fetch_submit'
    BACKWARD_FETCH_WAIT = 'backward_fetch_wait'
    BACKWARD_PREFETCH_SUBMIT = 'backward_prefetch_wait'
    FORWARD_ALL_GATHER = 'forward_all_gather'
    BACKWARD_ALL_GATHER = 'backward_all_gather'
    """Handles partitioning and gathering of parameters."""

    @dataclass
    class __ParamInTrace:
        param: Parameter
        step_id_last_used_at: int

    def __init__(
        self,
        prefetch_bucket_sz: int,
        max_reuse_distance_in_numel: int,
        max_available_parameters_in_numel: int,
        allgather_stream: get_accelerator().Stream,
        inflight_param_registry: InflightParamRegistry,
        prefetch_nvme: bool = False,
        timers=None,
        zero_config=None,
        zero_quantized_weights=False,
        zero_quantized_nontrainable_weights=False,
    ) -> None:
        # mapping of param -> handle for each param that is currently in flight
        self.__inflight_param_registry = inflight_param_registry
        # keeps track of the number of submodules invoked so far.
        self.__step_id: int = 0
        # network tracing mode
        self.__trace_mode: ZeRoTraceMode = ZeRoTraceMode.INVALID
        # sequence of submodules/parameters in forward pass + backward pass
        self.__submodule_order: Iterable[Module] = []
        self.__param_order: Iterable[__class__.__ParamInTrace] = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(lambda: collections.deque())
        # number of available params, and max number of available params
        self.__n_available_params: int = 0
        self.__max_n_available_params: int = max_available_parameters_in_numel
        # max distance between two use of the module beyond which module is released
        self.__max_reuse_dist_in_numel: int = max_reuse_distance_in_numel
        # queue for parameters to fetch. parameters will be popped off the left
        # side of the dequeue as they are fetched
        self.__param_queue: Deque[__class__.__ParamInTrace] = None
        self.__prefetch_bucket_sz: int = prefetch_bucket_sz
        self.__prefetch_nvme: bool = prefetch_nvme
        self.hierarchy: int = 0
        self.zero_quantized_weights = zero_quantized_weights
        self.zero_quantized_nontrainable_weights = zero_quantized_nontrainable_weights

        # stream that will be used for allgather operations
        self.__allgather_stream: get_accelerator().Stream = allgather_stream

        # limit the number of fetch events that can be queued at once
        # otherwise, what happens is memory is allocated by the host thread at the
        # time of the call, but not used until later by the asynchronous cuda stream.
        # allowing an infinite number of these to queue up causes a lot of memory
        # pressure that then becomes detrimental to performance.
        # this is a much less elegant way of fixing this vs something like using
        # cudaMallocAsync/cudaFreeAsync. Choosing to not expose this to the user now
        # because ideally in the future its replaced by an async allocation
        # mechanism which doesn't require any configuration by the user.
        self.__ongoing_fetch_events: Deque[get_accelerator().Event] = collections.deque()
        # TODO. make this configurable via JSON
        self.__max_ongoing_fetch_events: int = 2
        self.__profiler = PartitionedParameterProfiler(timers if ENABLE_PROFILER else None)

        # CPU 캐시 관련 설정
        logger.info(f"[Init] zero_config type: {type(zero_config)}")
        if zero_config:
            logger.info(f"[Init] zero_config contents: {zero_config}")
        
        self.release_to_cpu = True  # 강제 설정
        logger.info(f"[Init] release_to_cpu setting: {self.release_to_cpu}")
        
        if self.release_to_cpu:
            self.cpu_buffer_size = zero_config.get('release_to_cpu_buffer_size', 1e11) if zero_config else 1e11
            self.pin_memory = zero_config.get('release_to_cpu_pin_memory', True) if zero_config else True
            self.cpu_param_cache = collections.OrderedDict()  # LRU 캐시
            self.cpu_buffer_used = 0
            self.param_access_stats = collections.defaultdict(int)  # 파라미터 접근 통계
            self.cache_hits = 0
            self.cache_misses = 0
            logger.info(f"Initialized CPU cache with size {self.cpu_buffer_size/1e9:.2f}GB")

        # 노드 내 GPU 그룹 설정
        self.local_rank = dist.get_rank()
        self.local_world_size = dist.get_world_size()
        self.gpus_per_node = zero_config.get('gpus_per_node', 8) if zero_config else 8
        self.node_id = self.local_rank // self.gpus_per_node
        
        # 노드 내 GPU 그룹 생성
        node_start_rank = self.node_id * self.gpus_per_node
        node_end_rank = min((self.node_id + 1) * self.gpus_per_node, self.local_world_size)
        self.local_group = dist.new_group(ranks=list(range(node_start_rank, node_end_rank)))
        
        logger.info(f"[Init] Local group: node_id={self.node_id}, "
                   f"local_rank={self.local_rank}, "
                   f"gpus_per_node={self.gpus_per_node}, "
                   f"group={self.local_group}")

    """Tracing and Tracking
    TODO. consider performing trace before initializing PartitionedParameterCoordinator
    and passing trace results into constructor. This way all the code in here can
    just assume that the trace is complete and the results can be entirely
    immutable.

    Bookkeeping operations used to track where we are in the forward/backward pass
    """

    def _clear_trace_structures(self) -> None:
        self.__submodule_order = []
        self.__param_order = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        self.__param_queue = None

    def is_complete_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.COMPLETE

    def is_invalid_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.INVALID

    def is_record_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.RECORD

    def _clean_inflight_param_registry(self) -> None:
        for param, handle in self.__inflight_param_registry.items():
            handle.wait()
            self.__release_param(param)
        self.__inflight_param_registry.clear()

    def _invalidate_trace(self) -> None:
        if self.is_invalid_trace():
            raise RuntimeError("attempted to invalidate already invalid trace")
        self.__trace_mode = ZeRoTraceMode.INVALID
        self._clear_trace_structures()
        self._clean_inflight_param_registry()

    def trace_prologue(self, sub_module: Module) -> None:
        if self.is_complete_trace():
            # sub_module must match expectation else invalidate trace cache
            if len(self.__submodule_order) <= self.__step_id:
                print_rank_0(
                    f"Invalidate trace cache @ step {self.__step_id} and module {sub_module.id}: "
                    f"cache has only {len(self.__submodule_order)} modules",
                    force=True)
                self._invalidate_trace()
                return

            if sub_module != self.__submodule_order[self.__step_id]:
                expected_module_id = self.__submodule_order[self.__step_id].id
                print_rank_0(
                    f"Invalidate trace cache @ step {self.__step_id}: "
                    f"expected module {expected_module_id}, but got module {sub_module.id}",
                    force=True)
                self._invalidate_trace()

    @compiler.disable
    def record_module(self, sub_module: Module) -> None:
        """adds sub module to trace"""
        if is_compiling():
            return

        if not self.is_record_trace():
            raise RuntimeError(f"attempted to record trace when status = {self.__trace_mode}")

        self.__submodule_order.append(sub_module)
        self.__step_id_module_fetched_for[sub_module.id].append(self.__step_id)

    def record_parameters(self, sub_module: Module) -> None:
        if is_compiling():
            return
        """adds sub module to trace"""
        if not self.is_record_trace():
            raise RuntimeError(f"attempted to record trace when status = {self.__trace_mode}")

        step_id = self.__step_id_module_fetched_for[sub_module.id].popleft()
        for param in sorted(set(iter_params(sub_module, recurse=z3_leaf_module(sub_module))), key=lambda p: p.ds_id):
            self.__param_order.append(__class__.__ParamInTrace(param=param, step_id_last_used_at=step_id))

    def construct_parameter_trace_from_module_trace(self):
        """use module trace to construct parameter trace"""
        self.__param_order = []
        for sub_module in self.__submodule_order:
            self.record_parameters(sub_module)

    @compiler.disable
    def reset_step(self) -> None:
        """indicate that we have completed one fwd+bwd for the model"""
        if is_compiling():
            return

        self._clean_inflight_param_registry()

        if not self.is_complete_trace():  # not self.trace_complete:
            # Make sure that recorded submodule orders are identical across ranks
            assert_ints_same_as_other_ranks([m.id for m in self.__submodule_order])

            if self.is_record_trace():
                # Successfully recorded a trace
                self.construct_parameter_trace_from_module_trace()
                # Make sure that recorded parameter orders are identical across ranks
                assert_ints_same_as_other_ranks([p.param.ds_id for p in self.__param_order])
                assert_ints_same_as_other_ranks([p.step_id_last_used_at for p in self.__param_order])

                self.__submodule_order = tuple(self.__submodule_order)  # freeze
                self.__param_order = tuple(self.__param_order)  # freeze
                self.__trace_mode = ZeRoTraceMode.COMPLETE
                print_rank_0(
                    f"completed record trace of {len(self.__submodule_order)} sub modules: {[m.id for m in self.__submodule_order]}",
                    force=False)
            else:
                # Enable trace recording for next forward/backward pass
                self.__trace_mode = ZeRoTraceMode.RECORD

        else:
            if self.__profiler is not None:
                self.__profiler.log_events()

        self.__param_queue = collections.deque(self.__param_order)  # reset fetch queue
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(lambda: collections.deque())
        self.__step_id = 0
        self.__profiler.reset_events()

    def _dump_params(self, tag, sub_module, params, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        param_names = [debug_param2name_id(p) for p in params]
        print_rank_0(f'{tag} step = {step_id} mod = {debug_module2name_id(sub_module)} p_names = {param_names}',
                     force=False)

    def _dump_param_ids(self, tag, mod_id, p_ids, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        print_rank_0(f'{tag} mod = {mod_id}, step = {step_id}, p_ids = {p_ids}', force=False)

    """Fetch and Release
    Fetching, prefetching, and releasing parameters
    """

    @compiler.disable
    @instrument_w_nvtx
    @torch.no_grad()
    def fetch_sub_module(self, current_submodule: Module, forward: bool) -> None:
        """This method does the following (in order):
        1. kick off fetch for parameters in immediately required sub module
        2. kick off fetch for next few parameters we will need later (prefetch)
        3. block on parameters in immediately required sub module
        """
        #phase = "Forward" if forward else "Backward"
        #logger.info(f"=== Starting {phase} for module: {current_submodule.__class__.__name__} ===")
        
        if logger.isEnabledFor(logging.DEBUG):
            debug_rank0(
                f"{self.__step_id}: M{current_submodule.id}({type(current_submodule).__name__}) P{[p.ds_id for p in iter_params(current_submodule, recurse=z3_leaf_module(current_submodule))]} "
                + str({
                    "avail": f"{self.__n_available_params:.1e}",
                    "queue_sz": f"{len(self.__param_queue or [])}",
                    "inflight": [p.ds_id for p in self.__inflight_param_registry],
                }))

        params_to_fetch = frozenset(iter_params(current_submodule, recurse=z3_leaf_module(current_submodule)))
        
        # 프로파일링 시작
        event_name = __class__.FORWARD_FETCH_SUBMIT if forward else __class__.BACKWARD_FETCH_SUBMIT
        fetch_numel = sum([p.partition_numel() for p in params_to_fetch if p.ds_status == ZeroParamStatus.NOT_AVAILABLE])
        
        if fetch_numel > 0:
            self._dump_param_ids(event_name, current_submodule.id,
                               [p.ds_id for p in params_to_fetch if p.ds_status == ZeroParamStatus.NOT_AVAILABLE])
            self.__profiler.start_event(event_name)
            
            if self.release_to_cpu:
                local_params = []
                remote_params = []
                restored_from_cache = []
                
                for param in params_to_fetch:
                    if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                        if param.ds_id in self.cpu_param_cache:
                            logger.info(f"[Cache Hit] Param {param.ds_id} found in CPU cache "
                                      f"(access count: {self.param_access_stats[param.ds_id]})")
                            self._restore_from_cache(param)
                            restored_from_cache.append(param)
                            self.cache_hits += 1
                        elif self._is_local_parameter(param):
                            local_params.append(param)
                        else:
                            remote_params.append(param)
                            self.cache_misses += 1

                # 캐시 통계는 히트가 있을 때만 출력
                if restored_from_cache:
                    hit_rate = len(restored_from_cache) / len(params_to_fetch) * 100
                    logger.info(f"[Cache Stats] Module {current_submodule.__class__.__name__}: "
                              f"hit rate {hit_rate:.1f}% ({len(restored_from_cache)}/{len(params_to_fetch)})")

                # 로컬/리모트 파라미터 처리
                if local_params:
                    logger.info(f"[{event_name}] Processing {len(local_params)} local parameters")
                    self.__all_gather_params(local_params, forward)
                    
                if remote_params:
                    try:
                        logger.info(f"[{event_name}] Processing {len(remote_params)} remote parameters via all-gather")
                        self.__all_gather_params(remote_params, forward)
                    except Exception as e:
                        logger.warning(f"[{event_name}] All-gather failed, using CPU path: {str(e)}")
                        self.__fetch_remote_params_via_cpu(remote_params)
                        
            else:
                self.__all_gather_params(params_to_fetch, forward)
            
            self.__profiler.stop_event(event_name, fetch_numel)

        # Wait 로직 유지
        wait_numel = 0
        wait_event_name = __class__.FORWARD_FETCH_WAIT if forward else __class__.BACKWARD_FETCH_WAIT
        self.__profiler.start_event(wait_event_name)
        # wait for parameters in the immediately needed submodule to become available
        for param in params_to_fetch:
            param.ds_active_sub_modules.add(current_submodule.id)
            if logger.isEnabledFor(logging.DEBUG):
                debug_rank0(f"-wait: {param.ds_summary()}")
            if param in self.__inflight_param_registry:
                wait_numel += param.partition_numel()
                with get_accelerator().stream(self.__allgather_stream):
                    while self.__ongoing_fetch_events and self.__ongoing_fetch_events[0].query():
                        self.__ongoing_fetch_events.popleft()
                    if len(self.__ongoing_fetch_events) > self.__max_ongoing_fetch_events:
                        self.__ongoing_fetch_events.popleft().synchronize()

                    self.__inflight_param_registry.pop(param).wait()

                    if not get_accelerator().handles_memory_backpressure():
                        event = get_accelerator().Event()
                        event.record()
                        self.__ongoing_fetch_events.append(event)

            assert param.ds_status == ZeroParamStatus.AVAILABLE, param.ds_summary()
        if not get_accelerator().resolves_data_dependency():
            get_accelerator().current_stream().wait_stream(self.__allgather_stream)
        self.__profiler.stop_event(wait_event_name, wait_numel)

        # kick off parameter prefetches for upcoming modules
        # don't prefetch if we dont have a completed model trace
        if self.is_complete_trace():
            # go through the parameters we need for the current module and pop them
            # off the fetch queue so that they aren't prefetched later.
            # if params have already been popped off the fetch queue by earlier
            # prefetches we won't look for them here
            discarded_from_prefetch_queue = set()
            params_not_already_fetched = set(
                filter(lambda p: self.__most_recent_step_id_param_fetched_for[p] < self.__step_id, params_to_fetch))
            while self.__param_queue and len(discarded_from_prefetch_queue) < len(params_not_already_fetched):
                param_in_trace = self.__param_queue.popleft()
                self.__most_recent_step_id_param_fetched_for[
                    param_in_trace.param] = param_in_trace.step_id_last_used_at
                discarded_from_prefetch_queue.add(param_in_trace.param)

            if discarded_from_prefetch_queue != params_not_already_fetched:
                raise RuntimeError(
                    f"tracing error at step {self.__step_id}: \n"
                    f"module id: {current_submodule.id}, training: {current_submodule.training}\n"
                    f"expected the next {len(params_not_already_fetched)} parameters in the "
                    f"parameter fetch queue to be {tuple(p.ds_summary(use_debug_name=True) for p in params_not_already_fetched)} \n"
                    f"but got \n {tuple(p.ds_summary(use_debug_name=True) for p in discarded_from_prefetch_queue)}.")

            def _is_currently_on_nvme(param):
                if param.nvme_swapper is None:
                    return False

                return param.ds_tensor.final_location == OffloadDeviceEnum.nvme \
                    and param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE

            # kick off all gather for params in the next few submodules (prefetch)
            if self.__prefetch_bucket_sz > 0:
                max_params_to_prefetch = min(self.__max_n_available_params - self.__n_available_params,
                                             self.__prefetch_bucket_sz)
                params_to_prefetch = set()
                numel_prefetching = 0
                while self.__param_queue and numel_prefetching < max_params_to_prefetch:
                    param_in_trace: __class__.__ParamInTrace = self.__param_queue.popleft()

                    if _is_currently_on_nvme(param_in_trace.param):
                        # nvme prefetch is handled elsewhere. Need to break here to preserve fetch order
                        self.__param_queue.appendleft(param_in_trace)
                        break

                    do_prefetch = param_in_trace.param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                    if param_in_trace.param in params_to_prefetch:
                        # Avoid duplicates
                        do_prefetch = False

                    self.__most_recent_step_id_param_fetched_for[param_in_trace.param] = \
                        max(self.__most_recent_step_id_param_fetched_for[param_in_trace.param],
                            param_in_trace.step_id_last_used_at)

                    if do_prefetch:
                        params_to_prefetch.add(param_in_trace.param)
                        numel_prefetching += param_in_trace.param.ds_numel

                if numel_prefetching > 0:
                    event_name = __class__.FORWARD_PREFETCH_SUBMIT if forward else __class__.BACKWARD_PREFETCH_SUBMIT
                    self.__profiler.start_event(event_name)
                    if logger.isEnabledFor(logging.DEBUG):
                        for param in params_to_prefetch:
                            debug_rank0(f"-prefetch: {param.ds_summary()}")
                    self.__all_gather_params(params_to_prefetch, forward)
                    self.__profiler.stop_event(event_name, numel_prefetching)

                if self.__prefetch_nvme:
                    self.__prefetch_nvme_param_partitions()

        self.__step_id += 1
        
        #logger.info(f"=== Completed {phase} for module: {current_submodule.__class__.__name__} ===")

    @instrument_w_nvtx
    @torch.no_grad()
    def release_sub_module(self, submodule: Module) -> None:
        """release the parameters of a sub module, assuming they meet conditions to
        be released."""
        params_to_release = (self.__params_to_release(submodule, self.__step_id) 
                            if self.is_complete_trace() 
                            else set(p.ds_id for p in iter_params(submodule, recurse=z3_leaf_module(submodule))))
        
        #logger.info(f"[Release] Module {submodule.__class__.__name__}: {len(params_to_release)} params to release")
        
        for param in iter_params(submodule, recurse=z3_leaf_module(submodule)):
            #logger.info(f"[Release Check] Param {param.ds_id}: "
            #           f"active_modules={len(param.ds_active_sub_modules)}, "
            #           f"in_release_set={param.ds_id in params_to_release}, "
            #           f"is_external={param.is_external_param}")
            
            param.ds_active_sub_modules.discard(submodule.id)
            if param.ds_id in params_to_release and not param.is_external_param:
                self.__release_param(param)

    @instrument_w_nvtx
    @torch.no_grad()
    def release_and_reset_all(self, module: Module) -> None:
        """release all module parameters"""
        for param in iter_params(module, recurse=True):
            if param in self.__inflight_param_registry:
                self.__inflight_param_registry.pop(param).wait()

            # TODO. make this throw if if there are still active submodules. currently
            # there's a hook execution issue
            param.ds_active_sub_modules.clear()
            self.__release_param(param)
        self.__n_available_params = 0
        for param in iter_params(module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")

    @instrument_w_nvtx
    def __all_gather_params(self, params: Set[Parameter], forward: bool) -> None:
        quantized_params = []
        nonquantized_params = []
        for param in params:
            if hasattr(param.ds_tensor, 'ds_quant_scale'):
                quantized_params.append(param)
            else:
                nonquantized_params.append(param)
        if quantized_params:
            self.__all_gather_params_(quantized_params, forward, quantize=True)
        if nonquantized_params:
            self.__all_gather_params_(nonquantized_params, forward, quantize=self.zero_quantized_weights)

    def __all_gather_params_(self, params: Set[Parameter], forward: bool, quantize: bool = False) -> None:
        """for each partitioned parameter, kick off an async allgather and store
        the work handle for the in flight parameters."""
        partitioned_params = []
        all_gather_numel = 0  # numel = num of elements
        for param in params:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                partitioned_params.append(param)
                all_gather_numel += param.ds_numel

        if partitioned_params:
            self.__n_available_params += all_gather_numel
            # here we need to handle a special case where some of the parameters have a valid hpz secondary tensor (e.g. they are not trainable so their secondary tensor never expire) but others do not.
            partitioned_params_with_secondary_tensors = [
                p for p in partitioned_params if p.ds_secondary_tensor is not None
            ]
            partitioned_params_without_secondary_tensors = [
                p for p in partitioned_params if p.ds_secondary_tensor is None
            ]
            for param_group in [
                    partitioned_params_with_secondary_tensors, partitioned_params_without_secondary_tensors
            ]:
                if not param_group:
                    continue
                with get_accelerator().stream(self.__allgather_stream):
                    event_name = __class__.FORWARD_ALL_GATHER if forward else __class__.BACKWARD_ALL_GATHER
                    self.__profiler.start_event(event_name)
                    handle = param_group[0].all_gather_coalesced(param_group, quantize=quantize)
                    self.__profiler.stop_event(event_name, all_gather_numel)
                for param in param_group:
                    assert param.ds_status == ZeroParamStatus.INFLIGHT, param.ds_summary()
                    self.__inflight_param_registry[param] = handle

            # Release swap buffers for persisted params on nvme since they will never be partitioned or evicted from GPU
            swap_persisted_params = [
                p for p in partitioned_params if p.ds_persist and p.ds_tensor.final_location == OffloadDeviceEnum.nvme
            ]
            if swap_persisted_params:
                swap_persisted_params[0].nvme_swapper.remove_partition_and_release_buffers(swap_persisted_params)

    @compiler.disable
    @instrument_w_nvtx
    def __release_param(self, param: Parameter) -> None:
        #logger.info(f"[Release Check] Param {param.ds_id}: "
        #           f"status={param.ds_status}, "
        #           f"active_modules={len(param.ds_active_sub_modules)}, "
        #           f"release_to_cpu={self.release_to_cpu}")
        
        if param.ds_status == ZeroParamStatus.AVAILABLE and not param.ds_active_sub_modules:
            try:
                # 같은 노드의 다른 GPU가 파라미터를 가지고 있는지 확인
                has_local_copy = self._check_local_copies(param)
                
                if has_local_copy and param.ds_status != ZeroParamStatus.AVAILABLE:
                    # AVAILABLE 상태가 아닐 때만 gather 시도
                    self._gather_from_local_gpus(param)
                elif param.ds_tensor.final_location == OffloadDeviceEnum.nvme:
                    # NVME offload 사용하는 경우
                    param.partition()
                elif self.release_to_cpu:
                    # CPU로 이동
                    logger.info(f"[Cache Store] Moving param {param.ds_id} to CPU cache")
                    self._manage_cpu_cache(param)
                    param.partition()
                
            except Exception as e:
                logger.warning(f"[Release] Error releasing param {param.ds_id}: {str(e)}")
                # 에러 발생 시 기본 동작 - CPU로 이동
                if self.release_to_cpu:
                    self._manage_cpu_cache(param)
                    param.partition()

    def _manage_cpu_cache(self, param):
        """CPU 캐시 관리 - 노드 내 GPU에 없는 파라미터만 캐시"""
        # 노드 내 GPU에 파라미터가 있는지 확인
        has_local_copy = self._check_local_copies(param)
        
        if has_local_copy:
            logger.info(f"[Cache Skip] Param {param.ds_id} exists in local GPUs, skipping CPU cache")
            return
        
        param_size = param.ds_numel * param.element_size()
        
        # 캐시 공간 확보
        while self.cpu_buffer_used + param_size > self.cpu_buffer_size and self.cpu_param_cache:
            lru_params = sorted(self.param_access_stats.items(), 
                              key=lambda x: (x[1], -list(self.cpu_param_cache.keys()).index(x[0])))
            param_to_evict = lru_params[0][0]
            
            if param_to_evict in self.cpu_param_cache:
                evicted_tensor = self.cpu_param_cache.pop(param_to_evict)
                removed_size = evicted_tensor.numel() * evicted_tensor.element_size()
                logger.info(f"[Cache Evict] Param {param_to_evict} "
                           f"(access count: {self.param_access_stats[param_to_evict]}, "
                           f"size: {removed_size/1e6:.2f}MB)")
                self.cpu_buffer_used -= removed_size
                del self.param_access_stats[param_to_evict]

        try:
            # 노드 내 GPU에 없는 파라미터만 캐시에 저장
            if self.pin_memory:
                cached_tensor = param.data.cpu().pin_memory().clone()
            else:
                cached_tensor = param.data.cpu().clone()
                
            self.cpu_param_cache[param.ds_id] = cached_tensor
            self.cpu_buffer_used += param_size
            self.param_access_stats[param.ds_id] = 1
            
            logger.info(f"[Cache Add] Param {param.ds_id} "
                       f"(size: {param_size/1e6:.2f}MB, total: {self.cpu_buffer_used/1e6:.2f}MB)")
                        
        except RuntimeError as e:
            logger.warning(f"[Cache Error] Failed to cache param {param.ds_id}: {str(e)}")

    @instrument_w_nvtx
    @functools.lru_cache(maxsize=None)
    def __params_to_release(self, submodule_to_release: Module, step_id: int) -> Set[int]:
        if not self.is_complete_trace():
            raise RuntimeError("expected trace to be complete")

        params_to_release = set(
            p.ds_id for p in iter_params(submodule_to_release, recurse=z3_leaf_module(submodule_to_release))
            if not p.ds_persist)

        # Problem: When prefetcher scans the param trace, it skips AVAILABLE params.
        # This creates issues if those params are released before the skipped uses:
        # 1) It hurts performance as the skipped uses are never prefetched.
        # 2) For nvme params, we run out of swap buffers because the prefetch order
        # diverges from the trace.
        # Solution: Don't release params whose reuse was skipped by prefetch. This is
        # possible because we detect such skips during prefetch and mark those params.
        for param in iter_params(submodule_to_release, recurse=z3_leaf_module(submodule_to_release)):
            if self.__most_recent_step_id_param_fetched_for[param] > step_id:
                params_to_release.discard(param.ds_id)

        # examine all modules within `max_reuse_dist_in_numel` of the current step,
        # if we see any of the candidate parameters to be released reoccur while
        # doing this, remove them from the set of parameters to release.
        params_traversed = 0
        for module in self.__submodule_order[step_id:]:
            if params_traversed >= self.__max_reuse_dist_in_numel:
                break
            for param in iter_params(module, recurse=z3_leaf_module(submodule_to_release)):
                params_to_release.discard(param.ds_id)
                params_traversed += param.ds_numel

        return params_to_release

    @instrument_w_nvtx
    def __prefetch_nvme_param_partitions(self) -> None:
        """swap in parameter partitions from nvme for those parameters that will be used
        after the ones that are already being prefetched into full parameters
        """
        if not self.is_complete_trace():
            return

        numel_in_flight = sum(param.ds_numel for param in self.__inflight_param_registry)

        numel_considered = 0
        swap_in_params = []
        for param_in_trace in self.__param_queue:
            param = param_in_trace.param
            if param.nvme_swapper is None:
                continue
            if (numel_considered > 2 * numel_in_flight
                    or len(swap_in_params) >= param.nvme_swapper.available_swap_in_buffers()):
                break
            if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                swap_in_params.append(param)
            numel_considered += param.ds_numel

        if swap_in_params:
            swap_in_params[0].nvme_swapper.swap_in(swap_in_params, async_op=True)

    def _restore_from_cache(self, param):
        """CPU 캐시에서 파라미터 복원 개선"""
        cached_tensor = self.cpu_param_cache[param.ds_id]
        
        # 1. GPU 메모리 명시적 관리
        if hasattr(param, 'gpu_buffer'):
            del param.gpu_buffer
        
        # 2. 동기화 보장
        with torch.cuda.stream(self.__allgather_stream):
            param.data = cached_tensor.cuda(non_blocking=True)
        
        # 3. 캐시 상태 업데이트
        param.ds_status = ZeroParamStatus.AVAILABLE
        self.param_access_stats[param.ds_id] += 1
        
        # 4. CPU 메모리 관리
        self.cpu_buffer_used -= cached_tensor.numel() * cached_tensor.element_size()
        del self.cpu_param_cache[param.ds_id]
        del cached_tensor

    def post_backward_hook(self, param):
        """파라미터 업데이트 후 CPU 캐시 동기화"""
        if self.release_to_cpu and param.requires_grad:
            if param.ds_id in self.cpu_param_cache:
                with torch.no_grad():
                    self.cpu_param_cache[param.ds_id].copy_(param.data.cpu())

    def _is_local_parameter(self, param: Parameter) -> bool:
        """Check if parameter is local to this process."""
        if not hasattr(param, 'ds_id'):
            return True
        return param.ds_status == ZeroParamStatus.AVAILABLE

    def _check_local_copies(self, param: Parameter) -> bool:
        """같은 노드의 다른 GPU가 파라미터를 가지고 있는지 확인"""
        try:
            # CPU 텐서로 통신하여 NCCL 에러 방지
            has_param = torch.tensor([param.ds_status == ZeroParamStatus.AVAILABLE], 
                                   device='cpu')
            gathered = [torch.zeros_like(has_param) for _ in range(self.gpus_per_node)]
            
            # 기존에 생성된 local_group 사용
            dist.all_gather(gathered, has_param, group=self.local_group)
            
            # 2개 이상의 GPU가 파라미터를 가지고 있는지 확인
            return sum(g.item() for g in gathered) > 1
            
        except Exception as e:
            logger.warning(f"[Local Check] Failed to check local copies for param {param.ds_id}: {str(e)}")
            # 에러 발생 시 안전하게 False 반환
            return False

    def _gather_from_local_gpus(self, param: Parameter) -> None:
        """노드 내 GPU들과 파라미터 all-gather"""
        logger.info(f"[Local Gather] Gathering param {param.ds_id} from node GPUs")
        
        # 이미 AVAILABLE 상태면 all_gather 건너뛰기
        if param.ds_status == ZeroParamStatus.AVAILABLE:
            logger.info(f"[Local Gather] Param {param.ds_id} already AVAILABLE, skipping gather")
            return
        
        with get_accelerator().stream(self.__allgather_stream):
            try:
                # 파라미터가 분할된 상태인지 확인
                if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    # 분할된 상태에서만 all_gather 수행
                    handle = param.all_gather_coalesced([param])
                    if handle:
                        handle.wait()
                        param.ds_status = ZeroParamStatus.AVAILABLE
                else:
                    logger.warning(f"[Local Gather] Param {param.ds_id} in unexpected state: {param.ds_status}")
                
            except RuntimeError as e:
                logger.warning(f"[Local Gather] Failed to gather param {param.ds_id}: {str(e)}")
                # 실패 시 CPU 캐시로 폴백
                if self.release_to_cpu:
                    logger.info(f"[Cache Store] Moving param {param.ds_id} to CPU cache")
                    self._manage_cpu_cache(param)
