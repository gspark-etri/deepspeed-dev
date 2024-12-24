# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import stat
import torch
import hashlib
from collections import defaultdict, OrderedDict, deque
from shutil import copyfile
import gc

from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from contextlib import contextmanager

from typing import Callable, Dict, Union, Iterable, Container

import deepspeed

from deepspeed import comm as dist
from deepspeed.runtime.utils import see_memory_usage, DummyOptim
from .zero.offload_config import OffloadDeviceEnum, OffloadStateTypeEnum
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.runtime.zero.utils import is_zero_supported_optimizer, ZeRORuntimeException
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.runtime.zero.config import ZERO_OPTIMIZATION
from deepspeed.runtime.zero.penguin import Penguin_Optimizer

from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

from deepspeed.linear.optimized_linear import LoRAOptimizedLinear

from deepspeed.runtime.config import DEEPSPEED_OPTIMIZERS, \
    ADAGRAD_OPTIMIZER, ADAM_OPTIMIZER, ADAMW_OPTIMIZER, LAMB_OPTIMIZER, ONEBIT_ADAM_OPTIMIZER, ONEBIT_LAMB_OPTIMIZER, \
    TORCH_ADAM_PARAM, ADAM_W_MODE, ADAM_W_MODE_DEFAULT, ZERO_ONE_ADAM_OPTIMIZER, MUADAM_OPTIMIZER, MUADAMW_OPTIMIZER, \
    MUSGD_OPTIMIZER, LION_OPTIMIZER

from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.runtime.constants import \
    ROUTE_TRAIN, ROUTE_PREDICT, ROUTE_EVAL, \
    PLD_THETA, PLD_GAMMA, BFLOAT16, FP16, AMP, GRADIENT_ACCUMULATION_STEPS, \
    DATA_PARALLEL_GROUP, GLOBAL_RANK
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.compression import compression_scheduler
from deepspeed.compression.constants import \
    WEIGHT_QUANTIZE_IN_FORWARD_ENABLED, \
    WEIGHT_QUANTIZATION, SHARED_PARAMETERS, \
    WEIGHT_QUANTIZE_ENABLED, \
    WEIGHT_QUANTIZE_GROUPS, \
    WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE, \
    WEIGHT_QUANTIZE_CHANGE_RATIO, \
    WEIGHT_QUANTIZE_TYPE, \
    WEIGHT_QUANTIZE_ROUNDING, \
    WEIGHT_QUANTIZE_VERBOSE, \
    WEIGHT_QUANTIZE_KERNEL
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT, FROZEN_PARAM_FRAGMENTS
from deepspeed.runtime.sparse_tensor import SparseTensor

from deepspeed.runtime import lr_schedules
from deepspeed.utils import groups
from deepspeed.utils import logger, log_dist, instrument_w_nvtx
from deepspeed.utils.timer import NoopTimer, ThroughputTimer, SynchronizedWallClockTimer, \
    FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER, \
    STEP_MICRO_TIMER, \
    FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER, \
    STEP_GLOBAL_TIMER
from deepspeed.utils.debug import debug_extract_module_and_param_names, debug_clear_module_and_param_names
from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.progressive_layer_drop import ProgressiveLayerDrop
from deepspeed.runtime.utils import clip_grad_norm_
from deepspeed.runtime.eigenvalue import Eigenvalue
from deepspeed.runtime.data_pipeline.constants import DATA_SAMPLING, \
    DATA_ROUTING, DATA_SAMPLING_ENABLED, CURRICULUM_LEARNING, \
    CURRICULUM_LEARNING_ENABLED, DATA_SAMPLING_NUM_WORKERS, RANDOM_LTD, \
    RANDOM_LTD_ENABLED, RANDOM_LTD_LAYER_ID, RANDOM_LTD_LAYER_NUM, \
    RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE, RANDOM_LTD_LAYER_TOKEN_LR_ENABLED, \
    RANDOM_LTD_GLOBAL_BATCH_SIZE, RANDOM_LTD_MICRO_BATCH_SIZE, DATA_EFFICIENCY
from deepspeed.runtime.data_pipeline.curriculum_scheduler import CurriculumScheduler
from deepspeed.runtime.data_pipeline.data_routing.scheduler import RandomLTDScheduler
from deepspeed.runtime.data_pipeline.data_routing.helper import remove_random_ltd_state_dict
from deepspeed.runtime.data_pipeline.data_routing.basic_layer import RandomLayerTokenDrop

from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from .pipe.module import PipelineModule
from .utils import get_ma_status
from .compiler import is_compile_supported
from ..ops.adam import FusedAdam
from ..moe.sharded_moe import TopKGate, MOELayer
from ..moe.layer import MoE
from ..moe.utils import is_moe_param, configure_moe_param_groups
from ..git_version_info import version

from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from deepspeed.utils.logging import print_json_dist, print_configuration

from deepspeed.accelerator import get_accelerator

from deepspeed.runtime.config import DtypeEnum

MEMORY_OPT_ALLREDUCE_SIZE = 500000000

DeepSpeedOptimizerCallable = \
    Callable[[Union[Iterable[Parameter], Dict[str, Iterable]]], Optimizer]
DeepSpeedSchedulerCallable = Callable[[Optimizer], _LRScheduler]

try:
    import apex
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    # Fail silently so we don't spam logs unnecessarily if user isn't using amp
    APEX_INSTALLED = False


def split_half_float_double_sparse(tensors):
    device_type = get_accelerator().device_name()
    supported_types = get_accelerator().supported_dtypes()

    for t in tensors:
        assert t.dtype in supported_types, f"attempting to reduce an unsupported grad type: {t.dtype}"

    sparse_tensor_buckets, dense_tensor_buckets = [], []
    for i, dtype in enumerate(supported_types):
        sparse_bucket, dense_bucket = [], []
        for t in tensors:
            if t.dtype == dtype:
                if isinstance(t, SparseTensor):
                    sparse_bucket.append(t)
                else:
                    dense_bucket.append(t)
        if sparse_bucket:
            sparse_tensor_buckets.append((dtype, sparse_bucket))
        if dense_bucket:
            dense_tensor_buckets.append((dtype, dense_bucket))
    return sparse_tensor_buckets, dense_tensor_buckets


class EngineTimers(object):
    r"""Wallclock timers for DeepSpeedEngine"""

    def __init__(self, enable_micro_timers, enable_global_timers):
        self.forward_timers = []
        self.backward_timers = []
        self.backward_inner_timers = []
        self.backward_reduce_timers = []
        self.step_timers = []
        self.global_timers = []
        self.micro_timers = []

        if enable_micro_timers:
            self.forward_timers += [FORWARD_MICRO_TIMER]
            self.backward_timers += [BACKWARD_MICRO_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_MICRO_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_MICRO_TIMER]
            self.step_timers += [STEP_MICRO_TIMER]
            self.micro_timers += [
                FORWARD_MICRO_TIMER, BACKWARD_MICRO_TIMER, BACKWARD_INNER_MICRO_TIMER, BACKWARD_REDUCE_MICRO_TIMER,
                STEP_MICRO_TIMER
            ]

        if enable_global_timers:
            self.forward_timers += [FORWARD_GLOBAL_TIMER]
            self.backward_timers += [BACKWARD_GLOBAL_TIMER]
            self.backward_inner_timers += [BACKWARD_INNER_GLOBAL_TIMER]
            self.backward_reduce_timers += [BACKWARD_REDUCE_GLOBAL_TIMER]
            self.step_timers += [STEP_GLOBAL_TIMER]
            self.global_timers += [
                FORWARD_GLOBAL_TIMER, BACKWARD_GLOBAL_TIMER, BACKWARD_INNER_GLOBAL_TIMER, BACKWARD_REDUCE_GLOBAL_TIMER,
                STEP_GLOBAL_TIMER
            ]


class DeepSpeedEngine(Module):
    r"""DeepSpeed engine for training."""

    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 model_parameters=None,
                 training_data=None,
                 lr_scheduler=None,
                 mpu=None,
                 dist_init_required=None,
                 collate_fn=None,
                 config=None,
                 config_class=None,
                 mesh_device=None,
                 dont_change_device=False):
        super(DeepSpeedEngine, self).__init__()
        self.dont_change_device = dont_change_device
        self.client_optimizer = optimizer
        self.client_lr_scheduler = lr_scheduler
        self.training_data = training_data
        self.collate_fn = collate_fn
        self.mpu = mpu
        self.all_to_all_group = None
        self.data_parallel_group = None
        self.global_steps = 0
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gradient_average = True
        self.warn_unscaled_loss = True
        self.config = config
        self._config = config_class
        self.loaded_checkpoint_mp_world_size = None
        self.loaded_checkpoint_dp_world_size = None
        self.enable_backward_allreduce = True
        self.inside_no_sync_ctxt = False
        self.progressive_layer_drop = None
        self.eigenvalue = None
        self.block_eigenvalue = None
        self.gas_boundary_ctr = 0
        self.dist_backend = get_accelerator().communication_backend_name()
        self.has_moe_layers = False
        self.num_experts = []
        self.gate_modules = []
        self.moe_layers = []
        self._step_applied = False
        self._global_grad_norm = None
        self.use_ds_comm = False  # False --> Use torch.dist, True --> Use ds.comm backend.

        self.checkpoint_engine = None

        self._is_gradient_accumulation_boundary = None
        self.scale_wrt_gas = None
        self.losses = None
        self.mesh_device = mesh_device

        # for debug purposes - can then debug print: debug_get_module_name(module)
        debug_extract_module_and_param_names(model)

        if self.mesh_device:
            groups.mesh_device = self.mesh_device

        self._do_args_sanity_check(args)
        self._configure_with_arguments(args, mpu)
        self._do_sanity_check()
        see_memory_usage(f"DeepSpeed Engine: After args sanity test", force=self.memory_breakdown())
        if mpu is not None:
            if self.elasticity_enabled():
                if not self.is_elastic_model_parallel_supported():
                    assert not self.elasticity_enabled(), ("Elasticity is not currently supported"
                                                           " with model parallelism.")

        self._set_distributed_vars(args)

        dist.configure(self._config)

        self.monitor = MonitorMaster(self._config.monitor_config)

        see_memory_usage(
            f"DeepSpeed Engine: Before configure distributed model",
            force=self.memory_breakdown(),
        )

        self.pipeline_parallelism = isinstance(model, PipelineModule)

        # Configure distributed model
        self._configure_distributed_model(model)

        # needed for zero_to_fp32 weights reconstruction to remap nameless data to state_dict
        self.param_names = {param: name for name, param in model.named_parameters()}

        self._get_model_parameters()

        see_memory_usage(f"DeepSpeed Engine: After configure distributed model")

        # Configure wall clock timers
        self.timers = SynchronizedWallClockTimer()
        # Throughput timer
        self.tput_timer = ThroughputTimer(self._config.timers_config,
                                          batch_size=self.train_batch_size(),
                                          steps_per_output=self.steps_per_print(),
                                          monitor_memory=False)

        log_dist(f"DeepSpeed Flops Profiler Enabled: {self.flops_profiler_enabled()}", ranks=[0])

        if self.flops_profiler_enabled():
            self.flops_profiler = FlopsProfiler(self.module, self, self.flops_profiler_recompute_fwd_factor())

        if training_data:
            self.training_dataloader = self.deepspeed_io(training_data)
        else:
            self.training_dataloader = None

        # Configure optimizer and scheduler
        self.optimizer = None
        self.basic_optimizer = None
        self.lr_scheduler = None
        has_optimizer = False

        if optimizer or self.optimizer_name():
            has_optimizer = True
        # If no parameters given by init default to module parameters
        if model_parameters is None:
            model_parameters = self.module.parameters()

        # Convert model parameters from generator to list
        if not isinstance(model_parameters, list):
            model_parameters = list(model_parameters)

        if has_optimizer:
            self._configure_optimizer(optimizer, model_parameters)
            self._configure_lr_scheduler()
            self._report_progress(0)
        elif self.zero_optimization():
            # no optim selected but zero is enabled
            self.optimizer = self._configure_zero_optimizer(optimizer=None)
        elif self.bfloat16_enabled():
            self.optimizer = self._configure_bf16_optimizer(optimizer=None)

        # Hook optimizer for snip_momentum pruning
        if hasattr(model, 'pruners'):
            from ..compression.helper import rewrite_optimizer_step
            self.optimizer.pruners = model.pruners
            rewrite_optimizer_step(self.optimizer)

        # Bookkeeping for sparse support
        self.sparse_tensor_module_names = set()
        # if self.sparse_gradients_enabled():
        for name, module in self.module.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)) and self.sparse_gradients_enabled():
                self.sparse_tensor_module_names.add(name + ".weight")
                logger.info("Will convert {} to sparse tensor during training".format(name))

        self._optimized_linear_offload_setup()

        self.save_non_zero_checkpoint = False
        self.save_zero_checkpoint = False
        if not isinstance(self.optimizer, DeepSpeedZeRoOffload):
            self._configure_checkpointing(dist_init_required)

        if self.eigenvalue_enabled():
            self.eigenvalue = self._configure_eigenvalue()

        if self.pld_enabled():
            self.progressive_layer_drop = self._configure_progressive_layer_drop()

        if self.curriculum_enabled_legacy():
            self.curriculum_scheduler_legacy = self._configure_curriculum_scheduler_legacy()

        if self.random_ltd_enabled():
            random_ltd_config = self.random_ltd_config()
            random_ltd_config[RANDOM_LTD_GLOBAL_BATCH_SIZE] = self.train_batch_size()
            random_ltd_config[RANDOM_LTD_MICRO_BATCH_SIZE] = self.train_micro_batch_size_per_gpu()
            self.random_ltd_scheduler = self._configure_random_ltd_scheduler(random_ltd_config)

        # Engine timers

        self.engine_timers = EngineTimers(enable_micro_timers=self.wall_clock_breakdown(),
                                          enable_global_timers=self.wall_clock_breakdown()
                                          or self.flops_profiler_enabled())

        if self.global_rank == 0:
            self._config.print("DeepSpeedEngine configuration")
            if self.dump_state():
                print_configuration(self, "DeepSpeedEngine")

        # Use torch (un)flatten ops
        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        self._is_compiled = False

    def _optimized_linear_offload_setup(self):
        self.optimized_linear_base_weight_sharding = False
        self.optimized_linear_lora_enabled = False
        offload_ratio = None
        for _, module in self.module.named_modules():
            if isinstance(module, LoRAOptimizedLinear):
                self.optimized_linear_lora_enabled = True
                offload_ratio = None
                if offload_ratio is not None:
                    assert offload_ratio == module.lora_config.offload_ratio, \
                        "all lora_config offload ratios should be the same across the model"
                offload_ratio = module.lora_config.offload_ratio
                if module.zero_shards > 1:
                    # set attr so checkpoint saving can handle BWS properly
                    self.optimized_linear_base_weight_sharding = True

        if offload_ratio is None:
            # Nothing enabled, do nothing
            return

        total_params = 0
        for _, p in self.module.named_parameters():
            if hasattr(p, 'ds_optim_param'):
                total_params += p.numel()

        offload_limit = total_params * offload_ratio
        logger.info(f'offloading {offload_ratio*100}% of eligible params, specifically {offload_limit} params')
        total_offloaded = 0
        for _, p in self.module.named_parameters():
            if hasattr(p, 'ds_optim_param'):
                if total_offloaded < offload_limit:
                    total_offloaded += p.numel()
                    p.ds_offload = True
                    p.offload()
                else:
                    p.ds_offload = False

    def destroy(self):
        if self.optimizer is not None and hasattr(self.optimizer, 'destroy'):
            self.optimizer.destroy()
        debug_clear_module_and_param_names()

    def _get_model_parameters(self):
        if self.autotuning_profile_model_info():
            self.autotuning_model_info = {}
            num_params = 0
            trainable_num_params = 0

            for p in self.module.parameters():
                # since user code might call deepspeed.zero.Init() before deepspeed.initialize(), need to check the attribute to check if the parameter is partitioned in zero 3 already or not
                n = 0
                if hasattr(p, "ds_tensor"):  # if the parameter is partitioned in zero 3
                    n += p.ds_numel
                else:  # if the parameter is not partitioned in zero 3 yet
                    n += p.numel()
                num_params += n
                if p.requires_grad:
                    trainable_num_params += n
            if self.global_rank == 0:
                self.autotuning_model_info["num_params"] = num_params * self.mp_world_size
                self.autotuning_model_info["trainable_num_params"] = trainable_num_params * self.mp_world_size

            logger.info(f"model parameter = {num_params}")

    def get_batch_info(self):
        """Get all training batch related settings.
        Returns:
            train_batch_size (int): The effective training batch size. This is the amount of data
                samples that leads to one step of model update.
            train_micro_batch_size_per_gpu (int): Batch size to be processed by one GPU in one
                step (without gradient accumulation).
            gradient_accumulation_steps (int): Number of training steps to accumulate gradients
                before averaging and applying them.
        """
        return (
            self.train_batch_size,
            self.train_micro_batch_size_per_gpu,
            self.gradient_accumulation_steps,
        )

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        Raises:
            ValueError: if ``train_batch_size`` is not divisible by the
                configured micro-batch size and data parallelism.
        """
        if train_batch_size % (self.train_micro_batch_size_per_gpu() * self.dp_world_size) != 0:
            #print(f'{train_batch_size=} {self.train_micro_batch_size_per_gpu()=} {self.dp_world_size=}')
            raise ValueError(f'Train batch size must be divisible by micro-batch data parallelism')
        new_gas = train_batch_size // (self.train_micro_batch_size_per_gpu() * self.dp_world_size)
        # overwrite config
        self._config.train_batch_size = train_batch_size
        self._config.gradient_accumulation_steps = new_gas

    def set_train_micro_batch_size(self, micro_batch_size):
        """Adjust the micro batch size(i.e., the micro batch size in every data parallel group),
        while keep the gradient accumulation steps the same.
        Args:
            micro_batch_size (int): The new micro batch size for training.
        """
        # overwrite config
        new_global_batch_size = micro_batch_size * self._config.gradient_accumulation_steps * self.dp_world_size
        self._config.train_batch_size = new_global_batch_size
        self._config.train_micro_batch_size_per_gpu = micro_batch_size

    def set_data_post_process_func(self, post_process_func):
        if self.training_dataloader is not None:
            self.training_dataloader.post_process_func = post_process_func

    def set_custom_curriculum_learning_schedule(self, schedule_func_dict):
        if self.training_dataloader is not None and self.curriculum_learning_enabled():
            self.training_dataloader.data_sampler.set_custom_curriculum_learning_schedule(schedule_func_dict)

    def get_global_grad_norm(self) -> float:
        """Return the 2-norm of all gradients. If there is model parallelism,
        the norm will be global.
        The computed norm will be cached and reused until the next step() pass.
        .. note::
            In the presence of model parallelism, this is a collective call
            and acts as a barrier among ``mpu.get_model_parallel_group()``.
        Returns:
            float: norm
        """
        return self._global_grad_norm

    def __getattr__(self, name):
        """
        Pass through attributes defined in the model if they are not overridden by ds-engine.
        """

        _module = {}
        if "module" in self.__dict__:
            _module = self.__dict__['module']
        if name in dir(self):
            return getattr(self, name)
        elif name in dir(_module):
            return getattr(_module, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def checkpoint_tag_validation_enabled(self):
        return self._config.checkpoint_tag_validation_enabled

    def checkpoint_tag_validation_fail(self):
        return self._config.checkpoint_tag_validation_fail

    def elasticity_enabled(self):
        return self._config.elasticity_enabled

    def is_elastic_model_parallel_supported(self):
        if self.elasticity_enabled():
            # Add code for finding number of GPUs per node automatically
            if self._config.num_gpus_per_node % self._config.elastic_model_parallel_size == 0:
                return True
            else:
                return False

    def pld_enabled(self):
        return self._config.pld_enabled

    def pld_params(self):
        return self._config.pld_params

    def pld_theta(self):
        return self.pld_params()[PLD_THETA]

    def pld_gamma(self):
        return self.pld_params()[PLD_GAMMA]

    def eigenvalue_enabled(self):
        return self._config.eigenvalue_enabled

    def eigenvalue_verbose(self):
        return self._config.eigenvalue_verbose

    def eigenvalue_max_iter(self):
        return self._config.eigenvalue_max_iter

    def eigenvalue_tol(self):
        return self._config.eigenvalue_tol

    def eigenvalue_stability(self):
        return self._config.eigenvalue_stability

    def eigenvalue_gas_boundary_resolution(self):
        return self._config.eigenvalue_gas_boundary_resolution

    def eigenvalue_layer_name(self):
        return self._config.eigenvalue_layer_name

    def eigenvalue_layer_num(self):
        return self._config.eigenvalue_layer_num

    def curriculum_enabled_legacy(self):
        return self._config.curriculum_enabled_legacy

    def curriculum_params_legacy(self):
        return self._config.curriculum_params_legacy

    def data_efficiency_enabled(self):
        return self._config.data_efficiency_enabled

    def data_efficiency_config(self):
        return self._config.data_efficiency_config

    def data_sampling_enabled(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][DATA_SAMPLING_ENABLED]

    def data_sampling_config(self):
        return self._config.data_efficiency_config[DATA_SAMPLING]

    def curriculum_learning_enabled(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]

    def curriculum_learning_config(self):
        return self._config.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING]

    def random_ltd_enabled(self):
        return self._config.data_efficiency_config[DATA_ROUTING][RANDOM_LTD][RANDOM_LTD_ENABLED]

    def random_ltd_config(self):
        return self._config.data_efficiency_config[DATA_ROUTING][RANDOM_LTD]

    def random_ltd_initialize(self):
        assert self.random_ltd_enabled()
        random_ltd_config = self.random_ltd_config()
        random_ltd_queue = deque([x for x in sorted(random_ltd_config[RANDOM_LTD_LAYER_ID])])
        count = 0
        for name, layer in self.module.named_modules():
            if isinstance(layer, RandomLayerTokenDrop):
                if len(random_ltd_queue) != 0 and str(random_ltd_queue[0]) in name:  ###[1,2,3]
                    layer.init_config(random_ltd_config, self.random_ltd_scheduler, count)
                    random_ltd_queue.popleft()
                    count += 1

        if random_ltd_config[RANDOM_LTD_LAYER_NUM] != count:
            raise ValueError(f'random_ltd_layer_num {random_ltd_config[RANDOM_LTD_LAYER_NUM]} must be \
                equivalent to the len of random_ltd_layer_id {count}')

        if random_ltd_config[RANDOM_LTD_LAYER_TOKEN_LR_SCHEDULE][RANDOM_LTD_LAYER_TOKEN_LR_ENABLED]:
            assert self.client_lr_scheduler is None
            raise ValueError(f'not yet support')
            #self.lr_scheduler = lr_schedules.WarmupLayerTokenDecayLR(self.optimizer, self.random_ltd_scheduler)

    def get_sequence_parallel_group(self):
        return self.seq_parallel_group

    def wall_clock_breakdown(self):
        return self._config.wall_clock_breakdown

    def flops_profiler_enabled(self):
        return self._config.flops_profiler_config.enabled or self.autotuning_enabled()

    def flops_profiler_recompute_fwd_factor(self):
        return self._config.flops_profiler_config.recompute_fwd_factor

    def flops_profiler_profile_step(self):
        step = self._config.flops_profiler_config.profile_step
        if self._config.autotuning_config.enabled:
            step = self.autotuning_start_profile_step()
        return step

    def flops_profiler_module_depth(self):
        return self._config.flops_profiler_config.module_depth

    def flops_profiler_top_modules(self):
        return self._config.flops_profiler_config.top_modules

    def flops_profiler_detailed(self):
        if self._config.autotuning_config.enabled:
            return False
        return self._config.flops_profiler_config.detailed

    def flops_profiler_output_file(self):
        return self._config.flops_profiler_config.output_file

    def memory_breakdown(self):
        return self._config.memory_breakdown

    def autotuning_enabled(self):
        return self._config.autotuning_config.enabled

    def autotuning_start_profile_step(self):
        return self._config.autotuning_config.start_profile_step

    def autotuning_end_profile_step(self):
        return self._config.autotuning_config.end_profile_step

    def autotuning_metric_path(self):
        path = self._config.autotuning_config.metric_path
        if not path:
            path = os.path.join(os.getcwd(), "autotuning_metric.json")
        return path

    def autotuning_model_info_path(self):
        path = self._config.autotuning_config.model_info_path
        if not path:
            path = os.path.join(os.getcwd(), "autotuning_model_info.json")
        return path

    def autotuning_metric(self):
        return self._config.autotuning_config.metric

    def autotuning_profile_model_info(self):
        return self.autotuning_enabled(
        ) and self._config.autotuning_config.model_info and self._config.autotuning_config.model_info.get(
            "profile", False)

    def sparse_gradients_enabled(self):
        return self._config.sparse_gradients_enabled

    def train_batch_size(self):
        return self._config.train_batch_size

    def train_micro_batch_size_per_gpu(self):
        return self._config.train_micro_batch_size_per_gpu

    def optimizer_name(self):
        return (self.client_optimizer.__class__.__name__ if self.client_optimizer else self._config.optimizer_name)

    def optimizer_params(self):
        return self._config.optimizer_params

    def optimizer_legacy_fusion(self):
        return self._config.optimizer_legacy_fusion

    def scheduler_name(self):
        return self._config.scheduler_name

    def scheduler_params(self):
        return self._config.scheduler_params

    def quantize_training(self):
        return (
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS]
            [WEIGHT_QUANTIZE_IN_FORWARD_ENABLED],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_ENABLED],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_GROUPS],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS]
            [WEIGHT_QUANTIZE_FP16_MIXED_QUANTIZE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_CHANGE_RATIO],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_TYPE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_ROUNDING],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_VERBOSE],
            self._config.compression_config[WEIGHT_QUANTIZATION][SHARED_PARAMETERS][WEIGHT_QUANTIZE_KERNEL],
        )

    def zero_optimization(self):
        return self._config.zero_enabled

    def zero_allow_untested_optimizer(self):
        return self._config.zero_allow_untested_optimizer

    def zero_force_ds_cpu_optimizer(self):
        return self._config.zero_force_ds_cpu_optimizer

    def zero_reduce_scatter(self):
        return self._config.zero_config.reduce_scatter

    def zero_overlap_comm(self):
        return self._config.zero_config.overlap_comm

    def zero_offload_optimizer(self):
        return self._config.zero_config.offload_optimizer

    def zero_offload_param(self):
        return self._config.zero_config.offload_param

    def zero_use_cpu_optimizer(self):
        if self._config.zero_config.offload_optimizer is not None:
            return self._config.zero_config.offload_optimizer.device in [OffloadDeviceEnum.cpu, OffloadDeviceEnum.nvme]
        return False

    def zero_cpu_offload(self):
        if self._config.zero_config.offload_optimizer is not None:
            return self._config.zero_config.offload_optimizer.device == OffloadDeviceEnum.cpu
        return False

    def zero_partial_offload(self):
        return getattr(self._config.zero_config.offload_optimizer, "ratio", 1.0)

    def zero_sub_group_size(self):
        return self._config.zero_config.sub_group_size

    def zero_optimization_stage(self):
        return self._config.zero_optimization_stage

    def mics_shard_size(self):
        return self._config.mics_shard_size

    def zero_reduce_bucket_size(self):
        return self._config.zero_config.reduce_bucket_size

    def zero_multi_rank_bucket_allreduce(self):
        return self._config.zero_config.use_multi_rank_bucket_allreduce

    def zero_allgather_bucket_size(self):
        return self._config.zero_config.allgather_bucket_size

    def zero_optimization_partition_gradients(self):
        return self.zero_optimization_stage() >= ZeroStageEnum.gradients

    def zero_optimization_partition_weights(self):
        return self.zero_optimization_stage() >= ZeroStageEnum.weights

    def is_first_weights_partition_group(self):
        ret = True if self.mics_shard_size() < 0 \
            and self.zero_optimization_partition_weights() else False
        if self.mics_shard_size() > 0 and self.global_rank < self.mics_shard_size():
            ret = True
        return ret

    def zero_contiguous_gradients(self):
        return self._config.zero_config.contiguous_gradients

    def zero_load_from_fp32_weights(self):
        return self._config.zero_config.load_from_fp32_weights

    def zero_elastic_checkpoint(self):
        return self._config.zero_config.elastic_checkpoint

    def zero_has_nvme_offload(self):
        if not hasattr(self.optimizer, "swap_optimizer"):
            return False
        return self.optimizer.swap_optimizer or self.optimizer.params_in_nvme_and_cpu

    def zero_max_live_parameters(self):
        return self._config.zero_config.max_live_parameters

    def zero_max_reuse_distance(self):
        return self._config.zero_config.max_reuse_distance

    def zero_prefetch_bucket_size(self):
        return self._config.zero_config.prefetch_bucket_size

    def zero_module_granularity_threshold(self):
        return self._config.zero_config.module_granularity_threshold

    def zero_param_persistence_threshold(self):
        return self._config.zero_config.param_persistence_threshold

    def zero_model_persistence_threshold(self):
        return self._config.zero_config.model_persistence_threshold

    def zero_gather_16bit_weights_on_model_save(self):
        return self._config.zero_config.gather_16bit_weights_on_model_save

    def zero_grad_hooks(self):
        return self._config.zero_config.grad_hooks

    def zero_legacy_stage1(self):
        return self._config.zero_config.legacy_stage1

    def zero_ignore_unused_parameters(self):
        return self._config.zero_config.ignore_unused_parameters

    def graph_harvesting(self):
        return self._config.graph_harvesting

    def fp16_enabled(self):
        return self._config.fp16_enabled

    def bfloat16_enabled(self):
        return self._config.bfloat16_enabled

    def fp16_master_weights_and_gradients(self):
        return self._config.fp16_master_weights_and_gradients

    def amp_enabled(self):
        return self._config.amp_enabled

    def amp_params(self):
        return self._config.amp_params

    def fp16_auto_cast(self):
        return self._config.fp16_auto_cast

    def loss_scale(self):
        return self._config.loss_scale

    def gradient_accumulation_steps(self):
        return self._config.gradient_accumulation_steps

    def use_node_local_storage(self):
        return self._config.use_node_local_storage

    def load_universal_checkpoint(self):
        return self._config.load_universal_checkpoint

    @property
    def communication_data_type(self):
        res = self._config.communication_data_type
        if res is not None:
            return res

        if self.fp16_enabled():
            return torch.float16

        if self.bfloat16_enabled():
            return torch.bfloat16

        return torch.float32

    @communication_data_type.setter
    def communication_data_type(self, value):
        self._config.communication_data_type = value

    def postscale_gradients(self):
        return not self._config.prescale_gradients

    def gradient_predivide_factor(self):
        return self._config.gradient_predivide_factor

    def steps_per_print(self):
        return self._config.steps_per_print

    def zero_allgather_partitions(self):
        return self._config.zero_config.allgather_partitions

    def zero_round_robin_gradients(self):
        return self._config.zero_config.round_robin_gradients

    def zero_hpz_partition_size(self):
        return self._config.zero_config.zero_hpz_partition_size

    def zero_quantized_weights(self):
        return self._config.zero_config.zero_quantized_weights

    def zero_quantized_nontrainable_weights(self):
        return self._config.zero_config.zero_quantized_nontrainable_weights

    def zero_quantized_gradients(self):
        return self._config.zero_config.zero_quantized_gradients

    def dump_state(self):
        return self._config.dump_state

    def gradient_clipping(self):
        return self._config.gradient_clipping

    def dynamic_loss_scale(self):
        return self._config.loss_scale == 0

    def initial_dynamic_scale(self):
        return self._config.initial_dynamic_scale

    def dynamic_loss_scale_args(self):
        return self._config.dynamic_loss_scale_args

    def swap_tensor_config(self):
        return self._config.swap_tensor_config

    def aio_config(self):
        return self._config.aio_config

    def get_data_types(self):
        model_dtype = torch.float32
        if self.fp16_enabled():
            model_dtype = torch.float16
        elif self.bfloat16_enabled():
            model_dtype = torch.bfloat16

        if self._config.grad_accum_dtype is None:
            if model_dtype == torch.bfloat16 and not self.zero_optimization():
                grad_accum_dtype = torch.float32
            else:
                grad_accum_dtype = model_dtype
        else:
            grad_accum_dtype = DtypeEnum(self._config.grad_accum_dtype).value

        return (model_dtype, grad_accum_dtype)

    def _optimizer_has_ckpt_event_prologue(self):
        return self.optimizer is not None and hasattr(self.optimizer, 'checkpoint_event_prologue')

    def _optimizer_has_ckpt_event_epilogue(self):
        return self.optimizer is not None and hasattr(self.optimizer, 'checkpoint_event_epilogue')

    def _configure_lr_scheduler(self):
        if self.client_lr_scheduler:
            if isinstance(self.client_lr_scheduler, Callable):
                log_dist('DeepSpeed using client callable to create LR scheduler', ranks=[0])
                self.lr_scheduler = self.client_lr_scheduler(self.basic_optimizer)
            else:
                log_dist('DeepSpeed using client LR scheduler', ranks=[0])
                self.lr_scheduler = self.client_lr_scheduler
        else:
            # load lr scheduler from json configuration if lr scheduler is not defined and passed in
            lr_scheduler = self._scheduler_from_config(self.optimizer)
            log_dist(f"DeepSpeed using configured LR scheduler = {self.scheduler_name()}", ranks=[0])
            self.lr_scheduler = lr_scheduler

        log_dist(f'DeepSpeed LR Scheduler = {self.lr_scheduler}', ranks=[0])

    def _configure_checkpointing(self, dist_init_required):
        self.checkpoint_engine = TorchCheckpointEngine()

        if self._config is not None and self._config.nebula_config.enabled:
            try:
                from deepspeed.runtime.checkpoint_engine.nebula_checkpoint_engine import \
                    NebulaCheckpointEngine
                self.checkpoint_engine = NebulaCheckpointEngine(config_params=self._config.nebula_config)
            except ImportError as err:
                logger.error(f"No torch_nebula was found! Will fall back to torch.save. Details: {err}")
                self.checkpoint_engine = TorchCheckpointEngine()

        dp_rank = groups._get_sequence_data_parallel_rank()

        rank = self.local_rank if self.use_node_local_storage() else dp_rank

        # only the first data parallel process needs to store the model checkpoint
        # if you want to use node local storage this must be done by rank 0 on each
        # node
        self.save_non_zero_checkpoint = (rank == 0) or (self.zero_optimization_partition_weights()
                                                        and self.is_first_weights_partition_group())

        if self.zero_optimization() or self.bfloat16_enabled():
            param_rank = dist.get_rank(group=self.optimizer.dp_process_group)

            # Only the first parameter parallel process needs to store the
            # optimizer state checkpoints for zero
            self.save_zero_checkpoint = param_rank == dp_rank

    def _scheduler_from_config(self, optimizer):
        scheduler_name = self.scheduler_name()
        if scheduler_name is not None:
            if hasattr(lr_schedules, scheduler_name):
                scheduler = getattr(lr_schedules, scheduler_name)
            else:
                assert hasattr(torch.optim.lr_scheduler,
                               scheduler_name), f"DeepSpeed does not recognize LR scheduler {scheduler_name}"

                scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)

            scheduler_params = self.scheduler_params()
            instantiated_scheduler = scheduler(optimizer, **scheduler_params)
            return instantiated_scheduler
        else:
            return None

    def _set_distributed_vars(self, args):
        device_rank = args.device_rank if args is not None and hasattr(args, 'device_rank') else self.local_rank
        if device_rank >= 0:
            get_accelerator().set_device(device_rank)
            self.device = torch.device(get_accelerator().device_name(device_rank))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.global_rank = 0
            self.device = get_accelerator().device()

    # Configure based on command line arguments
    def _configure_with_arguments(self, args, mpu):
        # After the distributed backend is initialized we are guaranteed the LOCAL_RANK
        # environment variable is set. We must align args.local_rank to this value for
        # backwards compatibility with scripts relying on [args|self].local_rank containing
        # the correct local rank info. _do_args_sanity_check will ensure this is the case.

        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            ompi_local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
            local_rank = os.environ.get('LOCAL_RANK', ompi_local_rank)
            assert ompi_local_rank == local_rank, f"LOCAL_RANK ({local_rank}) != OMPI_COMM_WORLD_LOCAL_RANK ({ompi_local_rank}), " \
                "not sure how to proceed as we're seeing conflicting local rank info."
            os.environ['LOCAL_RANK'] = local_rank

        self.local_rank = int(os.environ['LOCAL_RANK'])
        if hasattr(args, 'local_rank'):
            args.local_rank = self.local_rank

    # Validate command line arguments
    def _do_args_sanity_check(self, args):
        assert "LOCAL_RANK" in os.environ or "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ, "DeepSpeed requires the LOCAL_RANK environment " \
            "variable, it is set by the deepspeed launcher, deepspeed.init_distributed, or the torch's launcher. If using a " \
            "different launcher please ensure LOCAL_RANK is set prior to initializing deepspeed."

        if hasattr(args, 'local_rank') and args.local_rank is not None:
            assert isinstance(args.local_rank,
                              int), f"args.local_rank of {args.local_rank} is an unknown type {type(args.local_rank)}"
            if args.local_rank >= 0:
                env_local_rank = int(os.environ.get("LOCAL_RANK"))
                assert (
                    env_local_rank == args.local_rank
                ), f"Mismatch in local rank setting, args.local_rank={args.local_rank} but env['LOCAL_RANK']={env_local_rank}."

    def _is_supported_optimizer(self, optimizer_name):
        return (optimizer_name in DEEPSPEED_OPTIMIZERS or getattr(torch.optim, optimizer_name, None) is not None)

    def _supported_optims(self):
        FairseqOptimizer = None
        try:
            from fairseq.optim.fairseq_optimizer import FairseqOptimizer
        except ImportError:
            pass

        expected_optim_types = [Optimizer]
        if FairseqOptimizer:
            # fairseq optims are not torch.optim objects
            expected_optim_types.append(FairseqOptimizer)
        return expected_optim_types

    # Validate configuration based on command line arguments
    def _do_sanity_check(self):
        if self.fp16_enabled() and not get_accelerator().is_fp16_supported():
            raise ValueError("Type fp16 is not supported on your device.")

        if self.bfloat16_enabled() and not get_accelerator().is_bf16_supported():
            raise ValueError("Type bf16 is not supported on your device.")

        expected_optim_types = self._supported_optims()
        expected_optim_types += [type(None), Callable]
        assert isinstance(self.client_optimizer, tuple(expected_optim_types)), \
            f'Client Optimizer is of unexpected type {type(self.client_optimizer)}'

        if not self.client_optimizer:
            if self.optimizer_name() is not None:
                assert self._is_supported_optimizer(
                    self.optimizer_name()
                ), "{} is not a supported DeepSpeed Optimizer".format(self.optimizer_name())

        if (self.optimizer_name() == LAMB_OPTIMIZER or self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER):
            assert (self.dynamic_loss_scale()), "DeepSpeed {} optimizer requires dynamic loss scaling".format(
                self.optimizer_name())

        # Detect invalid combinations of client optimizer and client scheduler
        if isinstance(self.client_lr_scheduler, _LRScheduler):
            assert isinstance(self.client_optimizer, Optimizer), \
                f'Client Optimizer (type = {type(self.client_optimizer)} is not instantiated but Client LR Scheduler is instantiated'

    def _broadcast_model(self):

        def is_replicated(p):
            if hasattr(p, "ds_status") and p.ds_status is not ZeroParamStatus.AVAILABLE:
                return False
            elif hasattr(p, 'ds_optim_param'):
                # do not broadcast OptimizedLinear parameters, they are unique per base weight shard
                return False
            return True

        for n, p in self.module.named_parameters():
            # Broadcast the model for different parameters
            if is_moe_param(p):
                if torch.is_tensor(p) and is_replicated(p):
                    dist.broadcast(p.data,
                                   groups._get_expert_broadcast_src_rank(p.group_name),
                                   group=self.expert_data_parallel_group[p.group_name])
            else:
                if torch.is_tensor(p) and is_replicated(p):
                    dist.broadcast(p.data, groups._get_broadcast_src_rank(), group=self.seq_data_parallel_group)

    @staticmethod
    def __check_params(model: Module, dtype: torch.dtype) -> None:
        return
        if not all(param.dtype == dtype for param in model.parameters()) and dist.get_rank() == 0:
            raise ValueError(f"{dtype} is enabled but the following parameters have dtype that is "
                             f"not {dtype}: "
                             f"{[(n, p.dtype) for n, p in model.named_parameters() if p.dtype != dtype]}")

    def _set_client_model(self, model):
        # register client model in _modules so that nn.module methods work correctly
        modules = self.__dict__.get('_modules')
        modules['module'] = model
        # register module attribute in engine but avoid getattr
        self.__dict__['module'] = model

    def _configure_distributed_model(self, model):
        self._set_client_model(model)
        is_zero_init_model = self.zero_optimization_partition_weights() and any(
            [hasattr(param, "ds_id") for param in self.module.parameters()])

        if self.fp16_enabled():
            if is_zero_init_model:
                self.__check_params(self.module, torch.half)
            self.module.half()
        elif self.bfloat16_enabled():
            if is_zero_init_model:
                self.__check_params(self.module, torch.bfloat16)
            self.module.bfloat16()
        else:
            self.__check_params(self.module, torch.float)

        # zero.Init() handles device placement of model
        if not (self.dont_change_device or is_zero_init_model):
            self.module.to(self.device)

        # MoE related initialization
        for _, module in self.module.named_modules():
            if isinstance(module, MoE):
                self.has_moe_layers = True
                self.num_experts.append(module.num_experts)

        if self.has_moe_layers:
            for _, module in self.module.named_modules():
                if isinstance(module, TopKGate):
                    self.gate_modules.append(module)
                    if self.wall_clock_breakdown():
                        module.wall_clock_breakdown = True
                if isinstance(module, MOELayer):
                    self.moe_layers.append(module)
                    if self.wall_clock_breakdown():
                        module.wall_clock_breakdown = True

        # Pass the mpu from here to groups. For subsequent use, just query groups
        if self.mpu is not None:
            groups.mpu = self.mpu

        # Set deepspeed parallelism spec. for the model including expert parallelism
        for _, module in self.module.named_modules():
            if hasattr(module, 'set_deepspeed_parallelism'):
                module.set_deepspeed_parallelism(self._config.use_data_before_expert_parallel_)

        # Query the groups module to get information about various parallel groups
        self.local_all_to_all_group = None
        if self.zero_quantized_gradients():
            log_dist("Using quantized gradients", ranks=[0])
            self.local_all_to_all_group = groups._get_local_all_to_all_group()
        self.data_parallel_group = groups._get_data_parallel_group()
        self.dp_world_size = groups._get_data_parallel_world_size()
        self.seq_data_parallel_group = groups._get_sequence_data_parallel_group()
        self.seq_dp_world_size = groups._get_sequence_data_parallel_world_size()
        self.mp_world_size = groups._get_model_parallel_world_size()
        self.expert_parallel_group = groups._get_expert_parallel_group_dict()
        self.expert_data_parallel_group = groups._get_expert_data_parallel_group_dict()
        self.sequence_parallel_size = groups._get_sequence_parallel_world_size()
        if self.sequence_parallel_size > 1:
            self.communication_data_type = self._config.seq_parallel_communication_data_type
            self.seq_parallel_group = groups._get_sequence_parallel_group()

        if not (self.amp_enabled() or is_zero_init_model):
            self._broadcast_model()

    # check if parameters are duplicated in optimizer param_groups
    def _check_for_duplicates(self, optimizer):
        for name, param in self.module.named_parameters():
            param_id = id(param)

            def ids_list(group):
                return [id(param) for param in group]

            occurrence = sum([
                ids_list(group['params']).count(param_id) if param_id in ids_list(group['params']) else 0
                for group in optimizer.param_groups
            ])
            assert occurrence <= 1, f"Parameter with name: {name} occurs multiple times in optimizer.param_groups. Make sure it only appears once to prevent undefined behavior."

    def _do_optimizer_sanity_check(self, basic_optimizer):
        model_dtype, grad_accum_dtype = self.get_data_types()
        zero_enabled = self.zero_optimization()
        amp_enabled = self.amp_enabled()
        # config based assertions
        assert (
            not (amp_enabled and zero_enabled)
        ), "Amp and ZeRO are not currently compatible, please use (legacy) fp16 mode which performs similar to amp opt_mode=O2"
        if zero_enabled:
            if not is_zero_supported_optimizer(basic_optimizer):
                assert (
                    self.zero_allow_untested_optimizer()
                ), 'You are using an untested ZeRO Optimizer. Please add <"zero_allow_untested_optimizer": true> in the configuration file to use it.'

                if self.global_rank == 0:
                    logger.warning("**** You are using ZeRO with an untested optimizer, proceed with caution *****")
            if model_dtype == torch.bfloat16 and grad_accum_dtype == torch.float32 and self.zero_optimization_stage(
            ) == 1 and not self.zero_cpu_offload():
                return BFLOAT16
            return ZERO_OPTIMIZATION
        elif amp_enabled:
            if model_dtype != grad_accum_dtype:
                raise NotImplementedError(
                    "Model data type and gradient accumulation data type must be equal to use Amp")
            if model_dtype == torch.bfloat16 or model_dtype == torch.float16:
                raise NotImplementedError("Cannot enable both amp with (legacy) fp16 or bfloat16 mode")
            try:
                logger.info("Initializing Apex amp from: {}".format(amp.__path__))
            except NameError:
                # If apex/amp is available it will be imported above
                raise RuntimeError("Unable to import apex/amp, please make sure it is installed")
            return AMP
        # data type checks
        elif model_dtype == grad_accum_dtype:
            if model_dtype == torch.bfloat16:
                if self.pipeline_parallelism:
                    logger.warning(
                        "**** BF16 gradient accumulation is not safe numerically with large number of accumulation steps, proceed with caution *****"
                    )
                    return BFLOAT16
                else:
                    raise NotImplementedError(
                        "Bfloat16 wrapper must use a gradient accumulation type of fp32, enable ZeRO to use Bfloat16 gradient accumulation"
                    )
            if model_dtype == torch.float16:
                return FP16
            # else optimizer_wrapper = None
        elif model_dtype == torch.bfloat16 and grad_accum_dtype == torch.float32:
            return BFLOAT16
        else:
            raise NotImplementedError("unsupported mix of model dtype and gradient accumulation type")

        return None

    # Configure optimizer
    def _configure_optimizer(self, client_optimizer, model_parameters):
        if client_optimizer is None:
            if self.has_moe_layers:
                model_parameters = configure_moe_param_groups(model_parameters)
            basic_optimizer = self._configure_basic_optimizer(model_parameters)
            log_dist(f"Using DeepSpeed Optimizer param name {self.optimizer_name()} as basic optimizer", ranks=[0])
        else:
            if isinstance(client_optimizer, tuple(self._supported_optims())):
                basic_optimizer = client_optimizer
                log_dist('Using client Optimizer as basic optimizer', ranks=[0])
            else:
                basic_optimizer = client_optimizer(model_parameters)
                log_dist('Using client callable to create basic optimizer', ranks=[0])

            if self.zero_use_cpu_optimizer() and not isinstance(basic_optimizer, deepspeed.ops.adam.DeepSpeedCPUAdam):
                if self.zero_force_ds_cpu_optimizer():
                    msg = f'You are using ZeRO-Offload with a client provided optimizer ({type(basic_optimizer)}) which in most cases will yield poor performance. Please either use deepspeed.ops.adam.DeepSpeedCPUAdam or set an optimizer in your ds-config (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters). If you really want to use a custom optimizer w. ZeRO-Offload and understand the performance impacts you can also set <"zero_force_ds_cpu_optimizer": false> in your configuration file.'
                    raise ZeRORuntimeException(msg)

        basic_optimizer.param_groups[:] = [pg for pg in basic_optimizer.param_groups if len(pg["params"]) != 0]
        log_dist("Removing param_group that has no 'params' in the basic Optimizer", ranks=[0])

        self._check_for_duplicates(basic_optimizer)

        self.basic_optimizer = basic_optimizer
        log_dist("DeepSpeed Basic Optimizer = {}".format(basic_optimizer.__class__.__name__), ranks=[0])

        optimizer_wrapper = self._do_optimizer_sanity_check(basic_optimizer)

        if optimizer_wrapper == ZERO_OPTIMIZATION:
            self.optimizer = self._configure_zero_optimizer(basic_optimizer)
        elif optimizer_wrapper == AMP:
            amp_params = self.amp_params()
            log_dist(f"Initializing AMP with these params: {amp_params}", ranks=[0])
            model, self.optimizer = amp.initialize(self.module, basic_optimizer, **amp_params)
            self._set_client_model(model)
            self._broadcast_model()
            # TODO: maybe need to broadcast experts differently?
        elif optimizer_wrapper == FP16:
            self.optimizer = self._configure_fp16_optimizer(basic_optimizer)
        elif optimizer_wrapper == BFLOAT16:
            self.optimizer = self._configure_bf16_optimizer(basic_optimizer)
        else:
            self.optimizer = basic_optimizer

        log_dist("DeepSpeed Final Optimizer = {}".format(self.optimizer.__class__.__name__), ranks=[0])

        self.compression_scheduler = self._configure_compression_scheduler()
        self.quantizer = self._configure_quantization()

    def _configure_basic_optimizer(self, model_parameters):
        optimizer_parameters = self.optimizer_params()
        if optimizer_parameters is None:
            optimizer_parameters = {}
        # print(optimizer_parameters.keys())
        if "max_grad_norm" in optimizer_parameters.keys():
            raise ValueError(
                "'max_grad_norm' is not supported as an optimizer parameter, please switch to using the deepspeed parameter 'gradient_clipping' see: https://www.deepspeed.ai/docs/config-json/#gradient-clipping for more details"
            )

        if self.optimizer_name() in [ADAM_OPTIMIZER, ADAMW_OPTIMIZER]:
            torch_adam = optimizer_parameters.pop(TORCH_ADAM_PARAM, False)
            adam_w_mode = optimizer_parameters.pop(ADAM_W_MODE, ADAM_W_MODE_DEFAULT)

            # Optimizer name of Adam forces AdamW logic unless adam_w_mode is explicitly set
            effective_adam_w_mode = self.optimizer_name() == ADAMW_OPTIMIZER or adam_w_mode

            if torch_adam:
                if not effective_adam_w_mode:
                    optimizer = torch.optim.Adam(model_parameters, **optimizer_parameters)
                else:
                    optimizer = torch.optim.AdamW(model_parameters, **optimizer_parameters)
            else:
                if self.zero_use_cpu_optimizer():
                    from deepspeed.ops.adam import DeepSpeedCPUAdam
                    optimizer = DeepSpeedCPUAdam(model_parameters,
                                                 **optimizer_parameters,
                                                 adamw_mode=effective_adam_w_mode)
                else:
                    from deepspeed.ops.adam import FusedAdam

                    optimizer = FusedAdam(
                        model_parameters,
                        **optimizer_parameters,
                        adam_w_mode=effective_adam_w_mode,
                    )

        elif self.optimizer_name() == ADAGRAD_OPTIMIZER:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.adagrad import DeepSpeedCPUAdagrad
                optimizer = DeepSpeedCPUAdagrad(model_parameters, **optimizer_parameters)
            else:
                optimizer = torch.optim.Adagrad(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == LAMB_OPTIMIZER:
            from deepspeed.ops.lamb import FusedLamb

            optimizer = FusedLamb(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == ONEBIT_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.adam import OnebitAdam

            optimizer = OnebitAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f"Currently the convergence of 1-bit Adam is only verified under FP16")
        elif self.optimizer_name() == ZERO_ONE_ADAM_OPTIMIZER:
            assert not self.zero_optimization(), "0/1 Adam is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

            optimizer = ZeroOneAdam(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f'Currently the convergence of 0/1 Adam is only verified under FP16')
        elif self.optimizer_name() == ONEBIT_LAMB_OPTIMIZER:
            assert not self.zero_optimization(), "1bit-Lamb is not compatible with ZeRO"
            from deepspeed.runtime.fp16.onebit.lamb import OnebitLamb

            optimizer = OnebitLamb(model_parameters, self, **optimizer_parameters)
            if not self.fp16_enabled():
                logger.warning(f"Currently the convergence of 1-bit Lamb is only verified under FP16")
        elif self.optimizer_name() == LION_OPTIMIZER:
            if self.zero_use_cpu_optimizer():
                from deepspeed.ops.lion import DeepSpeedCPULion
                optimizer = DeepSpeedCPULion(model_parameters, **optimizer_parameters)
            else:
                from deepspeed.ops.lion import FusedLion
                optimizer = FusedLion(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUADAM_OPTIMIZER:
            try:
                from mup import MuAdam
            except ImportError:
                logger.error(f"Install mup to use MuAdam optimizer")
            optimizer = MuAdam(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUADAMW_OPTIMIZER:
            try:
                from mup import MuAdamW
            except ImportError:
                logger.error(f"Install mup to use MuAdamW optimizer")
            optimizer = MuAdamW(model_parameters, **optimizer_parameters)
        elif self.optimizer_name() == MUSGD_OPTIMIZER:
            try:
                from mup import MuSGD
            except ImportError:
                logger.error(f"Install mup to use MuSGD optimizer")
            optimizer = MuSGD(model_parameters, **optimizer_parameters)
        else:
            torch_optimizer = getattr(torch.optim, self.optimizer_name())
            optimizer = torch_optimizer(model_parameters, **optimizer_parameters)
        return optimizer

    def _configure_compression_scheduler(self):
        return compression_scheduler(self.module, self._config.compression_config)

    def _configure_random_ltd_scheduler(self, configs):
        return RandomLTDScheduler(configs)

    def _configure_quantization(self):
        (
            quantize_weight_in_forward,
            quantize_enabled,
            q_groups,
            q_mixed_fp16,
            q_change_ratio,
            q_type,
            q_rounding,
            q_verbose,
            use_quantizer_kernel,
        ) = self.quantize_training()
        if quantize_enabled and not quantize_weight_in_forward:
            assert self.fp16_enabled(
            ), "MoQ (quantize in optimization step) weight quantization is only supported for FP16"
        quantizer = None
        if quantize_enabled and not quantize_weight_in_forward:
            from deepspeed.runtime.quantize import Quantizer

            quantizer = Quantizer(
                q_groups,
                q_mixed_fp16,
                q_change_ratio,
                q_type,
                q_rounding,
                q_verbose,
                self.eigenvalue_enabled(),
                use_quantizer_kernel,
                self.eigenvalue_layer_num() if self.eigenvalue_enabled() else 0,
            )
        return quantizer

    def _configure_fp16_optimizer(self, optimizer):
        initial_dynamic_scale = self.initial_dynamic_scale()
        dynamic_loss_args = self.dynamic_loss_scale_args()
        clip_grad = self.gradient_clipping()
        if APEX_INSTALLED:
            fused_opts = (apex.optimizers.FusedAdam, FusedAdam)
        else:
            fused_opts = FusedAdam
        if isinstance(optimizer, fused_opts) \
                or self.optimizer_name() in [ONEBIT_ADAM_OPTIMIZER, ZERO_ONE_ADAM_OPTIMIZER]:
            if self.dynamic_loss_scale():
                log_dist(f'Creating fp16 optimizer with dynamic loss scale', ranks=[0])
                timers = self.timers if self.wall_clock_breakdown() else NoopTimer()
                optimizer = FP16_Optimizer(
                    optimizer,
                    deepspeed=self,
                    dynamic_loss_scale=True,
                    initial_dynamic_scale=initial_dynamic_scale,
                    dynamic_loss_args=dynamic_loss_args,
                    mpu=self.mpu,
                    clip_grad=clip_grad,
                    fused_adam_legacy=self.optimizer_legacy_fusion(),
                    timers=timers,
                    has_moe_layers=self.has_moe_layers,
                )
            else:
                log_dist(f'Creating fp16 optimizer with static loss scale: {self.loss_scale()}', ranks=[0])
                optimizer = FP16_Optimizer(
                    optimizer,
                    deepspeed=self,
                    static_loss_scale=self.loss_scale(),
                    mpu=self.mpu,
                    clip_grad=clip_grad,
                    fused_adam_legacy=self.optimizer_legacy_fusion(),
                    has_moe_layers=self.has_moe_layers,
                )
        else:
            log_dist(f'Creating fp16 unfused optimizer with dynamic loss scale', ranks=[0])
            optimizer = FP16_UnfusedOptimizer(
                optimizer,
                deepspeed=self,
                static_loss_scale=self.loss_scale(),
                dynamic_loss_scale=self.dynamic_loss_scale(),
                dynamic_loss_args=dynamic_loss_args,
                mpu=self.mpu,
                clip_grad=clip_grad,
                fused_lamb_legacy=self.optimizer_name() == LAMB_OPTIMIZER,
            )

        return optimizer

    def _configure_bf16_optimizer(self, optimizer):
        clip_grad = self.gradient_clipping()

        if optimizer is None:
            optimizer = DummyOptim(list(self.module.parameters()))

        log_dist('Creating BF16 optimizer', ranks=[0])

        timers = self.timers if self.wall_clock_breakdown() else NoopTimer()
        optimizer = BF16_Optimizer(optimizer,
                                   self.param_names,
                                   mpu=self.mpu,
                                   clip_grad=clip_grad,
                                   allgather_bucket_size=self.zero_allgather_bucket_size(),
                                   dp_process_group=self.seq_data_parallel_group,
                                   timers=timers,
                                   grad_acc_dtype=self.get_data_types()[1],
                                   graph_harvesting=self.graph_harvesting(),
                                   immediate_grad_update=self._config.bfloat16_immediate_grad_update,
                                   has_moe_layers=self.has_moe_layers)

        return optimizer

    def _configure_zero_optimizer(self, *args, **kwargs):
        zero_stage = self.zero_optimization_stage()

        if zero_stage == 3:
            # config에서 직접 penguin 설정 확인 
            if 'penguin' in self.config.get('zero_optimization', {}):
                # Use Penguin optimizer
                if kwargs.get('optimizer') is None:
                    init_optimizer = DummyOptim(list(self.module.parameters()))
                else:
                    init_optimizer = kwargs['optimizer']
                    
                optimizer = Penguin_Optimizer(
                    module=self.module,
                    init_optimizer=init_optimizer,
                    timers=self.timers if self.wall_clock_breakdown() else NoopTimer(),
                    ds_config=self.config,
                    static_loss_scale=self.loss_scale(),
                    dynamic_loss_scale=self.dynamic_loss_scale(),
                    dynamic_loss_args=self.dynamic_loss_scale_args(),
                    clip_grad=self.gradient_clipping(),
                    contiguous_gradients=self.zero_contiguous_gradients(),
                    reduce_bucket_size=self.zero_reduce_bucket_size(),
                    prefetch_bucket_size=self.zero_prefetch_bucket_size(),
                    max_reuse_distance=self.zero_max_reuse_distance(),
                    max_live_parameters=self.zero_max_live_parameters(),
                    param_persistence_threshold=self.zero_param_persistence_threshold(),
                    model_persistence_threshold=self.zero_model_persistence_threshold(),
                    dp_process_group=self.data_parallel_group,
                    reduce_scatter=self.zero_reduce_scatter(),
                    overlap_comm=self.zero_overlap_comm(),
                    offload_optimizer_config=self.zero_offload_optimizer(),
                    offload_param_config=self.zero_offload_param(),
                    sub_group_size=self.zero_sub_group_size(),
                    mpu=self.mpu,
                    postscale_gradients=self.postscale_gradients(),
                    gradient_predivide_factor=self.gradient_predivide_factor(),
                    gradient_accumulation_steps=self.gradient_accumulation_steps()
                )
                return optimizer

        # 기존 코드 유지
        mics_shard_size = self.mics_shard_size()
        model_dtype, gradient_accumulation_dtype = self.get_data_types()
        timers = self.timers if self.wall_clock_breakdown() else NoopTimer()

        if kwargs.get('optimizer') is None:
            optimizer = DummyOptim(list(self.module.parameters()))
        else:
            optimizer = kwargs['optimizer']
        # ... (나머지 기존 코드)
