import os
import pytest
import torch
import deepspeed
from deepspeed.runtime.zero.penguin import Penguin_Init
from tests.unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator
import torch.distributed as dist
import tempfile
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.partition_parameters import (
    ZeroParamStatus,
    PartitionedParamStatus,
    GatheredParameters
)
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
import logging
import sys
from deepspeed.utils import logger

logger = logging.getLogger(__name__)

def random_dataloader(model, total_samples, hidden_dim, device, dtype=torch.float):
    batch_size = 4
    # [total_samples, hidden_dim] 형태로 데이터 생성
    train_data = torch.randn(total_samples, hidden_dim, dtype=dtype, device=device)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

class TestPenguinInterNodeOffload(DistributedTest):
    @property
    def world_size(self):
        return 8  # 2 nodes * 8 GPUs - 명시적으로 설정
        
    @property 
    def gpu_count(self):
        return self.world_size  # world_size와 동일하게 설정
        
    def setup_method(self, method):
        # 분산 환경이 이미 설정되어 있으므로 skip
        pass
        
    def init_distributed(self):
        # 분산 환경이 이미 설정되어 있으므로 skip
        pass
        
    def check_param_location(self, model, step_name):
        """파라미터의 현재 위치를 체크하고 출력"""
        rank = dist.get_rank()
        for name, param in model.named_parameters():
            if hasattr(param, 'ds_status'):
                location = "CPU" if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE else "GPU"
                print(f"[Rank {rank}] {step_name} - Param {name} is on {location}")

    def test(self):
        # 로그 레벨 설정
        deepspeed.utils.logger.setLevel(logging.INFO)
        
        # 환경변수 설정
        os.environ['NNODES'] = '1'
        os.environ['NDEV_PER_NODE'] = os.environ["WORLD_SIZE"]
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        config_dict = {
            "train_batch_size": world_size * 4,
            "train_micro_batch_size_per_gpu": 4,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4
                }
            },
            "zero_optimization": {
                "stage": 3,
                "penguin_hierarchial_params_gather": False,
                "penguin_shard_size": world_size,
                "allgather_bucket_size": 1e3,
                "reduce_bucket_size": 1e3,
                "stage3_prefetch_bucket_size": 1e3,
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True,
                    "buffer_count": 1,
                    "buffer_size": 1e4
                }
            }
        }
        
        hidden_dim = 10
        logger.info(f"[Rank {rank}] Initializing model with hidden_dim={hidden_dim}")
        
        # Penguin_Init 사용
        with deepspeed.zero.Penguin_Init(config_dict_or_path=config_dict):
            # 최소한의 모델 사용
            model = SimplestModel(hidden_dim)
            
            # 옵티마이저 생성
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            data_loader = random_dataloader(
                model=model,
                total_samples=50,
                hidden_dim=hidden_dim,
                device=torch.device(f'cuda:{rank}')
            )
            
            # 학습 루프
            for i, batch in enumerate(data_loader):
                # 파라미터 위치 체크
                self.check_param_location(model, "Before Forward")
                
                # 단순 스칼라 출력
                loss = model(batch[0])
                
                # 파라미터 위치 체크
                self.check_param_location(model, "Before Backward")
                
                # backward 계산
                loss.backward()
                
                # 파라미터 위치 체크
                self.check_param_location(model, "Before Step")
                
                # GatheredParameters 컨텍스트 내에서 업데이트 수행
                with GatheredParameters(model.param, modifier_rank=0):
                    optimizer.step()
                optimizer.zero_grad()
                
                if i >= 2:  # 처음 몇 iteration만
                    break

def create_penguin_comm_groups(shard_size, dp_group, hierarchical_allgather=True, mpu=None):
    ndevices_per_node = int(os.environ.get("NDEV_PER_NODE", get_accelerator().device_count()))
    n_nodes = int(os.environ.get("NNODES", "1"))
    
    # 전체 world size 확인
    world_size = ndevices_per_node * n_nodes
    assert dist.get_world_size() == world_size, "Mismatch in world size"

class SimplestModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        # world size에 맞게 파라미터 생성 (각 랭크에서 빈 텐서가 발생하지 않도록)
        self.param = torch.nn.Parameter(torch.ones(world_size))
        
    def forward(self, x):
        # GatheredParameters 컨텍스트 내에서 전체 파라미터 사용
        with GatheredParameters(self.param, modifier_rank=0):
            rank = dist.get_rank() if dist.is_initialized() else 0
            return self.param[rank] * x.mean()

def main():
    # DeepSpeed launcher가 제공하는 local_rank 사용
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # GPU 설정
    torch.cuda.set_device(local_rank)
    
    # 분산 환경 초기화가 필요한 경우
    if not dist.is_initialized():
        world_size = int(os.environ.get('WORLD_SIZE', '8'))
        rank = int(os.environ.get('RANK', str(local_rank)))
        
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
            world_size=world_size,
            rank=rank
        )
    
    # 테스트 인스턴스 생성 및 실행
    test = TestPenguinInterNodeOffload()
    test.test()

if __name__ == "__main__":
    main()
