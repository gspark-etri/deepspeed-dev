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
    PartitionedParamStatus
)
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
import logging
import sys
from deepspeed.utils import logger

logger = logging.getLogger(__name__)

def random_dataloader(model, total_samples, hidden_dim, device, dtype=torch.float):
    batch_size = 4
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
        
    def test(self):
        # 환경변수를 먼저 설정
        os.environ['NNODES'] = '1'  # 2개 노드
        os.environ['NDEV_PER_NODE'] = os.environ["WORLD_SIZE"]  # 노드당 8개 GPU
        
        # 그 다음 DeepSpeed 분산 환경 초기화
        deepspeed.init_distributed("nccl")
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # batch size 계산
        batch_size_per_gpu = 4
        train_batch_size = world_size * batch_size_per_gpu
        
        config_dict = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": batch_size_per_gpu,
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
                "stage3_prefetch_bucket_size": 1e3
            }
        }
        
        hidden_dim = 10
        logger.info(f"[Rank {rank}] Initializing model with hidden_dim={hidden_dim}")
        
        # 모델 생성 및 DeepSpeed 초기화
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config_dict
        )
        
        # 데이터 로더 생성 전 동기화
        dist.barrier()
        
        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device
        )
        
        # 학습 루프
        for i, batch in enumerate(data_loader):
            dist.barrier()  # 각 배치 시작 전 동기화
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            dist.barrier()  # 각 배치 완료 후 동기화
            
            if i >= 10:
                break

def create_penguin_comm_groups(shard_size, dp_group, hierarchical_allgather=True, mpu=None):
    ndevices_per_node = int(os.environ.get("NDEV_PER_NODE", get_accelerator().device_count()))
    n_nodes = int(os.environ.get("NNODES", "1"))
    
    # 전체 world size 확인
    world_size = ndevices_per_node * n_nodes
    assert dist.get_world_size() == world_size, "Mismatch in world size"

class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden = self.linear1(x)
        output = self.linear2(hidden)
        loss = self.cross_entropy(output, y)
        return loss

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
