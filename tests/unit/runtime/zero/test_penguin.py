import os
import pytest
import torch
import deepspeed
from deepspeed.runtime.zero.penguin import Penguin_Init
from unit.common import DistributedTest
from deepspeed.accelerator import get_accelerator
import torch.distributed as dist
import tempfile
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.partition_parameters import PartitionedParamStatus
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
import logging
import sys

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
        n_nodes = int(os.environ.get('NNODES', '1'))
        gpus_per_node = int(os.environ.get('NDEV_PER_NODE', '8'))
        return n_nodes * gpus_per_node

    def test(self):
        # 분산 환경 초기화 확인
        dist.barrier()  # 모든 프로세스가 여기까지 도달할 때까지 대기
        
        n_nodes = int(os.environ.get('NNODES', '1'))
        gpus_per_node = int(os.environ.get('NDEV_PER_NODE', '8'))
        node_rank = int(os.environ.get('NODE_RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        rank = dist.get_rank()
        
        print(f"Process info - Node: {node_rank}, Local rank: {local_rank}, Global rank: {rank}")
        sys.stdout.flush()  # 즉시 출력하도록 강제
        
        # 분산 환경 정보 출력
        if rank == 0:
            print(f"\nDistributed setup:")
            print(f"Number of nodes: {n_nodes}")
            print(f"GPUs per node: {gpus_per_node}")
            print(f"World size: {dist.get_world_size()}")
            print(f"Backend: {dist.get_backend()}")
            sys.stdout.flush()
        
        dist.barrier()  # 모든 프로세스가 여기까지 도달할 때까지 대기
        
        # 로깅 설정 추가
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - Node[%(node_rank)s] Rank[%(rank)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet")
            
        n_nodes = int(os.environ.get('NNODES', '1'))
        gpus_per_node = int(os.environ.get('NDEV_PER_NODE', '8'))
        node_rank = int(os.environ.get('NODE_RANK', '0'))
        rank = dist.get_rank()
        
        # logger에 node_rank와 rank 정보 추가
        logger = logging.LoggerAdapter(logger, {
            'node_rank': node_rank,
            'rank': rank
        })
        
        logger.info(f"Starting test with {n_nodes} nodes, {gpus_per_node} GPUs per node")
        
        config_dict = {
            "train_batch_size": 32,
            "gradient_accumulation_steps": 1,
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
                "penguin": {
                    "shard_size": gpus_per_node,
                    "hierarchial_params_gather": True
                }
            },
        }
        
        hidden_dim = 10
        logger.info(f"[Node {node_rank}, Rank {dist.get_rank()}] Initializing model with hidden_dim={hidden_dim}")

        class SimpleModel(torch.nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.cross_entropy = torch.nn.CrossEntropyLoss()
                logger.info(f"[Node {node_rank}, Rank {dist.get_rank()}] Model parameters:")
                for name, param in self.named_parameters():
                    logger.info(f"  - {name}: {param.shape}, device={param.device}")

            def forward(self, x, y):
                logger.info(f"[Node {node_rank}, Rank {dist.get_rank()}] Forward pass:")
                logger.info(f"  Input x: shape={x.shape}, device={x.device}")
                hidden = self.linear1(x)
                logger.info(f"  After linear1: shape={hidden.shape}, device={hidden.device}")
                output = self.linear2(hidden)
                logger.info(f"  After linear2: shape={output.shape}, device={output.device}")
                loss = self.cross_entropy(output, y)
                logger.info(f"  Loss: {loss.item()}, device={loss.device}")
                return loss

        with deepspeed.zero.Penguin_Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim)
            logger.info(f"[Node {node_rank}, Rank {dist.get_rank()}] Model initialized with Penguin")
            for name, param in model.named_parameters():
                logger.info(f"  - {name}: status={param.ds_status}, device={param.device}")

        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config_dict
        )
        logger.info(f"[Node {node_rank}, Rank {dist.get_rank()}] DeepSpeed initialized")

        # CPU 버퍼 초기화 확인
        for name, param in model.named_parameters():
            if not hasattr(param, 'penguin_cpu_buffer'):
                param.penguin_cpu_buffer = torch.zeros_like(param.data, device='cpu')
                logger.info(f"[Node {node_rank}, Rank {dist.get_rank()}] Created CPU buffer for {name}")
                logger.info(f"  - Original param: {param.device}, shape={param.shape}")
                logger.info(f"  - CPU buffer: {param.penguin_cpu_buffer.device}, shape={param.penguin_cpu_buffer.shape}")

        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device
        )
        logger.info(f"[Node {node_rank}, Rank {dist.get_rank()}] Created dataloader")

        dist.barrier()
        # Forward pass 전에 파라미터 값 저장
        initial_params = {}
        with deepspeed.zero.GatheredParameters(model.parameters()):
            for name, param in model.named_parameters():
                initial_params[name] = param.data.clone()

        # 학습 루프
        for i, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
            logger.info(f"Batch {i}, Loss: {loss.item()}")
            
            # 매 스텝마다 파라미터가 업데이트되는지 확인
            with deepspeed.zero.GatheredParameters(model.parameters()):
                for name, param in model.named_parameters():
                    param_changed = not torch.allclose(param.data, initial_params[name])
                    logger.info(f"Parameter {name} changed: {param_changed}")
            
            if i >= 2:
                break

        # 학습 후 파라미터 접근 가능 여부 확인
        logger.info(f"\n[Node {node_rank}, Rank {dist.get_rank()}] Checking parameter accessibility after training")
        with deepspeed.zero.GatheredParameters(model.parameters()):
            for name, param in model.named_parameters():
                try:
                    logger.info(f"  - {name}: shape={param.shape}, device={param.device}")
                    logger.info(f"    mean={param.data.mean().item()}, std={param.data.std().item()}")
                except Exception as e:
                    logger.error(f"  - {name}: Failed to access - {str(e)}")

def create_penguin_comm_groups(shard_size, dp_group, hierarchical_allgather=True, mpu=None):
    ndevices_per_node = int(os.environ.get("NDEV_PER_NODE", get_accelerator().device_count()))
    n_nodes = int(os.environ.get("NNODES", "1"))
    
    # 전체 world size 확인
    world_size = ndevices_per_node * n_nodes
    assert dist.get_world_size() == world_size, "Mismatch in world size"
