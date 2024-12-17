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
        # 환경 변수에서 노드 수와 노드당 GPU 수를 읽어옴
        n_nodes = int(os.environ.get('NNODES', '1'))
        gpus_per_node = int(os.environ.get('NDEV_PER_NODE', '8'))
        return n_nodes * gpus_per_node

    def test(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet")
            
        n_nodes = int(os.environ.get('NNODES', '1'))
        gpus_per_node = int(os.environ.get('NDEV_PER_NODE', '8'))
        node_rank = int(os.environ.get('NODE_RANK', '0'))
        
        print(f"[Node {node_rank}] Starting test with {n_nodes} nodes, {gpus_per_node} GPUs per node")
        
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
        print(f"[Node {node_rank}, Rank {dist.get_rank()}] Initializing model with hidden_dim={hidden_dim}")

        class SimpleModel(torch.nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.cross_entropy = torch.nn.CrossEntropyLoss()
                print(f"[Node {node_rank}, Rank {dist.get_rank()}] Model parameters:")
                for name, param in self.named_parameters():
                    print(f"  - {name}: {param.shape}, device={param.device}")

            def forward(self, x, y):
                print(f"[Node {node_rank}, Rank {dist.get_rank()}] Forward pass:")
                print(f"  Input x: shape={x.shape}, device={x.device}")
                hidden = self.linear1(x)
                print(f"  After linear1: shape={hidden.shape}, device={hidden.device}")
                output = self.linear2(hidden)
                print(f"  After linear2: shape={output.shape}, device={output.device}")
                loss = self.cross_entropy(output, y)
                print(f"  Loss: {loss.item()}, device={loss.device}")
                return loss

        with deepspeed.zero.Penguin_Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim)
            print(f"[Node {node_rank}, Rank {dist.get_rank()}] Model initialized with Penguin")
            for name, param in model.named_parameters():
                print(f"  - {name}: status={param.ds_status}, device={param.device}")

        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config_dict
        )
        print(f"[Node {node_rank}, Rank {dist.get_rank()}] DeepSpeed initialized")

        # CPU 버퍼 초기화 확인
        for name, param in model.named_parameters():
            if not hasattr(param, 'penguin_cpu_buffer'):
                param.penguin_cpu_buffer = torch.zeros_like(param.data, device='cpu')
                print(f"[Node {node_rank}, Rank {dist.get_rank()}] Created CPU buffer for {name}")
                print(f"  - Original param: {param.device}, shape={param.shape}")
                print(f"  - CPU buffer: {param.penguin_cpu_buffer.device}, shape={param.penguin_cpu_buffer.shape}")

        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device
        )
        print(f"[Node {node_rank}, Rank {dist.get_rank()}] Created dataloader")

        dist.barrier()
        for i, batch in enumerate(data_loader):
            print(f"\n[Node {node_rank}, Rank {dist.get_rank()}] Batch {i}")
            
            # Forward pass 전 파라미터 상태
            print("Parameter status before forward:")
            for name, param in model.named_parameters():
                print(f"  - {name}: status={param.ds_status}, device={param.device}")
                if hasattr(param.ds_tensor, 'final_location'):
                    print(f"    final_location={param.ds_tensor.final_location}")

            loss = model(batch[0], batch[1])

            # Forward pass 후 파라미터 상태
            print("\nParameter status after forward:")
            for name, param in model.named_parameters():
                print(f"  - {name}: status={param.ds_status}, device={param.device}")
                if hasattr(param.ds_tensor, 'final_location'):
                    print(f"    final_location={param.ds_tensor.final_location}")

            model.backward(loss)
            
            # Backward pass 후 파라미터 상태
            print("\nParameter status after backward:")
            for name, param in model.named_parameters():
                print(f"  - {name}: status={param.ds_status}, device={param.device}")
                if hasattr(param.ds_tensor, 'final_location'):
                    print(f"    final_location={param.ds_tensor.final_location}")

            model.step()
            
            if i >= 2:  # 처음 몇 배치만 자세히 출력
                break

        # 학습 후 파라미터 접근 가능 여부 확인
        print(f"\n[Node {node_rank}, Rank {dist.get_rank()}] Checking parameter accessibility after training")
        with deepspeed.zero.GatheredParameters(model.parameters()):
            for name, param in model.named_parameters():
                try:
                    print(f"  - {name}: shape={param.shape}, device={param.device}")
                    print(f"    mean={param.data.mean().item()}, std={param.data.std().item()}")
                except Exception as e:
                    print(f"  - {name}: Failed to access - {str(e)}")

def create_penguin_comm_groups(shard_size, dp_group, hierarchical_allgather=True, mpu=None):
    ndevices_per_node = int(os.environ.get("NDEV_PER_NODE", get_accelerator().device_count()))
    n_nodes = int(os.environ.get("NNODES", "1"))
    
    # 전체 world size 확인
    world_size = ndevices_per_node * n_nodes
    assert dist.get_world_size() == world_size, "Mismatch in world size"
