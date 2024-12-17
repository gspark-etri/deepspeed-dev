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
    # 1개 노드, 8개 GPU 환경으로 수정
    world_size = 8

    def test(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("CPU accelerator does not support this test yet")
            
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
                    "shard_size": 8,
                    "hierarchial_params_gather": True
                }
            },
        }
        if get_accelerator().is_fp16_supported():
            config_dict["fp16"] = {"enabled": True}
        
        hidden_dim = 10

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

        # Penguin Init으로 모델 초기화
        with deepspeed.zero.Penguin_Init(config_dict_or_path=config_dict):
            model = SimpleModel(hidden_dim)

        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config_dict
        )

        # 학습 데이터 생성
        data_loader = random_dataloader(
            model=model,
            total_samples=50,
            hidden_dim=hidden_dim,
            device=model.device
        )

        # 학습 실행 및 검증
        dist.barrier()
        for i, batch in enumerate(data_loader):
            # Forward pass 전에 파라미터 상태 확인
            for name, param in model.named_parameters():
                assert hasattr(param, 'penguin_cpu_buffer'), f"Parameter {name} missing penguin_cpu_buffer"
                assert param.penguin_cpu_buffer.device == torch.device('cpu'), \
                    f"Parameter {name}'s buffer is not on CPU"

            loss = model(batch[0], batch[1])

            # Forward pass 후 다른 노드의 파라미터가 CPU에 있는지 확인
            if dist.get_rank() % 8 == 0:  # 각 노드의 첫 번째 GPU에서만 체크
                other_node_params = [p for p in model.parameters() 
                                   if p.ds_tensor.ds_param_rank != dist.get_rank(group=p.comm.param_inter_node_shard_group)]
                for param in other_node_params:
                    assert param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE, \
                        f"Other node parameter should be NOT_AVAILABLE"
                    assert param.ds_tensor.final_location == OffloadDeviceEnum.cpu, \
                        f"Other node parameter should be on CPU"

            model.backward(loss)
            model.step()

            # Backward pass 후 파라미터가 GPU에 복원되었는지 확인
            if i == len(data_loader) - 1:  # 마지막 배치에서만 체크
                for name, param in model.named_parameters():
                    assert param.ds_tensor.status == PartitionedParamStatus.AVAILABLE, \
                        f"Parameter {name} not available after backward"
                    assert param.ds_tensor.final_location is None, \
                        f"Parameter {name} not on GPU after backward"

        # 학습 후 모델 상태 저장 및 복원 테스트
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_checkpoint(tmpdir)
            dist.barrier()  # 모든 프로세스가 저장을 완료할 때까지 대기

            # 체크포인트에서 모델 복원
            orig_state_dict = {}
            for name, param in model.module.named_parameters():
                with deepspeed.zero.GatheredParameters(param, modifier_rank=None):
                    orig_state_dict[name] = param.detach().cpu()

            # 복원된 모델과 원본 모델 비교
            if dist.get_rank() == 0:
                fp32_model = load_state_dict_from_zero_checkpoint(model.module, tmpdir)
                fp32_state_dict = fp32_model.state_dict()
                
                for name in orig_state_dict.keys():
                    assert torch.allclose(orig_state_dict[name].float(), fp32_state_dict[name].float()), \
                        f"Parameter {name} mismatch after restore"

        # 여기서 필요한 assertion 추가
        # 예: CPU offload가 제대로 되었는지, 파라미터가 올바르게 복원되었는지 등 