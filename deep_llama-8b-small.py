import json
import deepspeed

# config 로드 및 확인
with open('ds_config.json') as f:
    ds_config = json.load(f)

print("DeepSpeed config:", ds_config)

# DeepSpeed 초기화
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,  # 직접 딕셔너리로 전달
    model_parameters=model.parameters()
) 