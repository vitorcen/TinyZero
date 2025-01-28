# TinyZero AI Agent 运行记录

## 用户关键Prompts
1. "运行此项目" - 用户希望运行TinyZero项目

## 调试过程

### 1. 环境配置
1. 安装Miniconda
   - 下载Miniconda安装脚本
   - 运行安装脚本
   - 初始化conda环境

2. 创建conda环境
   ```bash
   conda create -n tinyzero python=3.10 -y
   conda activate tinyzero
   ```

3. 安装PyTorch和CUDA
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

4. 安装CUDA工具包
   ```bash
   conda install -c nvidia cuda-toolkit
   ```

5. 安装项目依赖
   ```bash
   export CUDA_HOME=/home/david/miniconda3/envs/tinyzero
   pip install -r requirements.txt
   ```

### 2. 项目结构分析
1. 主要组件：
   - `verl/trainer/` - 训练相关代码
   - `examples/` - 示例代码和配置
   - `verl/models/` - 模型定义
   - `verl/utils/` - 工具函数

2. 关键入口点：
   - `fsdp_sft_trainer.py` - SFT训练器
   - `main_eval.py` - 评估脚本
   - `main_generation.py` - 生成脚本
   - `main_ppo.py` - PPO训练器

### 3. 遇到的问题
1. CUDA相关问题：
   - 缺少CUDA工具包
   - 需要设置CUDA_HOME环境变量

2. 数据集问题：
   - 缺少GSM8K数据集
   - 需要运行数据预处理脚本

### 4. 待解决事项
1. 下载并预处理GSM8K数据集
2. 设置正确的数据路径
3. 配置模型参数和训练参数

## 尝试运行 Qwen2.5 系列模型

### 1. 尝试使用 Qwen2.5-1.5B-Instruct-AWQ 量化版本

首先尝试使用 AWQ 量化版本的模型以减少显存占用：

```bash
export N_GPUS=1
export BASE_MODEL=~/models/Qwen2.5-1.5B-Instruct-AWQ
export DATA_DIR=~/data/countdown
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-1.5b-awq
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    critic.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true
```

遇到错误：需要安装 `autoawq` 库来支持 AWQ 量化模型。安装后仍然遇到问题：

```
NotImplementedError: c10d::broadcast_: attempted to run this operator with Meta tensors, but there was no fake impl or Meta kernel registered.
```

这表明 TinyZero 框架可能还不完全支持 AWQ 量化模型。

### 2. 尝试使用 Qwen2.5-1.5B-Instruct 原始模型

接下来尝试使用非量化的原始模型：

```bash
export N_GPUS=1
export BASE_MODEL=~/models/Qwen2.5-1.5B-Instruct
export DATA_DIR=~/data/countdown
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-1.5b
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    critic.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true
```

遇到 CUDA 显存不足错误：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 90.00 MiB. GPU 0 has a total capacity of 7.78 GiB of which 6.50 MiB is free.
```

### 3. 尝试使用 Qwen2.5-1.5B-Instruct-GPTQ-Int8 量化模型

尝试使用 GPTQ Int8 量化版本的模型：

```bash
export N_GPUS=1
export BASE_MODEL=~/models/Qwen2.5-1.5B-Instruct-GPTQ-Int8
export DATA_DIR=~/data/countdown
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-1.5b-gptq
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero.sh \
    model=qwen2_5_1_5b_instruct_gptq \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    critic.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true
```

遇到的问题：

1. 缺少 `FSDP` 类：
   - 修改 `_build_critic_model_optimizer` 方法，导入 `FSDP` 类：
   ```python
   from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload
   ```

2. 配置路径错误：
   - 修改 `_build_critic_model_optimizer` 方法，使用 `config.model.fsdp_config` 代替 `config.fsdp_config`

3. 缺少 `create_optimizer` 函数：
   - 需要导入 optimizer 相关的工具函数

### 下一步计划

考虑到显存限制，建议尝试以下方案：

1. 使用更小的模型：Qwen2.5-0.5B-Chat
2. 进一步减小批次大小和其他内存相关参数
3. 探索其他内存优化技术，如 CPU 卸载等

## 2025-01-28: Qwen2.5 Training Optimization

### Issue Analysis
After analyzing the Qwen2.5-1.5B-Instruct-GPTQ-Int8 model training setup, several initialization issues were identified:

1. GPTQ model initialization conflicts with FSDP (Fully Sharded Data Parallel) setup
2. Incorrect dtype handling during model initialization
3. Device mapping conflicts between GPTQ and FSDP

### Modifications Made

#### 1. FSDP Worker Initialization Update
Modified `/verl/workers/fsdp_workers.py` to properly handle GPTQ model initialization:

```python
# Initialize model with proper GPTQ handling
model = AutoModelForCausalLM.from_pretrained(
    self.config.model.path,
    device_map=None,  # Required for FSDP
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    torch_dtype=torch.float16 if self.config.model.get('external_lib') == 'auto_gptq' else None
)

# Don't move GPTQ model to device here - it will be handled by FSDP
if self.config.model.get('external_lib') != 'auto_gptq':
    model = model.to(dtype=get_torch_dtype(self.config))
```

Key changes:
- Set `device_map=None` to avoid conflicts with FSDP
- Conditional dtype setting based on model type (GPTQ vs non-GPTQ)
- Deferred device movement for GPTQ models to FSDP handling

### Next Steps
1. Monitor training process for memory usage and performance
2. Fine-tune batch sizes if needed
3. Validate model outputs and training metrics

## 下一步计划
1. 运行数据预处理脚本准备GSM8K数据集
2. 使用示例配置运行训练或推理
3. 监控训练过程并调整参数

## 注意事项
1. 确保激活conda环境：`conda activate tinyzero`
2. 确保CUDA和PyTorch版本匹配
3. 检查数据路径和模型配置是否正确
