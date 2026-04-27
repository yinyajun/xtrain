# Prepare

这份文档是 `sft_study/` 的环境准备说明。

默认假设：

- 系统：Linux x86_64
- Shell：`zsh`
- GPU：NVIDIA 4090
- 目标模型：`Qwen/Qwen2.5-7B`

如果你的环境不是这个组合，命令可能需要做小改动；尤其是 `Miniconda` 安装脚本、CUDA、`flash-attn` 这几块。

## 1. 安装 Miniconda

推荐先装 Miniconda，再单独建一个干净环境。

```bash
cd ~
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

安装完成后，重新加载 shell 配置：

```bash
source ~/.zshrc
```

可选但推荐的一步：关闭 base 环境自动激活。

```bash
conda config --set auto_activate_base false
```

## 2. 创建项目环境

这里建议用 `Python 3.11`，兼容性通常比 `3.12/3.13` 更稳，尤其是 GPU 训练栈和 `flash-attn`。

```bash
conda create -n sft-study python=3.11 -y
conda activate sft-study
python -V
```

可选：先升级基础打包工具。

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. 安装项目依赖

先安装基础依赖：

```bash
cd /Users/yinyajun/Common/Learn/codes/xtrain
pip install -r sft_study/requirements.txt
```

这一步会安装：

- `torch`
- `transformers`
- `trl`
- `datasets`
- `langdetect`
- `immutabledict`
- `peft`
- `lm-eval`
- `wandb`

补充说明：

- 这里的 `lm-eval[hf,ifeval]` 已经包含了 Hugging Face 后端和 `IFEval` 任务依赖
- 为了减少环境差异导致的漏装，`langdetect` 和 `immutabledict` 也额外显式写进了基础依赖

如果你是 Linux + NVIDIA，并且想额外启用 `flash-attn` 等加速，建议分两步装。

先装可选前置依赖：

```bash
pip install -r sft_study/requirements-extra.txt
```

再单独安装 `flash-attn`：

```bash
pip install flash-attn --no-build-isolation
```

说明：

- `requirements.txt` 是基础依赖
- `requirements-extra.txt` 现在只放可选 GPU 加速的前置依赖
- `flash-attn` 单独安装是因为它常常需要 `--no-build-isolation`
- `flash-attn` 通常只建议在 Linux + CUDA 环境安装

## 4. 检查 PyTorch 和 CUDA

先确认 `torch` 能看到显卡。

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY
```

理想输出应该包含：

- `cuda available: True`
- `device 0: NVIDIA GeForce RTX 4090`

## 5. 配置 W&B

### 5.1 获取 API Key

先去 W&B 网页端创建或查看你的 API Key：

- 登录 W&B
- 进入 `User Settings`
- 找到 `API Keys`
- 复制新的或现有的 key

### 5.2 登录或写环境变量

你可以二选一。

方式 A：直接登录

```bash
wandb login
```

或者：

```bash
wandb login YOUR_WANDB_API_KEY
```

方式 B：用环境变量

把下面这些加入 `~/.zshrc`：

```bash
export WANDB_API_KEY="你的_api_key"
export WANDB_PROJECT="sft-study"
export WANDB_ENTITY="你的用户名或团队名"
```

然后重新加载：

```bash
source ~/.zshrc
```

### 5.3 本项目里的用法

这套脚本默认：

- `REPORT_TO=none`

想把训练日志打到 W&B，就这样跑：

```bash
REPORT_TO=wandb WANDB_PROJECT=sft-study \
  bash sft_study/runs/e1_no_robots_smoke.sh
```

如果你还配置了 `WANDB_ENTITY`，它会一起生效。

## 6. 下载模型

主线默认模型现在是：

- `Qwen/Qwen2.5-7B`

最省心的方式是直接下载到 Hugging Face 本地缓存；训练脚本之后会自动复用缓存。

### 6.1 可选：先登录 Hugging Face

公开模型通常不强制登录，但先登录会更稳，尤其是在网络不稳定或后续要访问私有资源时。

```bash
hf auth login
```

### 6.2 下载到本地目录

先建一个本地模型目录：

```bash
mkdir -p /Users/yinyajun/Common/Learn/codes/xtrain/sft_study/artifacts/models/Qwen2.5-7B
```

再下载：

```bash
hf download Qwen/Qwen2.5-7B \
  --local-dir /Users/yinyajun/Common/Learn/codes/xtrain/sft_study/artifacts/models/Qwen2.5-7B \
  --exclude "*.bin"
```

说明：

- `--local-dir` 会把文件放到你指定目录
- `--exclude "*.bin"` 是为了尽量避免下载旧格式权重，优先保留 `safetensors`

如果你不想手动下载，也可以直接让训练脚本首次运行时自动拉取，但第一次启动会更慢，也不如提前准备清晰。

### 6.3 下载后检查

```bash
ls -la /Users/yinyajun/Common/Learn/codes/xtrain/sft_study/artifacts/models/Qwen2.5-7B
```

你通常会看到这类文件：

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `generation_config.json`
- `model-*.safetensors`

## 7. 可选：设置缓存目录

如果你不想把模型、数据都堆在默认缓存位置，可以在 `~/.zshrc` 里加：

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export WANDB_DIR="/Users/yinyajun/Common/Learn/codes/xtrain/sft_study/artifacts/wandb"
```

然后执行：

```bash
mkdir -p "$HF_HUB_CACHE"
mkdir -p /Users/yinyajun/Common/Learn/codes/xtrain/sft_study/artifacts/wandb
source ~/.zshrc
```

这不是必须的，但对整理磁盘会更友好。

## 8. 训练前最后验证

先验证几个关键 import：

```bash
python - <<'PY'
import datasets
import peft
import torch
import transformers
import trl
import wandb
print("imports ok")
PY
```

再验证模型和 tokenizer 能否正常加载：

```bash
python - <<'PY'
from transformers import AutoTokenizer
path = "/Users/yinyajun/Common/Learn/codes/xtrain/sft_study/artifacts/models/Qwen2.5-7B"
tok = AutoTokenizer.from_pretrained(path, use_fast=True)
print("tokenizer ok:", tok.__class__.__name__)
PY
```

如果你想顺手排查 tokenizer / chat template / special tokens，也可以直接用统一诊断脚本：

```bash
python sft_study/scripts/debug/inspect_model.py template \
  --model_name_or_path Qwen/Qwen2.5-7B
```

如果这些都没问题，就可以开始跑：

```bash
bash sft_study/runs/e0_fixed_prompts_base.sh
bash sft_study/runs/e1_no_robots_smoke.sh
```

如果你想直接用本地下载的模型目录，而不是仓库名，也可以：

```bash
MODEL=/Users/yinyajun/Common/Learn/codes/xtrain/sft_study/artifacts/models/Qwen2.5-7B \
  bash sft_study/runs/e1_no_robots_smoke.sh
```

## 9. 常见问题

### `conda: command not found`

通常是安装后没有重新加载 shell，先试：

```bash
source ~/.zshrc
```

### `wandb` 没有记录日志

先检查：

- 是否传了 `REPORT_TO=wandb`
- `WANDB_API_KEY` 是否配置正确
- 是否已经 `wandb login`

### `hf: command not found`

一般是 `huggingface_hub` 的 CLI 没进 PATH。最简单的处理方式是：

```bash
python -m pip install -U "huggingface_hub"
hf --help
```

### `flash-attn` 安装失败

先确认：

- 当前是 Linux
- CUDA 和 PyTorch 版本匹配
- 编译工具链正常

如果你看到类似：

```text
ModuleNotFoundError: No module named 'torch'
```

通常不是主环境没装 `torch`，而是 `pip` 的隔离构建环境里看不到 `torch`。这时请改用：

```bash
pip install flash-attn --no-build-isolation
```

如果只是先开始做实验，这一步可以先跳过，不影响基础训练流程。

## 参考

- Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
- Miniconda installer index: https://repo.anaconda.com/miniconda/
- Hugging Face CLI: https://huggingface.co/docs/huggingface_hub/guides/cli
- `hf download`: https://huggingface.co/docs/huggingface_hub/en/package_reference/cli
- W&B quickstart: https://docs.wandb.ai/quickstart
- W&B env vars: https://docs.wandb.ai/guides/track/environment-variables/
- W&B login CLI: https://docs.wandb.ai/ref/cli/wandb-login/
