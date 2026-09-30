# 训练与评分复现 / Training and scoring reproduction

## 1. 交付范围 / Scope

本说明配套 `MP1_training_reproduction_support.zip`。它提供本项目自行训练的共同父权重、函数保持扩容后的初始化权重、原样训练源码、固定实验计划及必要历史前置凭证，支持重放最终 **Phase-B 的 control / auxiliary / expanded 三组训练**。不需要下载外部预训练权重，也不需要早期全部优化器状态。

This is a **last-stage training replay package from included self-trained parents**, not a verified end-to-end reproduction from random initialization. The historical lineage is documented in `ANCESTRY_RECIPE.md` and `ANCESTRY_RECORD.json`. Earlier script-version gaps and unbundled intermediate states are explicitly listed. No training, scoring, or model selection was performed while preparing these submission materials.

正式评分仍使用不可变 `MP1_push152_tested_candidate.zip` 中的模型，不能用重训结果替换已提交模型：

- 最终 checkpoint SHA-256：`752beccd552870a2f83dd5a460e1d1eb37ee0e809ecf40d6cac632222776bac5`。
- 评分 ZIP SHA-256：`0281330c1af8fb3eb4af9c4950706920cc3a4b126615dbb717c37a4b73fcaf95`。
- 正式 CPU FP32 test BPB：`1.455264035047553`。
- 另有 `MP1_push153_validation_supplement.zip`，披露未采纳动态混合的 4 个额外验证选择；它不是另一个提交模型。

## 2. 两个必要训练资产 / Minimal parent assets

| 相对 `code/` 的路径 | 未压缩字节 | 用途 |
|---|---:|---|
| `runs/push151-combination/checkpoint.pt` | 54,744,064 | 三组共同统计参考；control 与 auxiliary 的模型初始化 |
| `runs/push152-expanded-init/checkpoint.pt` | 57,477,120 | expanded 的函数保持扩容初始化 |
| 合计 / Total | **112,221,184** | 约 107.02 MiB；仅用于训练重放 |

共同父 SHA：`c72cffff6b4152492518473f46a274b53272f59dacb80072cd9e1497d2e0bdf5`。

扩容初始化 SHA：`b8b2053ab2303e86cd48200a9fbc2cbd7aeea3966e5d0912a72afee14f7bad68`。

这些训练支持资产不是最终评分模型的推理依赖，不能混入候选 **64 MiB inference-assets** 统计。最终完整推理资产仍为 57,639,728 字节（54.969528 MiB），已在评分包中单独核验。支持包不包含原始 train / validation / test 文本、不包含 OOF 训练概率数组，也不包含所有历史 checkpoint。

若要重放 Phase-A 的优化器保留/清空两组，还须另提供 `push150-oof-control/train_state.pt`（70,048,584 字节，SHA `9ef09a43f4e7703e498ec6801b85738d042b5fd721f2770a8e1622d935ca0d5d`），且该分支强制核对原 GPU/CUDA/runtime；它不在本支持包范围内。本包没有把 A 的训练结果或成本删除，原完整记录保留在评分包。

## 3. 建立独立重放目录 / Clean replay workspace

请把支持 ZIP 解压到一个新的目录，不要覆盖原评分包、原实验目录或已完成训练。

目录结构：

```text
training_support/
  README.md
  ANCESTRY_RECIPE.md
  ANCESTRY_RECORD.json
  TRAINING_ASSET_MANIFEST.json
  verify_training_support.py
  code/
    push152_train.py, supporting source, configs/, tests/
    data/tokenizer.json, data/manifest.json
    runs/push152_plan.json
    runs/push151-combination/checkpoint.pt
    runs/push152-expanded-init/checkpoint.pt, initialization.json
    runs/push152-expanded-preflight/... (historical metadata only)
  evidence/ancestry/... (historical metrics and plans, not runnable outputs)
```

解压后先在 `training_support/` 运行纯文件校验，不加载模型或语料：

```powershell
python -B verify_training_support.py --before-training
```

将老师提供的 `wikitext_train.txt` 和 `wikitext_validation.txt` 原字节放到 `code/data/`。**训练不需要 test 文本**，也不要用 test 来选择重训结果。固定 tokenizer 与数据 manifest 已随包提供，训练器会自行校验 train/validation 身份；本次准备材料没有重新读取原始语料。

提供的 `authorization.json`、audit 与三对 CPU timing 是原实验的历史前置凭证。它们让未修改的训练脚本核对当时的固定计划和初始化来源，不是假称在接收者电脑上重新做过资源测量。新电脑若要验证模型资源限制，应对评分包中同一个冻结模型按课程流程单独测量。

## 4. 环境与安装 / Recorded environment

原训练环境：Windows、Python 3.12.14、PyTorch `2.7.1+cu128`、CUDA 12.8、NumPy `2.5.3`、tokenizers `0.21.4`，一张 NVIDIA GeForce RTX 5080 Laptop GPU；4 个 CPU 线程。训练 BF16，完整验证 FP32。以下安装版本来自实际记录；安装可能需要网络，训练与评分本身不联网。

在 `training_support/code/`：

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-gpu\Scripts\python.exe -m pip install numpy==2.5.3 tokenizers==0.21.4
```

需与该 PyTorch CUDA 构建兼容的 NVIDIA 驱动及 BF16 GPU。不同驱动/GPU/库版本可能造成训练数值差异；即使 recipe、种子和父权重一致，也**不保证重新训练的 checkpoint 文件 SHA 或 BPB 逐位相同**。PyTorch checkpoint 还含 provenance/runtime 等元数据，文件 SHA 不是跨机器训练确定性的承诺。

可选纯前置检查（会导入 PyTorch、哈希源码/两父文件，但不加载权重、不加载语料、不训练/评分）：

```powershell
.\.venv-gpu\Scripts\python.exe -B ..\verify_training_support.py --driver-metadata --before-training
```

## 5. 精确末阶段命令 / Exact final-stage commands

下面三个命令与原训练日志中的实际子进程参数一致，使用未修改的 `push152_train.py`；按顺序执行，避免同时抢占 GPU：

```powershell
.\.venv-gpu\Scripts\python.exe -B push152_train.py --branch control --plan runs/push152_plan.json
.\.venv-gpu\Scripts\python.exe -B push152_train.py --branch auxiliary --plan runs/push152_plan.json
.\.venv-gpu\Scripts\python.exe -B push152_train.py --branch expanded --plan runs/push152_plan.json
```

只复现最终训练分支时可只执行第三个命令；要复现等预算对照与辅助损失消融，应执行三组。**不要运行 `push152_launch.py`**：它是原项目的防覆盖/旧档案保护 launcher，需要当地历史 ZIP，训练数学由上面的同一个 child trainer 完成。

三组共有设置：seed 193，20,000 steps，batch 32 × context 256（每组新增 163,840,000 processed targets），dropout 0.2，Muon + AdamW，梯度裁剪 1.0，EMA 0.99；训练温度 1.15，验证温度 1.3，MKN 权重 0.15 / order 5 / discount scale 0.7 固定。Muon LR `5e-4`，AdamW backbone `7.5e-6`、pointer `7.5e-5`、scalar `2.5e-4`；cosine，warmup 200，floor 0.1。完整设置以 `runs/push152_plan.json` 和源码为准。

- control：FFN rank 144，纯 neural+pointer NLL。
- auxiliary：同 rank 144，同一批标签，`0.9*NLL(neural+pointer)+0.1*NLL(neural)`；不把两个损失算成两份独立数据。
- expanded：FFN rank 176，纯 neural+pointer NLL；qkv rank 160、dense projection、其余结构不变。

初始化已保留旧因子，并将新增右因子设为 0；21 个统计 buffer 从共同父逐位继承。训练模型/损失不包含统计表；完整验证预测包含固定统计专家。每组只在 step 0 验证 raw，并在 4k/8k/12k/16k/20k 验证 raw 与 EMA，合计 11 个完整验证选择；不附加温度网格。

各组写入 `runs/push152-control/`、`runs/push152-auxiliary/`、`runs/push152-expanded/`，这些目录在支持包中故意不存在。输出包括 `metrics.json`、完整 `train_state.pt`、全部声明的验证快照；原历史结果放在 `evidence/`，不会冒充新运行。若中断，只按原脚本的同目录完整状态 resume 机制处理，例如：

```powershell
.\.venv-gpu\Scripts\python.exe -B push152_train.py --branch expanded --plan runs/push152_plan.json --resume runs/push152-expanded/train_state.pt
```

只允许对中断且未完成的同一新轨迹恢复；不要删除失败记录、把完成的运行延长，或用任意 best/EMA 权重代替匹配的优化器状态。

## 6. 对照与期望结果 / Comparisons and expected outcomes

原记录中最终 20k EMA 的 validation BPB：control `1.4386994835099134`，auxiliary `1.4386843681544519`，expanded `1.4372712466444915`。扩容对同预算 control 改善约 `0.00142824`。这些是原测量，不是对重放数值的保证，不得把重训后最有利的一次 test 当成新模型选择依据。

三组共享祖先 targets `638,976,000`，每组新增 `163,840,000`，所选 expanded 谱系为 `802,816,000`。三组全部重放会产生 `491,520,000` 个新增训练目标发生次数，而不是把共同祖先重复加三遍。历史基线仅 `9,830,400` targets，**不是**与最终谱系等预算的 baseline；本轮公平对照是共享同一父模型的 B 三组。

原整个 push152 搜索另含 A 两组，因此其新增目标总数为 `622,592,000`，61 个训练内验证选择；不能用本支持包三组成本替代完整原搜索成本。push153 的 4 个验证选择须另加。更早全项目搜索历史未穷尽恢复；本材料不宣称全项目从未查看过 test。

## 7. 不重训评分与提交 / Evaluation without retraining

正式成绩复现请使用独立评分 ZIP 的根 README。其最终模型 SHA 已固定，不依赖本训练支持包。接收者把课程三份原始文本放回评分包 `code_overlay/data/` 后，用 CPU FP32、4 线程、同机前后基线核验；不要评分这里的初始化父权重当成正式模型。

原始代码、所有辅助脚本、训练支持和评分包应关联同一不可变代码版本。仓库 README 应包含本说明、AI 协助与复用来源，另附最多 10 页报告及权重下载链接。学生姓名/学号/仓库链接/权重链接由本人填写并确认；本材料不会自动提交或上传。

保留 Muon MIT 许可；数据使用课程提供的 WikiText-2/BPE-2048，训练文本仅来自课程 train。父权重是本项目自行训练所得，不是外部 pretrained weights。历史完整训练仍有版本/中间产物缺口：本支持包补齐了最后关键三组重放，但不虚称端到端从随机初始化已重新验证。
