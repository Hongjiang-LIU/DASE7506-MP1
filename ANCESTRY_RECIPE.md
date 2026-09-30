# 模型谱系与历史复现边界 / Ancestry and historical reproduction boundaries

## 可核验的梯度谱系 / Verifiable selected training chain

下表是**选中权重的训练目标发生次数**，不是全部超参数搜索成本，也不是去重语料 token 数。统计表构建、SVD、验证校准、EMA 快照合并不产生新神经梯度目标。选中同一轨迹上的两个快照不能重复累计祖先。

| 阶段 / Stage | 新增目标 / Added targets | 累计谱系 / Cumulative | 原始证据 |
|---|---:|---:|---|
| 从随机初始化训练 dense large，seed 31，12k steps | 98,304,000 | 98,304,000 | `gpu-large-s31/metrics.json` |
| rank-128 SVD 后恢复训练，seed 71，6k | 49,152,000 | 147,456,000 | `large-svd128/compression.json`, `lowrank128-s71/metrics.json` |
| learned pointer head，seed 103，2k | 16,384,000 | 163,840,000 | `push145-cache-head-s103/metrics.json` |
| backbone+pointer joint，seed 107，2k | 16,384,000 | 180,224,000 | `push145-cache-joint-s107/metrics.json` |
| dropout 0.2 续训，seed 131，4k | 32,768,000 | 212,992,000 | `push146-continue-drop20/metrics.json` |
| signed-distance cache，seed 149，8k | 65,536,000 | 278,528,000 | `push147-signed-cache/metrics.json` |
| Muon，seed 157，8k | 65,536,000 | 344,064,000 | `push148-muon/metrics.json` |
| low-LR Muon，seed 163，4k | 32,768,000 | 376,832,000 | `push149-muon-low/metrics.json` |
| function-preserving rank160/dense-proj/FFN144 扩容，seed173，20k | 163,840,000 | 540,672,000 | `push150-expanded-cosine/metrics.json` |
| fold-legal control 续训，seed211，12k | 98,304,000 | 638,976,000 | `push150-oof-control/metrics.json` |
| 10k/12k EMA 各半平均 + train-only 信息代理剪枝统计表 | 0 | 638,976,000 | `push151_plan.json`, `push151-average/build.json`, `push151-statistics/build.json`, combination evidence |
| FFN144→176 初始化 + seed193，20k | 163,840,000 | **802,816,000** | `push152-expanded-init/initialization.json`, `push152-expanded/metrics.json` |

机器可读 `ANCESTRY_RECORD.json` 保留每阶段 parent SHA、选中 snapshot SHA、recipe、实际记录文件 SHA、可恢复的原命令或重建命令说明。证据源复制到支持包 `evidence/ancestry/`，不需要打开模型才能阅读。

## 非梯度变换 / Transformations without new SGD

SVD 从本项目自己训练的 dense checkpoint 分解，不是使用外部权重。pointer/MKN 的来源通过各阶段 `parent_sha256` 链连接；验证集用于模型与配置选择，本轮 152/153 菜单预先固定，不声称更早每阶段都有完整预注册记录。`push151` 神经参数取同一 D-control 轨迹 10k/12k EMA 的 0.5/0.5 平均，缓冲不平均；训练目标并集取最大 638,976,000，不把两个来源相加。统计表由课程 train 构建，当前表是 order 1–5 的 21 个缓冲；其 train 扫描另行计费，不能说“本轮新增扫描 0”就等于历史统计构建成本 0。

这些变换各自有初始化/构建/校准记录。最新 B 三组没有重新扫描计数，21 个表缓冲从共同父原样继承。

## 能运行的训练复现层级 / Reproduction levels

1. **不重训评分 / Frozen scoring**：评分 ZIP 提供最终权重及完整推理源码，原课程数据由接收者提供。此层支持完整 CPU FP32 评分。
2. **末阶段匹配训练 / Matched final-stage replay**：本支持 ZIP 提供两个训练父权重、原样 `push152_train.py` 及前置凭证，可按 `README.md` 重放 B 三组。原实验的真实命令和新 GPU 数值差异边界均已说明。
3. **早期完整历史 / Entire history from scratch**：尚不能声称已完整验证。下面列出已确认的具体缺口，而不是把它隐藏成“运行一个脚本即可”。

## 已知历史缺口 / Known gaps

- 最早 `gpu-large-s31` 记录的 `train_bold.py` SHA 为 `158cbe51cd6f011849172f5643b0daaee4cc6dd0a3d21fa827f6db6428c96794`；目前源码 SHA 为 `533969a2e7d6fc814d96250b626647a5abf1bf784a56503545c76f8679c46393`，匹配后续 lowrank 恢复阶段。早期精确脚本版本未在本次材料中恢复，因此无法保证“当前 train_bold 即原首阶段源码”。
- 更早校准/训练有 recipe 与 parent SHA，但并非每一步都有完整原始命令行、完整 RNG/optimizer 状态或可移植的全部中间权重。`ANCESTRY_RECORD.json` 对不可确认的命令标为 reconstruction / not executed，不伪造历史 launch receipt。
- 全部 152/153 验证选择与成本已保存；全项目更早搜索成本和历史 test 暴露并非穷尽。这一谱系表仅描述选中模型的共享训练链。
- 重新训练可能产生不同权重和不同文件 SHA；必须保留最终冻结评分权重，不能事后替换。我们没有在准备材料时再次训练或评分以“验证相同成绩”。

因此本材料改进的是**可用的精确末阶段 recipe 与其真实输入资产**，而不是承诺从零完整历史复現已经完成。若课程要求提交一键从随机初始化完全重建全部历史的产物，仍需要整理早期原版本和所有变换前置材料；不得将这个限制改写为老师已经豁免。

## 从零训练的参考命令 / Historical command reconstruction (not a complete executable pipeline)

以下两项参数从早期 metrics 与当前 CLI 核对得到，只是研究路线说明；缺失版本及后续统计/校准/状态依赖没有被绕过。它们**不是**本支持包保证能完整复现最终 SHA 的入口。

```powershell
# Dense run: earliest source hash differs; reconstruction only, not the preserved original invocation.
python -B train_bold.py --config configs/large_gpu.json --run-dir runs/gpu-large-s31 --implementation efficient --steps 12000 --eval-every 500 --lr 0.001 --seed 31 --patience 5
# Train-derived SVD (no new SGD targets); requires the previous checkpoint.
python -B compress_lowrank.py --checkpoint runs/gpu-large-s31/checkpoint.pt --run-dir runs/large-svd128 --rank 128
# Recovery: source hash matches the currently preserved train_bold.py, but exact BF16 values are not guaranteed.
python -B train_bold.py --config runs/large-svd128/config.json --run-dir runs/lowrank128-s71 --implementation lowrank --parent runs/large-svd128/checkpoint.pt --steps 6000 --eval-every 500 --lr 0.0003 --seed 71 --patience 6
```

后续阶段的显式 seed / learning rates / best step / EMA 与父 SHA 可从 `ANCESTRY_RECORD.json` 对应完整 records 审查；生产用的末阶段准确命令在配套 README，不需要尝试执行上述不完整的历史链。
