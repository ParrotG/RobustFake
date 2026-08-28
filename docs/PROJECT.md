# 项目说明：Transformation-Robust AIGC Recognizer

## 1. 目标与边界

本项目针对图片级 AIGC 检测，重点是在未见生成器以及真实传播变换下保持稳定。当前版本覆盖训练集流式获取、数据审计、双视图数据管道、模型、训练、验证和断点恢复，不包含最终比赛要求的目录到 JSON 推理脚本，也不包含高通残差或频域分支。

默认模型远低于题目要求的 2B 参数上限。CLIP 主干完全冻结，实际训练参数不超过约 5M，适合 8–12GB NVIDIA GPU。

## 2. 技术架构

### 2.1 基础模型

使用 `open_clip` 提供的 `ViT-B-16/openai` 视觉编码器。文本编码器不会保留在训练模型中。视觉编码器始终处于 `eval` 状态且所有参数的 `requires_grad=False`，checkpoint 不重复保存冻结权重。

### 2.2 双视图检测头

每张图片构造两个共享 backbone 的视图：

1. 保持完整构图的等比例缩放和 padding global view。
2. 从原图随机选择 50%–90% 范围的方形 local view。

两个 CLIP embedding 经过 L2 标准化后，对 view 维度计算 mean 和 standard deviation。二者拼接后进入 `LayerNorm → Linear → GELU → Dropout` 特征头，再产生图片级 logit。mean/std 聚合对视图顺序不敏感，并同时表达公共证据和区域间不一致性。

训练额外包含 projection head，只用于监督式对比损失。

### 2.3 成对鲁棒训练

同一基础图片产生四个输入：clean global、clean local、degraded global、degraded local。clean 与 degraded 使用相同 crop geometry，因此一致性损失主要约束传播退化，而不是不同内容区域。

总损失为：

```text
L = 1.0 * mean(BCE_clean, BCE_transformed)
  + 0.5 * SmoothL1(logit_transformed, stop_grad(logit_clean))
  + 0.1 * supervised_contrastive_loss
```

题目要求的所有增强均在线生成。默认 transformed 分支有 25% 保持 clean、50% 使用一个操作、25% 组合两个不同操作。还以较低权重加入 double-JPEG、WebP 和多种 resize 插值核。

## 3. 数据获取与合规

### 3.1 来源和许可

当前只使用 `OwensLab/CommunityForensics-Small`。该数据集包含约 278k 生成图和 278k 真实图，许可为 CC BY-NC-SA 4.0，仅适用于非商业研究；项目展示和发布时必须保留正确署名，并再次核对比赛使用场景和具体生成器许可。

代码必须使用 `streaming=True`，不会调用普通全量 `load_dataset`。首次运行会解析并记录当前 Hugging Face revision；后续在同一输出目录运行时复用审计文件中的 revision，避免远端更新破坏复现。

### 3.2 默认规模与抽样

默认总计 20k 基础图片：

| split | real | fake |
|---|---:|---:|
| train | 8,000 | 8,000 |
| val | 2,000 | 2,000 |

fake 的默认架构目标为 LatDiff 60%、GAN 15%、PixDiff 10%、other 15%。Systematic 子集每个 `model_name` 最多取 6 张，Manual/Commercial 上限为 100 张。fake 按生成器名称稳定哈希划分，因此同一生成器不会同时进入 train 和 val。

默认过滤：

- `nsfw_flag=true`。
- DALL-E、Dalle、OpenAI 生成器，防止污染题目 DALL·E Advanced 验证域。
- 真实图 `real_source` 含 COCO 的记录，保守规避 COCO val2017 泄漏。
- 非二分类标签、损坏图、解压炸弹、精确重复和感知哈希重复。

原始图像编码保持不变，manifest 只引用安全验证后的本地副本。图片不会预先缩放到 224×224。

### 3.3 空间保护与失败语义

默认图片字节预算为 22GiB、最大扫描 150k 行。满足全部类别和架构配额后立即停止；达到扫描或字节上限但配额不足时：

1. 原子写出当前 `manifest.jsonl` 和 `audit.json`。
2. 命令以退出码 2 失败。
3. 不会用偏斜的部分数据静默开始训练。

若 15–25GB 存储仍不足，可在配置中同时把 `train_per_class` 从 8000 降到 6000，其他代码无需修改。

## 4. 集中配置

所有调试参数均在 `configs/default.yaml`：

- `project`：seed 和 run name。
- `data`：远端 revision、路径、配额、扫描/字节上限、过滤和去重。
- `views`：输入尺寸、local crop 比例、padding 和插值。
- `augmentations`：题目变换范围、操作数量概率和额外编码增强。
- `model`：backbone 身份、embedding/head/projection 维度和 dropout。
- `loss`：三类损失权重与温度。
- `training`：epoch、micro-batch、梯度累积、AMP、优化器和恢复路径。
- `output`：运行产物目录和 checkpoint 策略。

CLI 只额外支持 `--set section.key=value`。未知 section、未知 key、错误概率和不支持的 backbone 会在下载或加载权重前报错。

## 5. 运行流程

```bash
uv sync --extra dev
uv run aigc-prepare --config configs/default.yaml
uv run aigc-train --config configs/default.yaml
```

8GB GPU 如果发生 OOM：

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set training.batch_size=2 \
  --set training.gradient_accumulation_steps=16
```

断点恢复：

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set training.resume_from=artifacts/runs/clip_b16_multiview/last.pt
```

checkpoint 包含检测头、optimizer、scheduler、AMP scaler、epoch、global step、随机状态、完整配置、数据 revision、backbone 身份和参数计数。恢复时会拒绝 backbone 或数据 revision 不一致的 checkpoint。

## 6. 验证指标与输出

每轮在固定 seed 的验证增强上分别计算 clean/transformed 的 AUROC、Average Precision、Balanced Accuracy 和 F1，并以 clean/transformed AUROC 均值作为最佳模型指标。

`metrics.jsonl` 的每条记录包含 schema version、UTC timestamp、session ID、epoch 和 global step，允许恢复训练后继续形成明确的时间序列。

## 7. 复现与已知限制

- 数据 revision、抽样 seed、生成器划分、验证增强 seed 都会固定。
- 在线训练增强会随 epoch 变化；checkpoint 保存完整 RNG 状态用于恢复。
- CommunityForensics 中 Stable Diffusion 衍生模型占比较高，即使按生成器限额抽样也不能等价于真正独立的生成架构。
- 感知哈希去重目前排除完全相同的 pHash，不进行昂贵的全局近邻检索。
- 验证集衡量未见 `model_name`，但同一基础模型的不同社区微调版本仍可能跨 split。
- 被动检测器不能作为真实性证明，实际应用必须表达不确定性并控制真实图片误报。
- 后续版本应加入 WildFake/SID_Set 外部验证、完整变换严重度矩阵、阈值校准和最终提交推理接口。
