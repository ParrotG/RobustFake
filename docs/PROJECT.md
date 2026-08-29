# 项目说明：Transformation-Robust AIGC Recognizer

## 1. 目标与边界

本项目针对图片级 AIGC 检测，重点是在未见生成器以及真实传播变换下保持稳定。当前版本覆盖训练集分片获取、数据审计、双视图数据管道、模型、训练、验证和断点恢复，不包含最终比赛要求的目录到 JSON 推理脚本，也不包含高通残差或频域分支。

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

默认不再调用仓库级 `load_dataset(..., streaming=True)`。该数据集在 Hugging Face Dataset Viewer 中只有 4 个 `partial` 自动转换分片，实测这些分片只覆盖 LatDiff，无法满足 GAN、PixDiff 和 other 配额。默认 `local_shards` 模式直接访问固定 revision 下的原始 Parquet 文件，避免 Dataset Viewer 的不完整转换层。

默认选择原始 shard `68/70/77/78/83`（多类 fake）和 `115/116/156/157`（VISION/LandscapesHQ real），远端总大小约 9.86GiB。选择依据来自原始 Parquet 的 metadata-only 扫描；这些 shard 在过滤 COCO 后仍能覆盖默认四类架构和 train/val 配额。`hf_hub_download` 将每个文件完整缓存后再由 PyArrow 离线读取，下载中断会保留 Hub 临时状态，重跑会继续未完成文件。下载逐 shard 进行，不会触发约 241.9GiB 的 186 个原始 shard 全量下载。

`data.hf_auth` 默认为 `auto`：程序显式读取当前用户由 `hf auth login` 保存的 token，并把它传给 revision 查询和 shard 下载，但不会在日志或配置中打印或保存 token。若运行环境必须鉴权，可设为 `required`，此时找不到 token 会立即失败；公开匿名访问场景可设为 `disabled`。

### 3.2 默认规模与抽样

默认总计 20k 基础图片：

| split | real | fake |
|---|---:|---:|
| train | 8,000 | 8,000 |
| val | 2,000 | 2,000 |

fake 的默认架构目标为 LatDiff 60%、GAN 15%、PixDiff 10%、other 15%。Systematic 子集每个 `model_name` 最多取 6 张。原数据的 Manual 架构只有少量生成器（other 甚至只有一个），因此 Manual/Commercial 上限提高为 2,000，并按 `model_name + image_name` 做稳定图像级划分；否则 train/val 四架构配额在数学上不可满足。Systematic 仍按生成器名称划分并保持生成器级不相交。

默认过滤：

- `nsfw_flag=true`。
- DALL-E、Dalle、OpenAI 生成器，防止污染题目 DALL·E Advanced 验证域。
- 真实图的来源实际位于 `model_name`，而 `real_source` 为 `N/A`；程序统一解析有效来源并排除 COCO，保守规避 COCO val2017 泄漏。
- 非二分类标签、损坏图、解压炸弹、精确重复和感知哈希重复。

原始图像编码保持不变，manifest 只引用安全验证后的本地副本。图片不会预先缩放到 224×224。

### 3.3 空间保护与失败语义

默认已选 shard 缓存上限为 12GiB、最终图片字节预算为 22GiB、最大扫描 150k 行。开始下载前会用 Hub dry-run 元数据核对 shard 总大小。满足全部类别和架构配额后立即停止；达到扫描或字节上限但配额不足时：

1. 原子写出当前 `manifest.jsonl` 和 `audit.json`。
2. 命令以退出码 2 失败。
3. 不会用偏斜的部分数据静默开始训练。

此外每扫描 `checkpoint_every_scanned` 行会原子更新 manifest 和审计。收到常规异常、SIGTERM、SIGHUP 或 Ctrl-C 时也会立即保存。`local_shards` 下，完整 shard 已经在本地，恢复时重放少量行也不再消耗网络；若中断发生在 shard 下载中，Hub 下载缓存负责续传。常见 Hub 连接错误会采用指数退避自动重试。

如果中断时只留下 `images/` 而没有 manifest/audit，则无法可靠反推出生成器元数据；直接重跑即可，已有相同文件不会重复写入，但不在最终 manifest 中的孤立文件不会参与训练。

若 15–25GB 存储仍不足，可在配置中同时把 `train_per_class` 从 8000 降到 6000，其他代码无需修改。

## 4. 集中配置

所有调试参数均在 `configs/default.yaml`：

- `project`：seed 和 run name。
- `data`：固定 revision、原始 shard 列表/缓存上限、路径、配额、过滤和去重。
- `data` 中的运行参数：HF 鉴权策略、manifest 检查点间隔和网络重试退避。
- `views`：输入尺寸、local crop 比例、padding 和插值。
- `augmentations`：题目变换范围、操作数量概率和额外编码增强。
- `model`：backbone 身份、embedding/head/projection 维度和 dropout。
- `loss`：三类损失权重与温度。
- `training`：epoch、micro-batch、梯度累积、DataLoader 预取、AMP、优化器和恢复路径。
- `output`：运行产物目录和 checkpoint 策略。

CLI 只额外支持 `--set section.key=value`。未知 section、未知 key、错误概率和不支持的 backbone 会在下载或加载权重前报错。

## 5. 运行流程

```bash
uv sync --extra dev
uv run aigc-prepare --config configs/default.yaml
uv run aigc-train --config configs/default.yaml
```

当前默认训练使用 micro-batch 32、梯度累积 1 和 16 个 DataLoader worker，有效 batch size 为 32。clean/transformed 张量会合并成一次较大的 CLIP 前向，以减少小 kernel 和两次串行调用造成的 GPU 空隙。由于视觉主干冻结且前向位于 `no_grad` 中，显存占用本来就会显著低于端到端微调；判断性能时应同时观察每秒样本数和 GPU compute utilization，而不是以占满显存为目标。内存受限时应优先降低 `training.num_workers` 和 `training.batch_size`。

DataLoader worker 默认不跨 epoch 常驻，以限制 PIL/编码库长期运行时的内存高水位。训练入口将 `SIGTERM` 和 `SIGHUP` 转换成可清理的中断，并在所有退出路径显式关闭 worker；Linux worker 还设置父进程死亡信号，以应对终端或 IDE 只强制终止主进程的情况。如确认运行环境稳定，可设置 `training.persistent_workers=true` 减少每轮 worker 启动开销。

8GB GPU 如果发生 OOM：

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set training.batch_size=8 \
  --set training.gradient_accumulation_steps=4
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

训练完成后可使用题目指定的 WildFake 展示子集做独立测试。该子集严格由元数据定义为 4,998 张 COCO val2017 真实图与 8,843 张 DALL·E Advanced 生成图，存放在 `data/evaluation/`，不会写入或复用训练 manifest：

```bash
uv run aigc-prepare-official-eval --config configs/default.yaml
uv run aigc-evaluate-official --config configs/default.yaml
```

WildFake 仓库整体约 1.2TB，而 DALL·E ZIP 约 25.6GB。准备命令校验上游 archive SHA-256，并通过 ZIP HTTP Range 只提取题目指定成员，实际选择图片约 2.93GB；下载中断后根据原子 manifest 继续。若上游 archive 身份或官方元数据数量不再是 real 4,998/fake 8,843，命令会拒绝评测。

评测矩阵包括 clean、JPEG quality 90/70/50/30、Gaussian blur sigma 0.5/1.0/2.0、resize 0.5/0.25、Gaussian noise sigma 0.02/0.05/0.10、color jitter ±20% 和 center crop 80%。每个场景输出 AUROC、AP、balanced accuracy、F1、真假类别 recall 与混淆计数，同时保留逐图片置信度，便于完成鲁棒性表格和错误分析。

`metrics.jsonl` 的每条记录包含 schema version、UTC timestamp、session ID、epoch 和 global step，允许恢复训练后继续形成明确的时间序列。

## 7. 复现与已知限制

- 数据 revision、抽样 seed、生成器划分、验证增强 seed 都会固定。
- 在线训练增强会随 epoch 变化；checkpoint 保存完整 RNG 状态用于恢复。
- CommunityForensics 中 Stable Diffusion 衍生模型占比较高，即使按生成器限额抽样也不能等价于真正独立的生成架构。
- 感知哈希去重目前排除完全相同的 pHash，不进行昂贵的全局近邻检索。
- Systematic 验证集衡量未见 `model_name`；Manual/Commercial 因架构生成器数量过少采用图像级划分，不能作为未见生成器指标。
- 被动检测器不能作为真实性证明，实际应用必须表达不确定性并控制真实图片误报。
- 后续版本应加入 SID_Set 等第二外部域、阈值校准和最终提交推理接口。
