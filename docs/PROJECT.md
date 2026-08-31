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

当前使用 gated 数据集 `Shanmuk4622/ai-image-detection-dataset`，固定 revision 为 `8f1f536676f96cbc58bffd520ed50d1e7b9e894a`。数据集将 5,000 张 COCO 与 5,000 张 ImageNet 真实图统一处理为 512×512 RGB PNG，并为每张真实图使用同一 BLIP-2 caption 生成六张伙伴图。数据集继承 ImageNet 的非商业研究限制以及各生成器许可；使用者必须在 Hugging Face 页面接受条款并执行 `hf auth login`。

准备程序显式读取当前用户 token，并用于 revision 查询、metadata 和所有图片分片请求。token 不会出现在日志、配置或项目产物中。缺少权限、revision 漂移或 schema 不匹配都会在下载大文件前失败。

### 3.2 默认规模与抽样

先验证每个 `source_real_id` 恰好包含一张 real 和六张 fake，且整组属于同一官方 split。然后在每个 split 内固定 seed 打乱父 ID，按随机化的六生成器轮转选择一张 fake，因此生成器计数最多相差一张：

| split | real | fake |
|---|---:|---:|
| train | 7,056 | 7,056 |
| val | 1,446 | 1,446 |
| test | 1,498 | 1,498 |

test 保留在 manifest 中，但 `aigc-train` 只读取 train/val。每张入选图都校验 PNG 编码、512×512 尺寸、pipeline version、上游 SHA-256 和安全解码。精确重复或相同高位 pHash 会按父组整体排除。

由于真实来源含 COCO，而题目禁止训练 COCO val2017，训练准备依赖已经隔离准备好的 WildFake 官方真实子集。程序使用 256-bit pHash 与 dHash 组合比较 COCO real，确认匹配后同时排除 real 与其 fake 伙伴。官方 manifest 或图片缺失时拒绝完成训练数据准备，不以未经审计的数据继续。

### 3.3 空间保护与失败语义

远端图片分片约 24.5GB，最终 20k 图片通常约 7GB。程序先下载约 3MB metadata，确定完整的 20k 目标 ID，再扫描图片分片。默认同时预取两个完整分片，并用 `max_download_gb` 与 `max_shard_cache_gb` 在下载前检查预算。

`preparation_state.json`、manifest 和 audit 均原子写入，状态包含 revision、配置指纹、完整目标 ID、已处理 shard 和已提取 ID。常见网络错误指数退避。重跑从首个未完成 shard 继续，不依赖可能因重试改变的全局行号。已处理 shard 只从项目专用 payload cache 删除，不触碰用户全局 HF cache。异常和常规终止信号都会保存最近状态；训练入口只接受 `complete: true` 的 audit。

### 3.4 标签无关标准化与偏置探针

在线预处理在 EXIF/RGB 规范化之后、global/local 几何和题目增强之前，以 0.75 概率执行温和标准化：resize round-trip、JPEG/WebP 重编码或两者组合。该函数不接收 label 或来源字段；clean/transformed 使用同一标准化基础图。train 随机采样，val/test/官方评估按 record ID 固定。

数据准备完成后，nuisance classifier 从颜色统计、亮度直方图、熵、饱和度、边缘、Laplacian sharpness、8×8 blockiness 和径向频谱能量中预测真假标签。它在 train 拟合 `HistGradientBoostingClassifier`，在 val/test 输出 AUROC、AP、balanced accuracy、F1、混淆矩阵、每生成器结果和 permutation importance；原始规范图与确定性标准化图分别报告。来源、生成器、路径、prompt 和原始尺寸不会成为分类器输入。该报告只做诊断，不阻断训练；高频可分性也可能是真实生成器指纹。

## 4. 集中配置

所有调试参数均在 `configs/default.yaml`：

- `project`：seed 和 run name。
- `data`：固定 revision、生成器、配对规模、缓存/网络边界、路径、去重和官方泄漏审计。
- `standardization`：标签无关 resize/codec 概率、范围和权重。
- `nuisance_audit`：低层探针的特征尺寸与分类器参数。
- `views`：输入尺寸、local crop 比例、padding 和插值。
- `augmentations`：题目变换范围、操作数量概率和额外编码增强。
- `model`：backbone 身份、embedding/head/projection 维度和 dropout。
- `loss`：三类损失权重与温度。
- `training`：epoch、micro-batch、梯度累积、DataLoader 预取、AMP、优化器和恢复路径。
- `output`：运行产物目录和 checkpoint 策略。
- `watermark`：可见水印 OCR 开关、Tesseract 路径、四角区域比例、缩放和超时。

CLI 只额外支持 `--set section.key=value`。未知 section、未知 key、错误概率和不支持的 backbone 会在下载或加载权重前报错。

## 5. 运行流程

```bash
uv sync --extra dev
uv run aigc-prepare-official-eval --config configs/default.yaml
hf auth login
uv run aigc-prepare --config configs/default.yaml
uv run aigc-train --config configs/default.yaml
```

当前默认训练使用 micro-batch 32、梯度累积 1 和 8 个 DataLoader worker，有效 batch size 为 32。clean/transformed 张量会合并成一次较大的 CLIP 前向，以减少小 kernel 和两次串行调用造成的 GPU 空隙。由于视觉主干冻结且前向位于 `no_grad` 中，显存占用本来就会显著低于端到端微调；判断性能时应同时观察每秒样本数和 GPU compute utilization，而不是以占满显存为目标。内存受限时应优先降低 `training.num_workers` 和 `training.batch_size`。

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

`aigc-provenance` 会在四角区域尝试识别豆包/Seedream、即梦、通义万相、腾讯混元及主流海外生成器的中英文水印；结果写入每条记录的 `watermark`，并在没有更高等级 C2PA 证据时以中等置信度参与 `authenticity_summary`。OCR 依赖外部 Tesseract，缺少该程序时只报告降级状态，不会使 provenance 检查失败。也可使用 `aigc-watermark` 只输出水印报告。可见水印不是签名，可能被裁剪、复制或后加，漏检也不能证明图片真实。

## 7. 复现与已知限制

- 数据 revision、抽样 seed、生成器划分、验证增强 seed 都会固定。
- 在线训练增强会随 epoch 变化；checkpoint 保存完整 RNG 状态用于恢复。
- 当前六种 fake 均为 text-to-image，不覆盖图生图、局部编辑或题目官方 DALL·E 域。
- nuisance classifier 只能证明低层统计可分，不能自动判断信号是采集偏置还是有意义的生成器指纹。
- COCO val2017 泄漏审计依赖感知哈希，对极端裁剪或重绘版本不能提供密码学意义的无重叠证明。
- 被动检测器不能作为真实性证明，实际应用必须表达不确定性并控制真实图片误报。
- 后续版本应加入 SID_Set 等第二外部域、阈值校准和最终提交推理接口。
