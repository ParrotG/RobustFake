# 项目说明：Transformation-Robust AIGC Recognizer

## 1. 目标与边界

本项目针对图片级 AIGC 检测，重点是在未见生成器以及真实传播变换下保持稳定。当前版本覆盖训练集分片获取、数据审计、双视图数据管道、模型、训练、验证、断点恢复和比赛要求的目录到 JSON 推理入口，并提供默认关闭的实验性 Residual Statistics Branch。

默认模型远低于题目要求的 2B 参数上限。CLIP 主干完全冻结，实际训练参数不超过约 5M，适合 8–12GB NVIDIA GPU。

## 2. 技术架构

### 2.1 基础模型

使用 `open_clip` 提供的 `ViT-B-16-quickgelu/openai` 视觉编码器，使激活函数与 OpenAI 原始权重的训练架构一致。文本编码器不会保留在训练模型中。视觉编码器始终处于 `eval` 状态且所有参数的 `requires_grad=False`，checkpoint 不重复保存冻结权重。

### 2.2 双视图检测头

每张图片构造两个共享 backbone 的视图：

1. 覆盖短边 90%–100% 的宽上下文方形 global view，避免固定 padding 暴露与标签相关的原始宽高比。
2. 从原图随机选择 50%–90% 范围的方形 local view。

默认在同一次 CLIP 前向中提取最终 projected embedding，以及第 4、7、10、12 个 transformer block 的归一化 CLS token。中间层经过可训练线性投影统一到 512 维，再由样本相关的 softmax gate 与最终层融合。融合后对 view 维度计算 mean 和 standard deviation，再进入 `LayerNorm → Linear → GELU → Dropout` 检测头。该结构保留最终层语义，同时允许中间层局部证据参与决策；将 `model.intermediate_layers` 设为空列表即可恢复旧的最终层模型。

训练额外包含 projection head，只用于监督式对比损失。

可选 Residual Statistics Branch 从每个归一化视图提取 24 维固定高通统计，包括方向残差矩、逐通道 Laplacian 统计和水平/垂直相邻像素差。两个视图的 mean/std 经小型 MLP 后与 CLIP 聚合特征拼接。该分支默认关闭，确保旧 checkpoint 兼容；其统计量可在现有 CLIP feature cache 上另建 sidecar，不需要重新计算 backbone embedding。

### 2.3 成对鲁棒训练

同一基础图片产生四个输入：clean global、clean local、degraded global、degraded local。clean 与 degraded 使用相同 crop geometry，因此一致性损失主要约束传播退化，而不是不同内容区域。

总损失为：

```text
L = 1.0 * mean(BCE_clean, BCE_transformed)
  + ramp(epoch) * 0.5 * SmoothL1(sigmoid(logit_transformed), stop_grad(sigmoid(logit_clean)))
  + 0.05 * supervised_contrastive_loss
```

一致性项在第一个 epoch 线性 ramp-up，并作用于有界概率而非无界 logit，避免分类 margin 增长时 clean/transformed logit 的绝对差持续主导总损失。

题目要求的所有增强均在线生成。默认 transformed 分支有 25% 保持 clean、50% 使用一个操作、25% 组合两个不同操作。还以较低权重加入 double-JPEG、WebP 和多种 resize 插值核。

## 3. 数据获取与合规

### 3.1 来源、固定版本与许可

默认混合池只包含四个固定来源：Shanmuk 配对集、WildFake 官方 train split、Community Forensics-Small、Tiny-GenImage。配置中的 revision 均为 40 位 commit SHA；正式准备会拒绝 `main/master`。Tiny-GenImage 固定为 `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`，是 GenImage 的第三方小型衍生集，按 CC BY-NC-SA 4.0 使用。Community Forensics-Small 固定为 `6c539a534c07917307c381f5af4053c6091b5278`。WildFake 的 ModelScope commit 和 train metadata SHA-256 也同时固定。

不同组成部分仍受其上游图片、ImageNet、生成器和非商业条款约束；统一 manifest 不改变或重新授权原数据。使用者必须自行确认非商业研究范围，在 Hugging Face 接受 gated 条款并执行 `hf auth login`。token 只从当前用户环境读取，不写入配置、日志或产物。

### 3.2 配额、分层和完整域留出

80k 指包含训练和两个验证角色的物理总数：

| 来源 | train real/fake | val_id real/fake | val_dg real/fake |
|---|---:|---:|---:|
| Shanmuk | 4k / 4k | 1k / 1k | 0 / 0 |
| WildFake train | 9k / 9k | 0 / 0 | 4k / 2k |
| Community Forensics-Small | 13k / 13k | 1k / 2k | 0 / 1k |
| Tiny-GenImage | 6k / 6k | 2k / 1k | 0 / 1k |

train、val_id、val_dg 各自都严格 50/50，总计 real/fake 40k/40k。来源配额是硬约束；桶内按可用量平方根分配，使大域仍获得更多样本、但增长幅度受限。fake 分生成器家族、架构和模型，real 分采集来源和内容域。技术属性仅做软约束与报告，不机械抹平网络图片的自然异质性。

`val_dg` 完整留出 WildFake 的 AFHQ/Church 真实域、DDPM/GALIP/MAGE，Tiny-GenImage 的 Wukong，以及稳定哈希选出的 Community Forensics Systematic 模型 ID。规范化别名在所有来源统一排除。Shanmuk 的 real/fake 父组始终共同进入同一个 role。

固定 Tiny-GenImage revision 的 ClassLabel schema 声明八个 fake 类别，但 35,000 条物理记录中 SD14 为 0 条，其余七个生成器各 2,500 条。程序只对白名单中的这一固定异常放行，并写入 `tiny_genimage_schema_anomaly` 审计；Wukong 仍取 1k 作为 DG，另外六个生成器共同提供 6k train 与 1k ID，数量尽可能接近。不会把 SD15 重标为 SD14，也不会静默伪造缺失类别。

### 3.3 全局去重、泄漏控制与恢复

外部 WildFake official、WildFake broad、SID-Set manifest 都是必需 denylist；缺少任意一项时准备失败。WildFake 混合来源只读固定 `total_split/train_metadata.csv`，不会读取 test metadata，并排除 DALL·E/COCO、已准备的 broad 路径及大体积 SD/Midjourney 分卷。

候选依次比较上游 provenance、原始 SHA-256、EXIF/RGB 后像素 SHA-256、256-bit pHash+dHash 和 crop-resistant hash。同标签重复从同一确定性后备池补齐；冲突标签的确认重复直接失败。跨 split 保留优先级为 external test、DG、ID、train。对象以原编码写入 `objects/<prefix>/<sha256>.<ext>`；已有本地来源优先硬链接复用。

部分外部评测 manifest 只保存内容 SHA。混合准备首次运行时会先显示 `Hash external denylist`，并行计算缺失的像素/感知哈希；结果每 500 张原子保存到项目缓存 `external_deny_index.json`。中断后只补未缓存记录，后续运行直接复用，不会再次静默扫描全部评测图片。

WildFake 通过 metadata-first 和并行 ZIP HTTP Range 获取；两个 Hugging Face 来源按 pinned Parquet 分片下载。每个来源独立原子记录已完成 shard/archive 及已落盘候选，统一层另写 `selection_state.json`。重跑跳过完成单元，常见网络错误指数退避，不清理全局 HF cache。默认网络上限为 80GiB，长期空间预期 25–45GiB；配额、域留出、revision、重复或泄漏任一验收失败都会留下 audit 并以非零状态退出。

### 3.4 标签无关标准化与偏置探针

在线预处理在 EXIF/RGB 规范化之后、global/local 几何和题目增强之前，以 0.75 概率执行温和标准化：resize round-trip、JPEG/WebP 重编码或两者组合。该函数不接收 label 或来源字段；clean/transformed 使用同一标准化基础图。train 随机采样，val/test/官方评估按 record ID 固定。

数据准备完成后，nuisance classifier 从颜色统计、亮度直方图、熵、饱和度、边缘、Laplacian sharpness、8×8 blockiness 和径向频谱能量中预测真假标签。它在 train 拟合 `HistGradientBoostingClassifier`，在 val_id/val_dg 输出 AUROC、AP、balanced accuracy、F1、混淆矩阵、逐来源/生成器结果和 permutation importance；原始规范图与确定性标准化图分别报告。来源、生成器、路径、prompt 和尺寸元数据不会成为分类器输入。该报告只做诊断，不阻断训练；高频可分性也可能是真实生成器指纹。

## 4. 集中配置

所有调试参数均在 `configs/default.yaml`：

- `project`：seed 和 run name。
- `data`：训练消费者读取的 manifest、audit 和输出路径。
- `mixed_data`：四来源固定 revision、硬配额、留出域、平方根分层、缓存/网络边界、全局去重和外部 denylist。
- `standardization`：标签无关 resize/codec 概率、范围和权重。
- `nuisance_audit`：低层探针的特征尺寸与分类器参数。
- `views`：输入尺寸、global/local crop 比例、透明图层合成底色和插值。
- `augmentations`：题目变换范围、操作数量概率和额外编码增强。
- `model`：backbone 身份、embedding/head/projection 维度和 dropout。
- `loss`：三类损失权重、一致性 ramp-up 和对比温度。
- `training`：epoch、micro-batch、梯度累积、DataLoader 预取、AMP、优化器和恢复路径。
- `output`：运行产物目录和 checkpoint 策略。
- `feature_cache`：缓存根目录、训练变体数、FP16/FP32、分片、batch 和预取设置。

CLI 只额外支持 `--set section.key=value`。未知 section、未知 key、错误概率和不支持的 backbone 会在下载或加载权重前报错。

## 5. 运行流程

```bash
uv sync --extra dev
uv run aigc-prepare-official-eval --config configs/default.yaml
uv run aigc-prepare-wildfake-eval --config configs/default.yaml
uv run aigc-prepare-sid-eval --config configs/default.yaml
hf auth login
uv run aigc-prepare --config configs/default.yaml
uv run aigc-cache-features --config configs/default.yaml
uv run aigc-train --config configs/default.yaml --set feature_cache.use_for_training=true
```

特征缓存按 manifest 内容、随机种子、backbone/权重、中间层、视图、标准化和增强配置生成 SHA-256 身份目录。train 的两个变体使用 `seed + variant + record_id` 确定性生成，两个 validation split 各缓存一个固定变体。每个分片先原子写入再更新 `cache_manifest.json`，中断后只继续未完成的连续后缀，不重复追加记录。缓存保存 clean/transformed、global/local 的最终层与中间层逐 view 特征，因此修改检测头、loss、学习率或 epoch 不会使缓存失效。默认四个中间层、FP16 和两个 train 变体约占 3.9GiB，另有少量元数据与序列化开销。

设置 `feature_cache.use_for_training=true` 后，训练只加载可训练融合层和检测头，不加载 CLIP，也不启动图像 DataLoader worker；每次读取 train record 时从两个缓存变体中随机选择一个。默认不启用该开关，仍可运行完整在线随机增强训练。

当前默认训练使用 micro-batch 64、梯度累积 1、12 个 DataLoader worker 和每 worker 1 个预取 batch，有效 batch size 为 64。clean/transformed 张量会合并成一次较大的 CLIP 前向，以减少小 kernel 和两次串行调用造成的 GPU 空隙；降低预取深度则限制 float32 视图队列的主机内存占用。由于视觉主干冻结且前向位于 `no_grad` 中，显存占用本来就会显著低于端到端微调；判断性能时应同时观察每秒样本数和 GPU compute utilization，而不是以占满显存为目标。内存受限时应优先降低 `training.num_workers` 和 `training.batch_size`。

DataLoader worker 默认不跨 epoch 常驻，以限制 PIL/编码库长期运行时的内存高水位。训练入口将 `SIGTERM` 和 `SIGHUP` 转换成可清理的中断，并在所有退出路径显式关闭 worker；Linux worker 还设置父进程死亡信号，以应对终端或 IDE 只强制终止主进程的情况。如确认运行环境稳定，可设置 `training.persistent_workers=true` 减少每轮 worker 启动开销。

8GB GPU 如果发生 OOM：

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set training.batch_size=32 \
  --set training.gradient_accumulation_steps=2
```

断点恢复：

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set training.resume_from=artifacts/runs/clip_b16_multilayer_v3/last.pt
```

checkpoint 包含检测头、optimizer、scheduler、AMP scaler、epoch、global step、随机状态、完整配置、数据 revision、backbone 身份和参数计数。恢复时会拒绝 backbone 或数据 revision 不一致的 checkpoint。

## 6. 验证指标与输出

每轮在固定 seed 的 `val_id` 和 `val_dg` 上分别计算 clean/transformed 的 AUROC、Average Precision、Balanced Accuracy 和 F1。最佳模型分数由 ID clean/transformed AUROC 均值占 50%、DG clean/transformed AUROC 均值占 50% 组成；验证集不做重加权。64k train 在准备阶段已经严格保持 real/fake 平衡，并在来源硬配额内部完成平方根域分配；训练 DataLoader 只做无放回 shuffle，确保每个已准备样本每轮恰好出现一次，不再重复施加带放回域平衡。

训练完成后可使用题目指定的 WildFake 展示子集做独立测试。该子集严格由元数据定义为 4,998 张 COCO val2017 真实图与 8,843 张 DALL·E Advanced 生成图，存放在 `data/evaluation/`，不会写入或复用训练 manifest：

```bash
uv run aigc-prepare-official-eval --config configs/default.yaml
uv run aigc-evaluate-official --config configs/default.yaml
```

正式外部评测前，可在内部 `val_id`/`val_dg` 的 clean/transformed 预测上拟合一次
checkpoint-bound affine Platt 校准。存在兼容训练特征缓存时只运行检测头，不重新计算
CLIP；输出默认写在 checkpoint 同目录，外部评测和目录推理会校验 checkpoint SHA-256
后自动应用：

```bash
uv run aigc-calibrate \
  --config artifacts/runs/clip_b16_multilayer_v3/resolved_config.yaml \
  --checkpoint artifacts/runs/clip_b16_multilayer_v3/best.pt
```

WildFake 仓库整体约 1.2TB，而 DALL·E ZIP 约 25.6GB。准备命令校验上游 archive SHA-256，并通过 ZIP HTTP Range 只提取题目指定成员，实际选择图片约 2.93GB；下载中断后根据原子 manifest 继续。若上游 archive 身份或官方元数据数量不再是 real 4,998/fake 8,843，命令会拒绝评测。

### 统一外部评测管线

官方 WildFake、广域 WildFake 和 SID_Set 的数据准备相互隔离，但最后都生成同一字段约定的只读 manifest。三个评测命令共用一套 checkpoint 校验、确定性标准化、global/local 视图、推理、指标和预测落盘实现。因此数据源适配不会复制或悄悄改变模型预处理。

广域 WildFake 默认从官方 `total_split/test_metadata.csv` 选择 3,000 real + 3,000 fake。fake 先在 GAN、diffusion、other 三个家族间等额，再在支持直接 Range 选择性读取的架构间等额；real 在 AFHQ、CelebA-HQ、Church、FFHQ、ImageNet、LAION-5B 间等额。DALL·E/COCO 因已存在于题目官方子集而排除，SD/Midjourney 因其数十 GB 多分卷归档会显著放大少量抽取成本而不进入默认样本。采样固定 seed，只使用官方 test split：

```bash
uv run aigc-prepare-wildfake-eval --config configs/default.yaml
uv run aigc-evaluate-wildfake --config configs/default.yaml
```

WildFake 准备器以 archive 为并发单元，同时读取多个 ZIP；每个 ZIP 内按成员 offset 排序，减少远程跳读。原子 state/manifest 记录每一张已完成图片，中断后只补缺失成员。

上游 GigaGAN 归档包含多张扩展名为 PNG、实际为零填充内容的损坏 member。准备器在正式抽样前根据异常低的 ZIP 压缩比定位候选，只对这些极小压缩内容执行 Range 读取和解码确认；当前固定归档共确认 89 张损坏图片。结果按 archive SHA-256 缓存在 `archive_integrity.json`，后续运行无需重复扫描。全部坏图在抽样前排除，并在同一 GigaGAN 分层内稳定补入候选样本，因此不删除 GAN 家族且维持精确配额。未完成 checkpoint 在排除项变化且既有记录仍是新样本集合子集时可以安全迁移，不需要删除已下载图片。

SID_Set 固定 revision `dc03ead57929879319ce30a82bfcfb8d317b10bd`，扫描官方 validation 的 34 个 Parquet shard，排除 tampered(label 2)，保留 2,000 real + 2,000 full_synthetic。每类配额近似均匀分布到所有 shard，再以 `seed + shard + img_id` 的稳定哈希选择。默认三个 shard 有界预取，使下载与本地扫描重叠；总扫描流量约 16.8GB，长期只保留入选图片：

```bash
uv run aigc-prepare-sid-eval --config configs/default.yaml
uv run aigc-evaluate-sid --config configs/default.yaml
```

评测矩阵包括 clean、JPEG quality 90/70/50/30、Gaussian blur sigma 0.5/1.0/2.0、resize 0.5/0.25、Gaussian noise sigma 0.02/0.05/0.10、color jitter ±20% 和 center crop 80%。此外默认启用六条有序复合流水线：resize→JPEG、JPEG→resize→JPEG、crop→resize→JPEG、blur→resize→JPEG、color→noise→JPEG，以及较强的 crop→blur→resize→JPEG。场景顺序固定且由 record ID 决定随机量；报告分别汇总 single/composed 的均值和最差 AUROC。可用 `evaluation.enable_composed_scenarios=false` 关闭复合场景。

外部评测按数据集、manifest、backbone、预处理、场景和抽样记录身份缓存冻结的
final/intermediate/residual features；同架构的新 checkpoint 可直接复用。首次全量评测
仍需计算所有 CLIP 特征。迭代阶段可为三个外部评测命令添加 `--fast`，固定抽取
2,000 张类别平衡样本，只运行 clean、各变换族最强单项和 stress composition，并写入
独立的 `results.fast.json`/`predictions.fast.jsonl`，不覆盖正式结果。

每个场景输出 AUROC、AP、balanced accuracy、F1、真假类别 recall 与混淆计数，并按 `source_name` 输出样本数、平均 fake 概率、预测 fake 比例和组内准确率；同时保留逐图片置信度，便于定位某个生成器或真实来源的系统性失败。公共模型路径、batch size、worker 数和场景列表均集中在 `evaluation` 配置段。

`metrics.jsonl` 的每条记录包含 schema version、UTC timestamp、session ID、epoch 和 global step，允许恢复训练后继续形成明确的时间序列。

## 7. 复现与已知限制

- 数据 revision、抽样 seed、生成器划分、验证增强 seed 都会固定。
- 在线训练增强会随 epoch 变化；checkpoint 保存完整 RNG 状态用于恢复。
- 混合来源覆盖面已扩大，但仍是人为整理的数据分布，不能视为真实网络流量的无偏样本。
- nuisance classifier 只能证明低层统计可分，不能自动判断信号是采集偏置还是有意义的生成器指纹。
- COCO val2017 泄漏审计依赖感知哈希，对极端裁剪或重绘版本不能提供密码学意义的无重叠证明。
- 被动检测器不能作为真实性证明，实际应用必须表达不确定性并控制真实图片误报。
- 后续版本仍需加入阈值校准和生成器留一实验，并通过受控消融决定是否启用 Residual Statistics Branch。
