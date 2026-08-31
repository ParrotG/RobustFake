# RobustFake 消融实验管线

## 1. 实验定义

主消融采用 leave-one-component-out：以发布模型为唯一 full baseline，每个实验只移除一个方法，其他数据 manifest、seed、缓存变体、训练轮数、优化器、checkpoint 选择和校准协议全部保持一致。这样得到的差值才主要归因于被移除的方法。

消融不等于把历史实验直接排列在一起。若两个 run 同时改变了数据、loss 和模型结构，它们只能作为开发历程，不能作为组件有效性的严格证据。交互消融和数据规模实验可作为附录，但不进入时间受限的核心矩阵。

## 2. 核心矩阵

| 实验 | 唯一变化 | 是否重算 CLIP cache | 故事作用 |
|---|---|---:|---|
| Full RobustFake | 无 | 否 | 最终基线 |
| Without residual statistics | `model.residual_statistics_enabled=false` | 否 | 验证固定取证统计是否补充 CLIP |
| Without multi-layer fusion | `model.multilayer_fusion_enabled=false` | 否 | 验证中间层信息是否优于只用最终 embedding |
| Without consistency | `loss.consistency_weight=0.0` | 否 | 验证 clean/degraded 不变性约束 |
| Without contrastive loss | `loss.contrastive_weight=0.0` | 否 | 验证类结构约束的边际贡献 |
| Without calibration | `evaluation.calibration_enabled=false` | 不训练 | 展示排序性能与操作阈值是两个问题 |

训练 cache 始终保留最终层、中间层和 residual sidecar。关闭 residual 或 multi-layer fusion 只让检测头忽略相应字段，不改变 cache identity。外部评测 cache 也只绑定冻结特征生产配置，并始终保存 residual statistics，因此上述 checkpoint 可直接复用现有官方场景缓存。

不把以下项目放进核心矩阵：

- 移除混合数据集：需要重新构造 manifest，且同时改变来源、生成器与样本量，成本和混杂都过高。
- 移除全部 augmentation：现有缓存已经固化 clean/transformed 对；严格实验需要重算训练 cache。
- 同时移除多个组件：无法把下降归因于一个方法。
- 修改 backbone：这属于模型比较，不是最终架构的单组件消融。

## 3. 两阶段运行

### 3.1 第一阶段：固定 seed 的低成本筛选

所有训练使用 `seed=2026`、现有 64k cached features、8 epoch 和相同 early stopping。Full 直接复用已完成的发布 checkpoint，其余四个训练消融分别运行：

```bash
uv run aigc-train \
  --config configs/default.yaml \
  --set project.run_name=ablation_without_residual \
  --set model.residual_statistics_enabled=false \
  --set feature_cache.use_for_training=true

uv run aigc-train \
  --config configs/default.yaml \
  --set project.run_name=ablation_without_multilayer \
  --set model.multilayer_fusion_enabled=false \
  --set feature_cache.use_for_training=true

uv run aigc-train \
  --config configs/default.yaml \
  --set project.run_name=ablation_without_consistency \
  --set loss.consistency_weight=0.0 \
  --set feature_cache.use_for_training=true

uv run aigc-train \
  --config configs/default.yaml \
  --set project.run_name=ablation_without_contrastive \
  --set loss.contrastive_weight=0.0 \
  --set feature_cache.use_for_training=true
```

每个训练 run 都使用自身内部验证集拟合相同的 constrained-minimax calibration。不得在 WildFake official 上拟合阈值：

```bash
uv run aigc-calibrate \
  --config artifacts/runs/ablation_without_residual/resolved_config.yaml \
  --checkpoint artifacts/runs/ablation_without_residual/best.pt
```

对另外三个 run 替换目录名重复该命令。

随后先执行 `--fast`。每个实验写入独立目录，避免覆盖最终结果：

```bash
uv run aigc-evaluate-official \
  --config artifacts/runs/ablation_without_residual/resolved_config.yaml \
  --checkpoint artifacts/runs/ablation_without_residual/best.pt \
  --set official_evaluation.results_path=artifacts/ablations/without_residual/results.json \
  --set official_evaluation.predictions_path=artifacts/ablations/without_residual/predictions.jsonl \
  --fast
```

Without calibration 不重新训练；直接用 full checkpoint 禁用 calibration，并写入独立结果：

```bash
uv run aigc-evaluate-official \
  --config artifacts/runs/clip_b16_multilayer_residual_consistency_v5/resolved_config.yaml \
  --checkpoint artifacts/runs/clip_b16_multilayer_residual_consistency_v5/best.pt \
  --set evaluation.calibration_enabled=false \
  --set official_evaluation.results_path=artifacts/ablations/without_calibration/results.json \
  --set official_evaluation.predictions_path=artifacts/ablations/without_calibration/predictions.jsonl \
  --fast
```

### 3.2 第二阶段：只扩展有叙事价值的实验

第一阶段不能用 official 结果重新选择 checkpoint。它只决定哪些消融值得花费展示成本：

- 若移除组件使内部 ID/DG monitor AUROC、fast worst-transform AUROC 或 real recall 明显下降，则运行完整官方矩阵。
- 若变化小于约 0.002 AUROC，报告为“在本次单 seed 实验中贡献不明显”，不要把噪声包装成提升。
- Full、Without residual、Without consistency 建议无论筛选结果如何都进入最终表，因为它们对应最清晰的语义/取证/鲁棒性故事。
- 若时间允许，对影响最大的一个消融补两个 seed；不必对所有组合做三 seed 网格。

完整评测只需去掉 `--fast`。冻结外部特征已缓存时，每个新 checkpoint 只运行小型检测头。

## 4. 公平比较口径

主表同时报告：

| 维度 | 指标 | 解释 |
|---|---|---|
| 内部验证 | ID clean/transformed AUROC | 熟悉来源上的能力与 robustness gap |
| 内部验证 | DG clean/transformed AUROC | 未见域/生成器泛化 |
| 官方排序 | clean、mean single、worst single AUROC | 题目要求的主要鲁棒性证据 |
| 阈值行为 | Balanced Accuracy、real/fake recall | 揭示 fake 偏置与 false positives |
| 压力测试 | mean/worst composed AUROC | 额外边界，不与规定单变换混合 |
| 成本 | trainable parameters、训练时间 | 支撑 feasibility |

结构/loss 消融各自拟合相同协议的内部 calibration，确保比较的是完整可部署方案；Without calibration 单独展示同一 final checkpoint 的 raw operating point。AUROC 是结构和训练方法的主证据，校准贡献则以 Balanced Accuracy、real recall、fake recall 和 ECE/Brier（若后续加入）呈现。

## 5. 自动汇总与可视化

评测后用以下命令生成 `summary.json`、`summary.csv` 和可直接放入 PPT 的 `auroc_comparison.svg`。第一个 `--result` 必须是 Full，后续顺序保持与汇报图例一致：

```bash
uv run robustfake-ablation-report \
  --result Full=artifacts/evaluations/wildfake_official/results.fast.json \
  --result NoResidual=artifacts/ablations/without_residual/results.fast.json \
  --result NoMultilayer=artifacts/ablations/without_multilayer/results.fast.json \
  --result NoConsistency=artifacts/ablations/without_consistency/results.fast.json \
  --result NoContrastive=artifacts/ablations/without_contrastive/results.fast.json \
  --result NoCalibration=artifacts/ablations/without_calibration/results.fast.json \
  --output-dir artifacts/ablations/report
```

建议最终制作三张图：

1. `Δ mean/worst single-transform AUROC` 横向条形图：说明每个组件对鲁棒排序的贡献。
2. `clean AUROC` 对 `worst-transform AUROC` 的二维散点图：右上角越好，展示 clean/robustness trade-off。
3. Full 与 Without calibration 的 real/fake recall 哑铃图：单独讲阈值偏置，避免把 calibration 与表征学习混为一谈。

图中应显示绝对值与相对 Full 的差值，不只展示截断坐标的柱长。标题明确标注 `single seed` 和 `fast diagnostic` 或 `full official matrix`，避免把筛选实验表述为统计显著结论。
