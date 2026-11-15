# 0.9470 版本结果诊断

诊断日期：2026-10-10  
数据来源：用户提供的 5 个 CSV。未收到 ZIP、checkpoint、main-only report、fold metrics、expert search、corrected/harmed CSV。  
本阶段未训练模型、未修改预测、未根据 leaderboard 推测 test 标签。

## 1. 文件完整性

| 文件 | 结果 |
|---|---|
| submission | 378×2，ID 唯一、Category 无缺失 |
| OOF predictions | 500×21，500 个唯一文件，5 folds 各 100 张 |
| test probabilities | 378×6，ID 与 submission 完全一致 |
| classification report | 8×5，与 OOF 独立重算结果一致 |
| confusion matrix | 5×5，与 OOF 独立重算结果一致 |

所有 main/final/expert/test 概率行和均仅有 CSV 浮点舍入误差；submission 的类别与 test probability argmax 逐行一致。

## 2. Main 与最终 OOF

| 指标 | Main-only | Final |
|---|---:|---:|
| Macro-F1 | 0.925738 | 0.925738 |
| Accuracy | 0.926000 | 0.926000 |
| 错误数 | 37 | 37 |

`main_prob_*` 与 `final_prob_*` 在全部 500 行逐元素相同；`main_pred` 与 `expert_final_pred` 也全部相同。因此：

- Expert 最终提升：**0**。
- Corrected：**0**。
- Harmed：**0**。
- Net gain：**0**。
- 最终 submission 实际为纯 main ConvNeXt-Tiny 输出；旧安全回退正确生效。

## 3. 每类结果

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Black-grass | 0.831683 | 0.840000 | 0.835821 | 100 |
| Common wheat | 0.941176 | 0.960000 | 0.950495 | 100 |
| Loose Silky-bent | 0.885417 | 0.850000 | 0.867347 | 100 |
| Scentless Mayweed | 0.990099 | 1.000000 | 0.995025 | 100 |
| Sugar beet | 0.980000 | 0.980000 | 0.980000 | 100 |

最差类别是 **Black-grass（F1=0.835821）**，其次是 **Loose Silky-bent（F1=0.867347）**。

## 4. Confusion matrix 与主要瓶颈

| True \ Pred | Black | Common | Loose | Mayweed | Sugar |
|---|---:|---:|---:|---:|---:|
| Black-grass | 84 | 5 | 11 | 0 | 0 |
| Common wheat | 2 | 96 | 0 | 0 | 2 |
| Loose Silky-bent | 15 | 0 | 85 | 0 | 0 |
| Scentless Mayweed | 0 | 0 | 0 | 100 | 0 |
| Sugar beet | 0 | 1 | 0 | 1 | 98 |

- Black-grass → Loose Silky-bent：**11**。
- Loose Silky-bent → Black-grass：**15**。
- 双向 grass confusion：**26/37 个错误，占 70.3%**。
- 第二大方向是 Black-grass → Common wheat：5。

结论：属于 **Case A（grass pair 仍占绝大多数错误）**，同时部分符合 **Case C（存在高置信错判）**；不属于以低 margin 为主的 Case D。

## 5. Fold 稳定性

| Fold | Macro-F1 | Accuracy | Black F1 | Silky F1 | B→S | S→B | 错误 | 置信度≥0.8 的错误 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.900486 | 0.90 | 0.837209 | 0.871795 | 2 | 3 | 10 | 3 |
| 1 | 0.929278 | 0.93 | 0.863636 | 0.857143 | 0 | 5 | 7 | 2 |
| 2 | 0.939153 | 0.94 | 0.842105 | 0.878049 | 3 | 2 | 6 | 2 |
| 3 | 0.939850 | 0.94 | 0.857143 | 0.842105 | 2 | 4 | 6 | 2 |
| 4 | 0.915732 | 0.92 | 0.764706 | 0.883721 | 4 | 1 | 8 | 5 |

- Fold mean Macro-F1：0.924900。
- Population std：0.015006；sample std：0.016777。
- 最差 fold：fold 0（0.900486）。
- Fold 4 的整体分数不是最差，但 Black F1 最差，且高置信错误最多，因此更适合作为“草类形态增强”筛选 fold。
- 波动存在但不属于单个 fold 崩溃；fold 0 到 fold 3 的差距约 0.0394，在每折仅 100 张的条件下仍需谨慎解释单折增益。

## 6. 错误置信度与 Top-2

37 个错误中：

- confidence ≥0.70：23。
- confidence ≥0.80：14。
- confidence ≥0.90：8。
- confidence ≥0.95：3。
- margin ≤0.10：3。
- margin ≤0.20：7。
- margin ≤0.30：10。
- 真实类别仍在 top-2：30。
- 真实类别不在 top-2：7。

这说明大部分错误不是简单的低 margin 犹豫。虽然 81.1% 的错误仍把真实类排在 top-2，但其中许多对错误 grass 类给出了很高概率，单纯调 threshold 或全局 temperature 不太可能修复 argmax。

最高置信错误包括：

| 文件 | True | Pred | Confidence | Margin |
|---|---|---|---:|---:|
| vbnsuditsj.png | Black | Loose | 0.982325 | 0.975709 |
| sqbddfgwfg.png | Black | Loose | 0.972283 | 0.962659 |
| dedzklxlrh.png | Black | Loose | 0.971691 | 0.961820 |
| timqrqkloy.png | Loose | Black | 0.937058 | 0.915797 |
| tpbvqcrocu.png | Black | Loose | 0.936601 | 0.892931 |

这些样本更支持检查训练时的形态破坏、裁剪与分辨率，而不是继续细调融合 threshold。

## 7. Expert 独立能力与 disagreement

在 200 张真实 grass 样本上：

- Expert binary accuracy：**0.805**。
- Main 的 Black/Loose 条件判断 accuracy：**0.865**。
- Main/expert disagreement：22。
- Disagreement 中 expert 独对：5。
- Disagreement 中 main 独对：17。

按真实类别：

- Black disagreements 8：main 赢 5，expert 赢 3。
- Loose disagreements 14：main 赢 12，expert 赢 2。

按 fold，expert 每折都只赢 1 个 disagreement，而 main 分别赢 5、3、2、3、4 个。该模式跨 fold 一致，说明当前 expert 不是一个与 main 互补的强分类器，而是系统性更弱；尤其容易破坏 Loose Silky-bent。

## 8. Outer-fold simple-rule evaluation

为修正旧协议，使用三个预注册规则进行零训练 outer-fold selection：

- A：main top1 为 grass，soft alpha=0.50。
- B：main top2 同时为 Black/Loose，soft alpha=0.50。
- C：main top1 为 grass，conditional soft alpha=0.75。

对每个 held-out fold，仅使用另外四折选择规则。五个 fold 均选择 C；拼接后的无偏结果：

| 指标 | Main | Outer-selected expert | Delta |
|---|---:|---:|---:|
| Macro-F1 | 0.925738 | 0.921755 | -0.003982 |
| Accuracy | 0.926000 | 0.922000 | -0.004000 |
| Corrected | — | 0 | — |
| Harmed | — | 2 | — |

受损样本：

- `zmtuxgsgyz.png`（fold 0，Loose → Black）。
- `czqwrgnyqs.png`（fold 3，Loose → Black）。

结论：**REJECT expert**。不应继续调 threshold，也不应把 expert 320→384 作为第一训练实验。

## 9. Calibration

- 10-bin ECE：0.0411。
- Multiclass Brier：0.1307。
- NLL：0.2678。
- 0.9–1.0 confidence bin：平均 confidence 0.9605，实际 accuracy 0.9778。

整体校准不算严重失真；最高置信区间甚至略偏保守。但存在 8 个 confidence≥0.9 的错误尾部。全局 temperature scaling 不改变 argmax，不能直接提升 Macro-F1，因此不是首选实验。

## 10. Test prediction distribution

| 类别 | Train true | Final OOF pred | Test pred |
|---|---:|---:|---:|
| Black-grass | 20.0% | 20.2% | 16.7%（63） |
| Common wheat | 20.0% | 20.4% | 8.7%（33） |
| Loose Silky-bent | 20.0% | 19.2% | 28.8%（109） |
| Scentless Mayweed | 20.0% | 20.2% | 23.8%（90） |
| Sugar beet | 20.0% | 20.0% | 22.0%（83） |

Test 与均衡 train 的预测分布差异明显，尤其 Common wheat 偏少、Loose 偏多。但 test 真实先验未知，禁止据此做 prior correction、手工改标签或调 leaderboard。这里只将其记录为 distribution-shift 监控项。

## 11. 错误图片尺寸

最终 37 个错误的中位尺寸为 283×283，宽高比中位数 1.0；没有显示“错误主要来自非正方形图片”。强制方形 Resize 不是当前最有证据的首个改动。

## 最终诊断

当前真正瓶颈是高置信的 Black-grass / Loose Silky-bent 形态混淆。现有 expert 明显弱于 main 且没有互补净收益。最有依据的下一个训练实验不是继续调 expert，而是只移除主模型 RandomErasing，在 fold 4 上检查细叶结构是否因遮挡增强而受损。

