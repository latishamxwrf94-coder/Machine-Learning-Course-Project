# 任务三：中央凹定位

Public最优：imagenet384_half_exploratory_v1，提交57629241、版本362122977；0.5三种子U-Net + 0.25两组ResNet18各一组。code/infer_imagenet_exploratory.py是最终融合推理，依赖前序输出；train_seed192_ensemble.py、train_seed192_three_seed.py及train_transfer384.py保留历史训练流程。configs保留官方Notebook输入和内嵌代码，复原第二组ResNet请使用对应配置的完整kernel.text，而非自行改训练种子。

train_sigma027.py对应Public 12.57089 / Private 20.70283，是赛后Private最低已提交候选；未将它追认为开发阶段已选最优。只使用比赛标签，公开ImageNet预训练。Python/torch/sklearn需支持代码中的GroupKFold(shuffle=True)。脚本依赖Kaggle挂载路径及前序权重；按原配置恢复输入后运行，不会由本归档自动提交。
