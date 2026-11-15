# 任务四：血管分割最高分路线

完整来源为报告已核实最高分源码包。Public 0.86114、Private 0.86278；artifacts保存源码清单、权重清单、环境及提交CSV。

基础code/train.py、predict.py、common.py；optimization_v2至v6依次包含高分辨率、直接512监督、U-Net++、Context及soft-clDice代码。最终入口见optimization_v6/finish.py与ensemble.py、frozen_manifest.json。最终七族×六权重共42检查点，D4推理。绿色通道和未选多尺度候选仍保留历史配置，最终以冻结清单为准。

原run.sh中/root/autodl-tmp路径是历史路径，需迁移；不要直接启动所有脚本。需另配原始课程图像、预训练资产和清单内42权重，未附训练/测试标签。原LICENSE.clDice.txt保留。报告评分摘要未复制隐藏标准答案。
