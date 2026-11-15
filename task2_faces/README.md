# 任务二：表情分类

artifacts/submission_R2_75_25.csv对应历史最高Public 0.725的文件名，缺少生成配置与概率的完整绑定。code/fer_strong_ensemble.py从已有Notebook代码单元提取，是旧三模型，不含R2。fer_final_historical_b2.py是更早B2方案。

FINAL_SUMMARY.json与selected概率对应Push80：固定Fold0验证71.4286%，Public历史0.721；不能将0.725归给该代码。原始课程图像及folds-v3数据需另行准备，Kaggle输入说明保留在代码开头。完整Push80 Small训练入口仍未恢复，保存的概率可以复核验证集结果，不能等同从头训练复现。
