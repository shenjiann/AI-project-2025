# Project of AI Course

course link: [https://econlab.xmu.edu.cn/datascience/AI_and_ML/index.html]

要求：

1. 总共4个章节，全连接神经网络、模型分析、卷积神经网络、序列模型，每个章节寻找经济数据进行分析，fcnn和模型分析使用同一个主题；
2. 直接调pytorch，英文，注意代码规范（函数和变量的命名、comment每个函数变量的介绍）
3. Fcnn：替换mnist例子，找一个二分类问题，包括描述统计、nn建模和估计，其他模型（如logistic）

## 全连接神经网络：税收不遵从分类
<!-- 可以参考的主题：税收行为识别，信用评估 -->

**数据来源**：锐思金融研究数据库 [http://db.resset.com]，2015 - 2024上市公司数据

- Y: 公司重大事项 — 重大事项违规处罚 - 2015.01.01至2024.12.31，涉及内容799其他，报表类型Q4年报
- X: 财务指标 - 财务指标 - 财务比率，2015.01.01至2024.12.31，报表类型Q4年报；
  股东与股本 - 股东户数，2015.01.01至2024.12.31，年末
  股东与股本 - 主要股东名单与股权结构，2015.01.01至2024.12.31，年末，股东排名<=1

**变量选择**
财务指标：企业常常会采取隐藏收入、虚增成本费用、虚构原始凭证等手段以减少税源或推迟纳税，能够揭示该类行为的指标包括：

- 每股指标：每股收益、每股经营活动现金流量、每股资本公积金
- 盈利能力：净利润、营业利润/营业收入
- 偿债能力：有形净值债务率、股东权益/负债合计、产权比率、资产负债率
- 成长能力：利润总额增长率、营业收入增长率、营业收入3年增长率
- 营运能力：股东权益周转率、
- 现金流量：经营现金净流量
- 分红能力：每股现金及现金等价物余额
- 资本结构：权益乘数
  
股权指标：不同股权性质和结构对上市企业税收筹划的非税成本具有不同影响，可能导致控股股东以及高层管理者产生不同的税收行为。包括以下方面：

- 股东性质：最大股东是否国有（股东类别）
- 股东规模：股东总户数、户均持股数

**结果整理**
数据天然存在着严重的类别不平衡问题，需要

1. smote上采样0.1+随机下采样：auprc = 0.16，Precision = 0.16， Recall = 0.5左右
2. smote上采样0.1: auprc = 0.25, Precision: 0.3472 | Recall: 0.3634
3. smote上采样0.2: auprc = 0.22
4. smote上采样0.2 lr=1e-3 epoch=500: auprc = 0.275, Precision: 0.3694 | Recall: 0.3892

## 模型分析

包含内容：

1. variance-bias
2. regularization
   1. l2: 基本实现，封装好的调节lambda的函数，lambda变化时train valid test loss的变化，
   2. dropout：基本实现，封装好的调节各层p的函数
   3. early stop：EarlyStop类的实现参考[https://github.com/Bjarten/early-stopping-pytorch]，MNIST示例[https://github.com/Bjarten/early-stopping-pytorch/blob/main/MNIST_Early_Stopping_example.ipynb]
3. batch normalization：基本实现，bn中各层的momentum有一些参数可能调

调参结论：
1. [64, 32, 16]的结构比[32, 8]的要好
2. batchnorm 普遍都好
3. early stop false比patience=50好


## 卷积神经网络

可选择的主题：

- 夜间灯光
  - [Combining satellite imagery and machine learning to predict poverty(Jean et al. 2016 Science)](https://www.science.org/doi/10.1126/science.aaf7894) 数据不好找

- [Measuring investor sentiment by photos from news](https://www.kuntara.net/uploads/1/1/4/9/114945401/1-s2.0-s0304405x21002683-main-3.pdf) 关于中国的数据不好找

- 时间序列转为图像
  - [基于CNN-DAE-SeqConvLSTM混合模型和多种时序成像算法的标普500指数波动率预测研究](https://kns.cnki.net/kcms2/article/abstract?v=N2LrlypoGYVdV8yHa2x7fL-mShOO2shXiF-MNSUemyzp6-B-0AmOAu31yhgVWnZom5gMg2lEHrULcDAl8CoskNsYy60iVotU_QLlRGe0ddfxKgttg0iauSv2Ok0uZcNO7usppg1aUFMsZPeZd39tgWtlfh6BgmCRLjo4fWbSf1g9A90n3u519w==&uniplatform=NZKPT&language=CHS)
  - [基于时序图像编码与遗传优化卷积神经网络的金融市场价格波动趋势预测](https://kns.cnki.net/kcms2/article/abstract?v=N2LrlypoGYX4jo1iHbFGa8E1cAc_FMLXBGud4gyszzHKLQlKsiLCIUOA81ELWPVUu2dhXBn4l90P5SPt3mid40_rg6vBneksIDTY11_BBAafDGXgBHCZGEW7sjUGBHJ0oc-4tzi30lLbXTLrx2aiJjOR5ViEN7yu27CHz5weZ8QKhnGv4Nx0HQ==&uniplatform=NZKPT&language=CHS)

Imaging Time Series ([Wang and Oates, 2015](https://www.ijcai.org/Proceedings/15/Papers/553.pdf))

Data: 201101 - 202405的城市商品住宅价格指数（201101=100），length=

## 序列模型：

- 市场情绪分类及预测：

1. 基于新闻数据(THUCnews)用预训练的FinBert计算得到情感得分，
 THUCnews没有时间戳，但是有标题
2. 得到情感得分的时间序列
3. 然后和其他信息一起输入LSTM，比较有情感和没有情感下的预测表现。
参考
(https://zhuanlan.zhihu.com/p/718500113)
(https://github.com/EagleAdelaide/FinSen_Dataset/blob/main/README.md)






