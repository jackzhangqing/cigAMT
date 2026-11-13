SAITS 官方结构适配说明
=====================

1. 来源和许可
官方仓库：https://github.com/WenjieDu/SAITS
固定提交：660b87f19c1277065f314f24134f646229e89ca9
提交时间：2025-08-26T00:29:53Z
论文：Wenjie Du, David Cote, Yan Liu. SAITS: Self-Attention-based Imputation
for Time Series. Expert Systems with Applications, 219:119619, 2023.
DOI：https://doi.org/10.1016/j.eswa.2023.119619
预印本：https://arxiv.org/abs/2202.08516
许可：MIT，完整版权和许可文本见 LICENSE。

upstream/ 保存未修改的官方 saits.py、layers.py、utils.py 和 LICENSE。
UPSTREAM.json 记录逐文件下载地址、字节数、SHA256、固定提交。
official_model.py 是官方 saits.py 的运行副本，只修改两条导入，改为包内导入。
layers.py 与官方文件逐字节一致，没有替换为 SDPA，也没有修改注意力。
metrics.py 只抽出官方所需 masked_mae_cal，避免引入无关的 sklearn/绘图库依赖。
adapter.py 是本项目新增的输入输出适配器。

2. 模型范围和默认配置
用于 Interpolation 和 Imputation，输入完整2048点时间轴，不分块、不降采样。
默认：4通道，T=2048，d_model=64，d_inner=128，n_layers=2，n_heads=4，
dropout=0.1，diagonal_attention_mask=True。
两个DMSA模块各有n_layers个独立层；采用官方inner_group策略，组内重复次数1。
q/k/v每头维度d_model/n_heads；包含最终attention-weighted combination。

3. 接口与标签隔离
from specialist_baselines.vendor.saits import SAITSAdapter
model = SAITSAdapter(sequence_length=2048)
output = model(values, observed_mask)
也支持 model({'input': values, 'observed_mask': observed_mask})。
values和mask形状均为[B,4,2048]，通道Ex/Ey/Hx/Hy，mask为0/1，1表示可见。
输出 {'prediction': tensor[B,4,2048]}，已观察点保持原输入值。
模型内部只读取input与observed_mask；字典中的target等字段不进入网络。
隐藏位置通过torch.where置零，NaN/Inf隐藏占位符不会进入模型。
已观察位置必须为有限值。掩码或形状错误直接报错。
官方EncoderLayer保存普通device属性，适配器每次forward同步到输入device，
保证.to(cuda)与检查点恢复后的对角掩码位置正确。

4. 训练协议说明（论文必须明确）
本适配器调用官方impute()，保留官方网络结构与重建流程。
训练由统一benchmark引擎对离线已固定缺失位置计算missing-target MSE。
不调用官方forward(inputs,stage)的三分支Observed Reconstruction Task MAE，
也不额外随机遮挡数据执行官方Masked Imputation Task及联合MAE训练。
这应表述为“官方SAITS结构在统一监督协议下的适配”，不能声称完整复现原论文
ORT+MIT训练协议，也没有加载官方预训练权重。

5. 显存与混合精度
这是全分辨率二次注意力。B=256,H=4,T=2048时，仅一张FP32 attention矩阵
就需要16GiB，训练还保留多层中间量和梯度，不能直接用物理batch256。
由外部引擎使用小物理microbatch（建议先1–8）和梯度累积，实现有效batch256；
实际能否使用8应先用服务器显存预检，不作A100显存/耗时保证。
保持官方显式注意力，第二DMSA块的最终attention权重用于融合，不能直接丢弃。
原官方masked_fill使用-1e9，在fp16中可能溢出；本适配器明确拒绝fp16。
可用fp32或bf16。CPU bf16接口已检查，CUDA执行尚待服务器预检。

6. 独立验证
在新的独立specialist基准包根目录执行：
python -B -m unittest specialist_baselines.vendor.saits.test_saits -v
若从旧开发暂存目录benchmark_20261023运行，命名空间前缀需增加specialist_baselines：
python -B -m unittest specialist_baselines.vendor.saits.test_saits -v
验证内容：官方来源哈希、运行副本仅导入变化、与官方参考forward及全部参数
梯度逐位一致、隐藏NaN/Inf与oracle标签不影响预测、形状和掩码校验、
检查点往返、完整2048×4默认结构前向反向、CPU bf16前向反向。
这些是正确性检查，不是正式训练或性能结果。

本目录可独立复制到新的specialist基准目录，不需要修改既有amtbench核心，
不会改变旧基准代码指纹。新训练和输出应放在新基准包目录。
