# LSTM_emotionclassfy_inference(循环神经网络_电影评论情感分析)
如何使用此代码？
环境：
操作系统：win10
版本：python3.9+
CUDA 工具包 10.2 配合cuDNN v7.6.5
平台：PaddlePaddle2.2
硬件：GPU:GTX 1050 Ti

操作过程：
1：安装所需模块
2：确保网络畅通
3：部分路径可能需要修改

注：1.预测阶段可能引发一个类型错误“int64”,解决办法：等待官方修复或更新paddle.fluid
2.训练过程图像收敛不明显，可以尝试修改隐藏层维度调整，最终准确率约为0.9
