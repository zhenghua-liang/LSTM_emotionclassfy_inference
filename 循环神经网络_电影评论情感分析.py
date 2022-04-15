# -*- coding: utf-8 -*-
import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
import os

# 训练集与测试集准备
BATCH_SIZE = 128
BUF_SIZE = 512
word_dict = paddle.dataset.imdb.word_dict()
dict_dim = len(word_dict)
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.imdb.train(word_dict),
        BUF_SIZE
    ),
    batch_size=BATCH_SIZE
)
test_reader = paddle.batch(
    paddle.dataset.imdb.test(word_dict),
    batch_size=BATCH_SIZE
)


# 定义三层栈式双向LSTM
def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):
    # 参数列表（data：传入数据,
    # input_dim：词典的大小,
    # class_dim：情感分类的类别数,
    # enm_dim：词向量维度,
    # hid_dim：隐藏层维度,
    # stacked_num：LSTM双向栈层数）

    # 计算词向量
    emb = fluid.layers.embedding(
        input=data,
        size=[input_dim, emb_dim],
        is_sparse=True
    )
    # 第一层栈，全连接层
    fcl = fluid.layers.fc(input=emb, size=hid_dim)
    # lstm层
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fcl, size=hid_dim)
    inputs = [fcl, lstm1]
    # 其余的所有栈结构
    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc,
            size=hid_dim,
            is_reverse=(i % 2) == 0
        )
        inputs = [fc, lstm]
    # 池化层
    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')
    # 全连接层,softmax预测
    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last],
        size=class_dim,
        act='softmax'
    )
    return prediction


# 数据层定义
paddle.enable_static()
# 定义输入数据，lod_level不为0指定输入数据为序列数据
words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
model = stacked_lstm_net(words, dict_dim, 2, 200, 128, 3)
# data：（传入数据）words
# input_dim：(词典的大小）dict_dim
# class_dim：(情感分类的类别数)2
# enm_dim：(词向量维度)200
# hid_dim：(隐藏层维度)128
# stacked_num：(LSTM双向栈层数)3

# 定义损失函数与准确率函数
# 使用交叉熵损失函数，描述真实样本标签和预测概率之间的差值
cost = fluid.layers.cross_entropy(input=model, label=label)
# 使用类交叉熵函数计算predict和label之间的损失函数
avg_cost = fluid.layers.mean(cost)
# 计算分类准确率
acc = fluid.layers.accuracy(input=model, label=label)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)

# 定义使用CPU还是GPU，使用CPU时use_cuda =False,使用GPU时use_cuda = True
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
# 创建一个Executor实例exe
exe = fluid.Executor(place)
# 正式进行网络训练前，需先执行参数初始化
exe.run(fluid.default_startup_program())
# 执行训练之前，需要定义输入的数据维度，一条句子对应一个标签
feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

# 展示模型训练曲线
all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []


def draw_train_process(title, iters, costs, accs, label_cost, label_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=label_acc)
    plt.legend()
    plt.grid()
    plt.show()


# 训练模型
EPOCH_NUM = 5
model_save_dir = "./emotionclassify.inference.model"
for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 给模型喂入数据
                                        fetch_list=[avg_cost, acc])  # fetch 误差、准确率

        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        # 每10个batch打印一次信息 误差、准确率
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f,Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # 执行训练程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        test_accs.append(test_acc[0])  # 每个batch的准确率
        test_costs.append(test_cost[0])  # 每个batch的误差

    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))

# 保存模型
# 如果保存路径不存在就创建
model_save_dir = r'D:\Pycharm_project\PaddlePaddle\LSTM_emotionclassify_inference'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,  # 保存预测Program的路径
                              ['words'],  # 预测需要feed的数据
                              [model],  # 保存预测结果
                              exe)  # executor 保存预测模型
print('训练模型保存完成!')

# 训练过程可视化
draw_train_process("training",
                   all_train_iters,
                   all_train_costs,
                   all_train_accs,
                   "training cost",
                   "training acc")

# 网络预测
infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()
inference_program = model_save_dir
# 定义预测数据
reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']
reviews = [c.split() for c in reviews_str]

# 构造预测数据
UNK = word_dict['<unk>']
lod = []
for c in reviews:
    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])
base_shape = [[len(c) for c in lod]]
tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

# 开始预测
with fluid.scope_guard(inference_scope):
    [
        inference_program,
        feed_target_names,
        fetch_targets
    ] = fluid.io.load_inference_model(model_save_dir, infer_exe)
    results = infer_exe.run(
        program=inference_program,
        feed={feed_target_names[0]: tensor_words},
        fetch_list=fetch_targets
    )
    for i, r in enumerate(results[0]):
        print("\'%s\'的预测结果为：正面概率为：%0.5f,负面概率为：%0.5f" % (reviews_str[i], r[0], r[1]))
