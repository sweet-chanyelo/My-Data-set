"""
数据链接：
数据描述：航天继电器故障诊断数据，共四种输入指标：6种故障状态：normal、
    作者：明志超
    时间：2021年夏

相关论文：
    最近更新：2022-09-09
    更新内容：新增噪声为50%情况下的数据集

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_process import Data_propress


def fig1(input, label):
    # 画图
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(411)
    ax1.plot(input[:, 0])
    ax2 = fig1.add_subplot(412)
    ax2.plot(input[:, 1])
    ax3 = fig1.add_subplot(413)
    ax3.plot(input[:, 2])
    ax4 = fig1.add_subplot(414)
    ax4.plot(input[:, 3])


if __name__ == '__main__':
    # 导入
    df1 = pd.read_excel("继电器数据.xlsx", '训练数据', header=None)
    input = np.array(df1)
    df2 = pd.read_excel("继电器数据.xlsx", 'train-label', header=None)
    label = np.array(df2)
    print(input.shape, label.shape)
    # 画图
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(411)
    ax1.plot(input[:, 0])
    ax2 = fig1.add_subplot(412)
    ax2.plot(input[:, 1])
    ax3 = fig1.add_subplot(413)
    ax3.plot(input[:, 2])
    ax4 = fig1.add_subplot(414)
    ax4.plot(input[:, 3])

    # 加入噪声
    noise = np.random.randn(1, 1200)  # 标准噪声
    train_data = np.zeros((input.shape[0], input.shape[1]))
    train_data[:, 0] = input[:, 0] + noise * 0.2 * input[:, 0]
    train_data[:, 1] = input[:, 1] + noise * 0.2 * input[:, 1]
    train_data[:, 2] = input[:, 2] + noise * 0.2 * input[:, 2]
    train_data[:, 3] = input[:, 3] + noise * 0.2 * input[:, 3]
    # 画图
    fig1 = plt.figure(2)
    ax1 = fig1.add_subplot(411)
    ax1.plot(train_data[:, 0])
    ax2 = fig1.add_subplot(412)
    ax2.plot(train_data[:, 1])
    ax3 = fig1.add_subplot(413)
    ax3.plot(train_data[:, 2])
    ax4 = fig1.add_subplot(414)
    ax4.plot(train_data[:, 3])

    # 存储
    np.savetxt("train_data_0.2.csv", train_data, delimiter=',')
    np.savetxt("train_label_data.csv", label, delimiter=',')

    plt.show()