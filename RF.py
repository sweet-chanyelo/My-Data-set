"""
算法：RF
测试精度: 80.09%
拟合精度：96.66%
"""
import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from data_process import Data_propress


if __name__ == '__main__':
    start = time.perf_counter()  # 开始时间
    out = []
    # 导入数据
    df1 = pd.read_csv('train_data_0.2.csv', header=None)
    df2 = pd.read_csv('fault_diagnosis_test_data.csv', header=None)
    X_train = np.array(df1)
    X_test = np.array(df2)
    # 导入标签数据
    df3 = pd.read_csv('train_label_data.csv', header=None)
    df4 = pd.read_csv('test_label_data.csv', header=None)
    Y_train = np.array(df3)
    Y_train_sequence = np.argmax(Y_train, axis=1)
    Y_test = np.array(df4)
    Y_test_sequence = np.argmax(Y_test, axis=1)

    # 数据处理
    dp = Data_propress()
    X_train = dp.minmaxscaler(data=X_train)  # 标准化
    X_test = dp.minmaxscaler(data=X_test)

    # 运行
    RF = RandomForestClassifier()
    # 训练集
    RF.fit(X_train, Y_train_sequence)
    # 测试集
    test_Y = RF.predict(X_test)
    Y_score = RF.predict_proba(X_test)

    # 计算平均roc
    # ROC = np.zeros(test_Y.shape[1])
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(test_Y.shape[1]):
    #     fpr[i], tpr[i], _ = metrics.roc_curve(test_Y[:, i], Y_score[:, i])
    #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    #     ROC[i] = metrics.auc(fpr[i], tpr[i])
    # print("二分类roc得分：", ROC)
    # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(test_Y.ravel(), Y_score.ravel())
    # roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    #

    # 输出
    end = time.perf_counter()  # 停止时间
    print('运行时间：%s' % (end - start))
    print("准确率为：", metrics.accuracy_score(Y_test_sequence, test_Y))
    # print("AUC值为：", metrics.roc_auc_score(test_Y, Y_score))

    # 画图
    # plt.figure(1)
    # plt.plot(test_Y, '-v')
    # plt.plot(Y_test, 'r')
    #
    # lw = 1
    # plt.figure(3)
    # plt.style.use('seaborn-darkgrid')
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle='-', linewidth=1)
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()
    # 存储
    # output_excel = pd.DataFrame(list(train_out) + list(test_out))
    # writer = pd.ExcelWriter('RF故障缺失.xlsx')
    # output_excel.to_excel(writer, sheet_name='data', startcol=0, index=False)
    # writer.save()