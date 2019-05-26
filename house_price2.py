# [markdown]
# ### 目录
# #### 1. 了解数据
# ##### 1.1 导入数据
# ##### 1.2 可视化数据
# #### 2. 数据清洗
# ##### 2.1 缺失值处理
# ##### 2.2 异常值处理
# ##### 2.3 去除重复的数据
# ##### 2.4 噪音数据的处理
# #### 3. 特征提取
# ##### 3.1 主成分分析 （PCA）
# ##### 3.2 线性判别分析法（LDA）
# #### 4. 划分训练集/验证集
# ##### 4.1 归一化
# ##### 4.2 划分训练集/验证集
# #### 5. 定义评估方法
# #### 6. 构建神经网络
# ##### 6.1 训练模型
# #### 7. 利用测试集评估学习结果
# [markdown]
# ---
# [markdown]
# #### 1. 可视化数据
# ##### 1.1 导入数据
from __future__ import absolute_import, division, print_function, unicode_literals


import os
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  # 回显所有输出

#


# Seperate the features and target value from the train.csv
# features 十三维的特征
# 目标值是房价： prices
data = pd.read_csv('./Row_Data/train.csv')
prices = data['medv']
features = data.drop('medv', axis=1)


#
# 目标：计算房价的最小值
minimum_price = min(prices)

# 目标：计算房价的最大值
maximum_price = max(prices)

# 目标：计算房价的平均值
mean_price = np.mean(prices)

# 目标：计算房价的中值
median_price = np.median(prices)

# 目标：计算房价的标准差
std_price = np.std(prices)

# 目标：输出计算的结果
print("Statistics for Boston housing dataset:")
print("Minimum price: \t$", minimum_price)
print("Maximum price: \t$", maximum_price)
print("Mean price: \t$", mean_price)
print("Median price: \t$", median_price)
print("Standard deviation of prices: \t$", std_price)


#
features.head()

# [markdown]
# ##### 1.2 先随便挑选几个特征来可视化

# #
# # 画出 features.crim 图像
# plt.figure()
# plt.plot(features.crim, color="cornflowerblue", label="crim", linewidth=1)
# plt.ylabel("crim")
# plt.title("The first feature: crim\n")
# plt.legend()
# plt.show()

# # 画出 features.zn 图像
# plt.figure()
# plt.plot(features.zn, color="yellowgreen", label="zn", linewidth=1)
# plt.ylabel("zn")
# plt.title("The second feature: zn\n")
# plt.legend()
# plt.show()

# # 画出 features.indus 图像
# plt.figure()
# plt.plot(features.indus, color="red", label="indus", linewidth=1)
# plt.ylabel("indus")
# plt.title("The third feature: indus\n")
# plt.legend()
# plt.show()

# # 画出 values 图像
# plt.figure()
# plt.plot(prices, color="black", label="medv", linewidth=1)
# plt.ylabel("medv")
# plt.title("The target value: medv\n")
# plt.legend()
# plt.show()

# [markdown]
# #### 2. 数据清洗
# ##### 2.1 缺失值处理

#
# 缺失值处理
# count = 0
# for i in features:
#     count += features[i].isnull().sum()
#     print(features[i].isnull().sum())
# print("空值个数为：", count)

# [markdown]
# ##### 2.2 异常值处理

#
# 异常值处理


# [markdown]
# ##### 2.3 去除重复的数据

#
# 去除重复的数据


# [markdown]
# ##### 2.4 噪音数据的处理

#
# 噪音数据的处理

# [markdown]
# #### 3. 特征提取
# ##### 3.1 主成分分析 （PCA）

#
# 用 PCA 从 13 维度降至 10 维度
features = features.drop('ID', axis=1)
features
pca = PCA(n_components=10)
newfeatures = pca.fit_transform(features)
# 查看前 3 行
newfeatures[:3]


#
features

# [markdown]
# ##### 3.2 线性判别分析法（LDA）

#
# features.shape
# prices.shape
# 载入 LDA 模型，设置 n_components = 10
# lda = LinearDiscriminantAnalysis(n_components=10)
# featuresnew = lda.fit(features, prices).transform(features)

# [markdown]
# #### 4. 划分训练集/验证集
# ##### 4.1 归一化
# [markdown]
# #### 因为各个 feature 的取值范围区别较大，所以用 minmax_normalization 对数据进行归一化.
# #### 这样可以把每个 feature 都压缩到 0-1 的范围.
# #### 常用的最小最大规范化方法：
# $$\hat x = {\frac{x - min(x)}{max(x) - min(x)}}$$

#
# 自定义一个最小最大规范化的函数


def minmax_normalization(data):
    xs_max = np.max(data, axis=0)
    xs_min = np.min(data, axis=0)
    xs = (1 - 0) * (data - xs_min) / (xs_max - xs_min) + 0
    return xs


#
# 将 data 传入上面的规范化函数
# 规范化得到的数据存放在 m_n_data 中
m_n_data = minmax_normalization(data)
m_n_data.head()

# [markdown]
# ##### 4.2 划分训练集/验证集
# [markdown]
# #### 分割比例为：80%的数据用于训练，20%用于测试 test_size = 0.2
# #### 将数据集分成训练集和测试集的好处：既可以用于训练又可以用于测试，而且不会相互干扰，而且可以对训练模型进行有效的验证。

#
m_n_features = m_n_data.drop('medv', axis=1)
m_n_prices = m_n_data['medv']
m_n_features.drop('ID', inplace=True, axis=1)

#
# Split the dataset as traing set and testing set
X_train, X_test, y_train, y_test = train_test_split(
    m_n_features, m_n_prices, test_size=0.2, random_state=1)  # random_state=1:不会随机划分
X_train.head()

# [markdown]
# #### 5. 定义评估方法
# [markdown]
# #### $SSE$ (最小化误差平方和 Sum of Squares for Error): $$SSE = {\sum_{k=1}^n (y_i - \hat y_i)^2}$$
# + 其中：$n$ 是数据的条数
# #### $MSE$ (均方误差 Mean Squared Error): $$MSE = {\frac{1} {n}}{\sum_{k=1}^n (y_i - \hat y_i)^2}$$
# + $MSE$ 越接近 0 模型的性能越好，但是不全面
# #### 残差平方和公式: $$R^2(y, \hat y) = {\frac{\sum_{k=1}^n (y_i - \hat y_i)^2}{\sum_{k=1}^n (y_i - \overline y)^2}}$$
# + 模型越好：$$R^2(y, \hat y)→1$$
# + 模型越差：$$R^2(y, \hat y)→0$$

#
# 自定义一个返回最终残差和的函数


def performance_metric(y_true, y_predict):
    '''计算实际值与预测值的R2分数'''
    score = r2_score(y_true, y_predict)
    return score

# [markdown]
# #### 6. 构建神经网络
# ##### 6.1 训练模型

#
# DNN


#
def build_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(13,)),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.01)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


#
# Display training progress by printing a single dot for each completed epoch


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(1)
    plt.subplot(311)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(312)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 1])
    plt.legend()

    plt.subplot(313)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(hist['epoch'], hist['loss'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'],
             label='Val Error')
    plt.ylim([0, 1])
    plt.legend()

    plt.show()





model = build_model()

model.summary()

#
keras.utils.plot_model(model, 'my_first_model.png')
keras.utils.plot_model(
    model, 'my_first_model_with_shape_info.png', show_shapes=True)

EPOCHS = 100

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(m_n_features, m_n_prices, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, cp_callback, PrintDot()])



test_data = pd.read_csv('./Row_Data/test.csv')
predictions = pd.DataFrame(test_data["ID"])
test =  test_data.drop("ID", axis=1)

test_predictions = model.predict(test).flatten()
predictions["medv"] = test_predictions


predictions.to_csv("./Row_Data/output.csv", encoding='utf-8', index=False)

plot_history(history)

# [markdown]
# #### 7. 利用测试集评估学习结果

#
# Evalution
