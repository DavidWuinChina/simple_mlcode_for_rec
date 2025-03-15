import numpy as np
import pandas as pd



# 假设 BaselineCFBySGD 类已经定义
class BaselineCFBySGD:
    def __init__(self, number_epochs, alpha, reg, columns=["uid", "iid", "rating"]):
        # 初始化模型参数
        
        self.number_epochs = number_epochs
        # 训练的迭代次数
        self.alpha = alpha
        # 学习率
        self.reg = reg
        # 正则化参数
        self.columns = columns
  
    def sgd(self):
        """
        利用随机梯度下降更新 bu, bi 的值
        :return: bu, bi
        """
        bu = dict(zip(self.users_ratings.index,np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index,np.zeros(len(self.items_ratings))))
    
        for epoch in range(self.number_epochs):
            print("iter%d" % epoch)
            for uid, iid, real_rating in self.dataset.itertuples(index=False):
                # 计算预测评分
                prediction = self.global_mean + bu[uid] + bi[iid]
                # 计算误差
                error = real_rating - prediction
                # 更新偏置项
                bu[uid] += self.alpha * (error - self.reg * bu[uid])
                bi[iid] += self.alpha * (error - self.reg * bi[iid])
        return bu, bi

    def fit(self, dataset):
        """
        :param dataset: uid, iid, rating
        """
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1],self.columns[2]]]
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0],self.columns[2]]]
        # 全局评分均值
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 初始化偏置项
        self.bu,self.bi = self.sgd()

    def predict(self, uid, iid):
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating


if __name__ == "__main__":
    # 定义数据类型
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]

    # 读取数据
    dataset = pd.read_csv(r"C:\python文件\推荐算法实例\ml-latest-small\ml-latest-small\ratings.csv", usecols=range(3), dtype=dict(dtype))

    # 实例化模型
    bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "movieId", "rating"])

    # 训练模型
    bcf.fit(dataset)

    while True:
        uid = int(input("uid: "))
        iid = int(input("iid: "))
        print(bcf.predict(uid, iid))
