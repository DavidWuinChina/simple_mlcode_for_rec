import numpy as np
import pandas as pd


# 假设 BaselineCFBySGD 类已经定义
class BaselineCFBySGD:
    def __init__(self, number_epochs, alpha, reg_bu, reg_bi, columns=["userId", "movieId", "rating"]):
        # 初始化模型参数

        self.number_epochs = number_epochs
        # 训练的迭代次数
        self.alpha = alpha
        # 学习率
        self.reg_bu = reg_bu
        self.reg_bi=reg_bi
        # 正则化参数
        self.columns = columns

    def als(self):
        '''
        利用随机梯度下降，优化bu，bi的值
        :return: bu，bi
        '''
        # 初始化bu、bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.number_epochs):
            print("iter%d" % i)
            for iid, uids, ratings in self.items_ratings.itertuples(index=True):
                _sum = 0
                for uid, rating in zip(uids, ratings):
                    _sum += rating - self.global_mean - bu[uid]
                bi[iid] = _sum / (self.reg_bi + len(uids))

            for uid, iids, ratings in self.users_ratings.itertuples(index=True):
                _sum = 0
                for iid, rating in zip(iids, ratings):
                    _sum += rating - self.global_mean - bi[iid]
                bu[uid] = _sum / (self.reg_bu + len(iids))
        return bu, bi

    def fit(self, dataset):
        """
        :param dataset: uid, iid, rating
        """
        self.dataset = dataset
        # 用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 全局评分均值
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 初始化偏置项
        self.bu, self.bi = self.als()

    def predict(self, uid, iid):
        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating


if __name__ == "__main__":
    # 定义数据类型
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]

    # 读取数据
    dataset = pd.read_csv(r"C:\python文件\推荐算法实例\ml-latest-small\ml-latest-small\ratings.csv", usecols=range(3),
                          dtype=dict(dtype))

    # 实例化模型
    bcf = BaselineCFBySGD(20, 0.1, 0.1, 0.1, ["userId", "movieId", "rating"])


    # 训练模型
    bcf.fit(dataset)

    while True:
        uid = int(input("uid: "))
        iid = int(input("iid: "))
        print(bcf.predict(uid, iid))
