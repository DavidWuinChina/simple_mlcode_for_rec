import numpy as np
import pandas as pd


def datasplit(data_path, x=0.8, random=False):
    print("打开评分数据集...")

    # 设定数据类型
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))

    testset_index = []
    for uid in ratings.groupby("userId").any().index:
        uesr_rating_data = ratings[ratings["userId"] == uid].dropna()
        if random:
            index = list(uesr_rating_data.index)
            np.random.shuffle(index)
            _index = round(len(uesr_rating_data.index) * x)
            testset_index += list(index[_index:])
        else:
            index = round(len(uesr_rating_data.index) * x)
            testset_index += list(uesr_rating_data.index.values[index:])

    testset = ratings.loc[testset_index]
    trainset = ratings.drop(testset_index)
    return trainset, testset


def accuracy(predict_results, method="all"):
    def rmse(predict_results):
        '''
        rmse评估指标
        :param predict_results:
        :return: rmse
        '''
        length = 0
        _rmse_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
        return round(np.sqrt(_rmse_sum / length), 4)

    def mae(predict_results):
        '''
        mae评估指标
        :param predict_results:
        :return: mae
        '''
        length = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _mae_sum += abs(pred_rating - real_rating)
        return round(_mae_sum / length, 4)

    def rmse_mae(predict_results):
        '''
        mae评估指标
        :param predict_results:
        :return: mae
        '''
        length = 0
        _rmse_sum = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
            _mae_sum += abs(pred_rating - real_rating)
        return round(np.sqrt(_rmse_sum / length), 4), round(_mae_sum / length, 4)

    if method.lower() == "rmse":
        rmse(predict_results)
    elif method.lower() == "mae":
        mae(predict_results)
    else:
        return rmse_mae(predict_results)


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
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

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
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        # 物品评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 全局评分均值
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 初始化偏置项
        self.bu, self.bi = self.sgd()

    def predict(self, uid, iid):
        '''评分预测'''
        if iid not in self.items_ratings.index:
            raise Exception(
                "无法预测用户<{uid}>对电影<{iid}>的评分，因为训练集中缺失<{iid}>的数据".format(uid=uid, iid=iid))

        predict_rating = self.global_mean + self.bu[uid] + self.bi[iid]
        return predict_rating

    def test(self, testset):
        '''预测测试集数据'''
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating


if __name__ == '__main__':

    trainset, testset = datasplit(r"C:\python文件\推荐算法实例\ml-latest-small\ml-latest-small\ratings.csv", random=True)

    bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "movieId", "rating"])
    bcf.fit(trainset)

    pred_results = bcf.test(testset)

    rmse, mae = accuracy(pred_results)

    print("rmse: ", rmse, "mae: ", mae)
