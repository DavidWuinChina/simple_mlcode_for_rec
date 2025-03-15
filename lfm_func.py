'''
LFM Model
'''
import pandas as pd
import numpy as np

# 评分预测   1-5
class LFM(object):

    def __init__(self, alpha, reg_p, reg_q, number_LatentFactors=10, number_epochs=10, columns=["uid", "iid", "rating"]):
        self.alpha = alpha  # 学习率
        self.reg_p = reg_p    # P矩阵正则
        self.reg_q = reg_q    # Q矩阵正则
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs    # 最大迭代次数
        self.columns = columns

    def fit(self, dataset):
        '''
        fit dataset
        :param dataset: uid, iid, rating
        :return:
        '''

        self.dataset = pd.DataFrame(dataset)

        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.globalMean = self.dataset[self.columns[2]].mean()

        self.P, self.Q = self.sgd()
    def _init_matrix(self):
        '''
        初始化P和Q矩阵，同时为设置0，1之间的随机值作为初始值
        :return:
        '''
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings),
                           self.number_LatentFactors).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings),
                           self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def sgd(self):
        '''
        使用随机梯度下降，优化结果
        :return:
        '''
        P, Q = self._init_matrix()

        for i in range(self.number_epochs):
            print("iter%d" % i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                v_pu = P[uid]
                v_qi = Q[iid]
                err = np.float32(r_ui - np.dot(v_pu, v_qi))

                v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (err * v_pu - self.reg_q * v_qi)

                P[uid] = v_pu
                Q[iid] = v_qi

                # for k in range(self.number_of_LatentFactors):
                #     v_pu[k] += self.alpha*(err*v_qi[k] - self.reg_p*v_pu[k])
                #     v_qi[k] += self.alpha*(err*v_pu[k] - self.reg_q*v_qi[k])

                error_list.append(err ** 2)
            print(np.sqrt(np.mean(error_list)))

        return P, Q

    def predict(self, uid, iid):
        '''
        预测评分
        :param uid: 用户ID
        :param iid: 物品ID
        :return: 预测评分值（范围1-5）
        '''
        # 处理冷启动问题：若用户/物品未参与训练，返回全局平均分
        if uid not in self.P or iid not in self.Q:
            return np.clip(self.globalMean, 1, 5)

        # 计算用户隐向量与物品隐向量的点积[1,7](@ref)
        pred_rating = np.dot(self.P[uid], self.Q[iid])

        # 将预测值裁剪到评分范围内（1-5分）
        return np.clip(pred_rating, 1.0, 5.0)

if __name__ == '__main__':
    # 定义数据类型
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    # 读取数据
    dataset = pd.read_csv(r"C:\python文件\推荐算法实例\ml-latest-small\ml-latest-small\ratings.csv", usecols=range(3), dtype=dict(dtype))
    lfm = LFM(0.01, 0.01, 0.01, 10, 20, ["userId", "movieId", "rating"])
    lfm.fit(dataset)
    while True:
        uid = int(input("请输入用户ID："))
        iid = int(input("请输入物品ID："))
        print("预测评分：", lfm.predict(uid, iid))
        # 修改最后一行
        real_rating = dataset[(dataset["userId"] == uid) & (dataset["movieId"] == iid)]["rating"]
        if len(real_rating) > 0:
            print("真实评分：", real_rating.values[0])
        else:
            print("真实评分：数据不存在")