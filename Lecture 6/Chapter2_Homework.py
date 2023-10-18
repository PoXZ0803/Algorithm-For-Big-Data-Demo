import numpy as np
from typing import List, Union


class EuclideanLSH:

    def __init__(self, tables_num: int, a: int, depth: int):
        """
        parameter tables_num: hash_table的个数
        parameter a: a越大，被纳入同个位置的向量就越多，即可以提高原来相似的向量映射到同个位置的概率，反之，则可以降低原来不相似的向量映射到同个位置的概率
        parameter depth: 向量的维度数
        """
        self.tables_num = tables_num
        self.a = a
        # 为了方便矩阵运算，调整了shape，每一列代表一个hash_table的随机向量
        self.R = np.random.random([depth, tables_num])
        self.b = np.random.uniform(0, a, [1, tables_num])
        # 初始化空的hash_table
        self.hash_tables = [dict() for i in range(tables_num)]


    def _hash(self, inputs: Union[List[List], np.ndarray]):
        """
        将向量映射到对应的hash_table的索引
        parameter inputs: 输入的单个或多个向量
        return: 每一行代表一个向量输出的所有索引，每一列代表位于一个hash_table中的索引
        """
        # TODO: 计算H(V) = |V·R + b| / a，R是一个随机向量，a是桶宽，b是一个在[0,a]之间均匀分布的随机变量，用hash_val表示H(V)
        #___________________________#

        return hash_val

    def insert(self, inputs):
        """
        将向量映射到对应的hash_table的索引，并插入到所有hash_table中
        """
        inputs = np.array(inputs)
        if len(inputs.shape) == 1:
            # TODO: 将inputs转化为二维向量
            #___________________________#

        hash_index = self._hash(inputs)
        for inputs_one, indexs in zip(inputs, hash_index):
            for i, key in enumerate(indexs):
                # i代表第i个hash_table，key则为当前hash_table的索引位置
                # inputs_one代表当前向量
                self.hash_tables[i].setdefault(key, []).append(tuple(inputs_one))


    def query(self, inputs, nums=20):
        """
        查询与inputs相似的向量，并输出相似度最高的nums个
        :param inputs: 输入向量
        :param nums:
        :return:
        """
        hash_val = self._hash(inputs).ravel()
        candidates = set()

        # TODO: 将相同索引位置的向量添加到候选集中
        #___________________________#

        # TODO: 根据向量距离进行排序
        #___________________________#
        return candidates[:nums]

    @staticmethod
    def euclidean_dis(x, y):
        """
        计算欧式距离
        """
        x = np.array(x)
        y = np.array(y)

        return np.sqrt(np.sum(np.power(x - y, 2)))


if __name__ == '__main__':
    data = np.random.random([10000, 100])
    query = np.random.random([100])

    lsh = EuclideanLSH(10, 1, 100)
    lsh.insert(data)
    res = lsh.query(query, 20)
    res = np.array(res)
    print(np.sum(np.power(res - query, 2), axis=-1))

    sort = np.argsort(np.sum(np.power(data - query, 2), axis=-1))
    print(np.sum(np.power(data[sort[:20]] - query, 2), axis=-1))
    print(np.sum(np.power(data[sort[-20:]] - query, 2), axis=-1))
