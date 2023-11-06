from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import log

iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666666)


class Node:
    def __init__(self,x_data, y_label, dimension, value):
        self.x_data = x_data
        self.y_label = y_label
        self.dimension = dimension
        self.value = value
        self.left = None
        self.right = None

class DTree:
    def __init__(self):
        self.root = None
    
    def fit(self,x_train, y_train):
        def entropy(y_label):
            counter = Counter(y_label)
            ent = 0.0
            for num in counter.values():
                # TODO: 计算熵：H=-\Sigma_{i=1}^n p_i \log \left(p_i\right)
                #_____________________________________#
            return ent
        
        def one_split(x_data, y_label):

            best_entropy = float('inf')
            best_dimension = -1
            best_value = -1

            for d in range(x_data.shape[1]):
                sorted_index = np.argsort(x_data[:, d])
                for i in range(1,len(x_data)):
                    if x_data[sorted_index[i], d] != x_data[sorted_index[i - 1], d]:
                        value = (x_data[sorted_index[i], d] + x_data[sorted_index[i-1], d]) / 2
                        # TODO: 做一次Split操作
                        #_____________________________________#

                        p_left = len(x_left) / len(x_data)
                        p_right = len(x_right) / len(x_data)
                        # TODO: 计算Split后的信息增益
                        #_____________________________________#
                        if ent < best_entropy:
                            best_entropy = ent
                            best_dimension = d
                            best_value = value
            return best_entropy, best_dimension, best_value
        
        def split(x_data, y_label, dimension, value):
            """
            x_data: 输入特征
            y_label: 输入标签类别
            dimension: 选取输入特征的维度索引
            value: 划分特征的数值

            return 左子树特征，右子树特征，左子树标签，右子树标签
            """
            # TODO: 定义以何种标准进行Split
            #_____________________________________#
            return x_data[index_left], x_data[index_right], y_label[index_left], y_label[index_right]
        
        def create_tree(x_data, y_label):
            ent, dim, value = one_split(x_data, y_label)
            x_left, x_right, y_left, y_right = split(x_data, y_label, dim, value)
            node = Node(x_data, y_label, dim, value)
            if ent < 0.000000001:
                return node
            node.left = create_tree(x_left, y_left)
            node.right = create_tree(x_right, y_right)
            return node

        self.root = create_tree(x_train, y_train)
        
        return self
    
    def predict(self,x_predict):
        def travel(x_data, node):
            p = node
            if x_data[p.dimension] <= p.value and p.left:
                pred = travel(x_data, p.left)
            elif x_data[p.dimension] > p.value and p.right:
                pred = travel(x_data, p.right)
            else:
                counter = Counter(p.y_label)
                pred = counter.most_common(1)[0][0]
            return pred
        
        y_predict = []
        for data in x_predict:
            y_pred = travel(data, self.root)
            y_predict.append(y_pred)
        return np.array(y_predict)
    
    def score(self,x_test,y_test):
        y_predict = self.predict(x_test)
        return np.sum(y_predict == y_test) / len(y_predict)
    
    def __repr__(self):
        return "DTree(criterion='entropy')"
    
dt = DTree()
dt.fit(x_train, y_train)
print(dt.score(x_test,y_test))