from multiprocessing import Process, Manager
from sklearn.tree import DecisionTreeRegressor

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import time


def ReadLittleData():
    print("reading 10000 row data ")
    train_all = pd.read_csv(r'data/train10000.csv', header=None)
    label_all = pd.read_csv(r'data/label10000.csv', header=None)
    test_all = pd.read_csv(r'data/test10000.csv', header=None)
    print(train_all.shape, label_all.shape, test_all.shape)
    return  train_all, label_all, test_all

class PA_RandomForestRegressor():

    def __init__(self,
                 n_estimators = 10,
                 criterion="squared_error",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs = 1,
                 random_state=None,
                 verbose=0,
                 splitter='best',
                 warm_start=False):

        self.random_state = random_state
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.warm_start = warm_start
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.job_forest_queue = Manager().Queue()

        self.splitter = splitter
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
    #job_forest_queue 表示 决策树队列 也可以认为是决策树森林


    def fit(self, X, Y):

        # 分出每个进程 应该生成几颗决策树
        job_tree_num = int ( self.n_estimators /  self.n_jobs)


        processes = list()

        #随机森林的决策树参数
        dtr_args = {            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "random_state": self.random_state,
            "splitter": self.splitter
        }

        # 生成N个进程
        for i in range(self.n_jobs):
            # 参数
            #job_forest_queue   为决策树队列  每个进程生成的决策树将加入到此队列  这是随机森林对象的一个属性
            #i                  为进程号
            #job_tree_num       表示该进程需要生成的决策树
            #X Y                表示训练数据 和结果数据
            #dtr_args           表示传入的决策树参数
            p = Process(target=signal_process_train, args=(self.job_forest_queue, i,job_tree_num , X, Y, dtr_args))
            print ('process Num. ' + str(i) + "  will start train")
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print ("Train end")

    def predict(self, X):

        result_queue = Manager().Queue()

        processes = list()
        # 分出每个线程 应该预测几颗决策树
        job_tree_num = int(self.n_estimators / self.n_jobs)
        # 生成N个进程
        for i in range(self.n_jobs):
            # 参数
            # job_forest_queue   为决策树队列   这是随机森林对象的一个属性
            # i                  为进程号
            # job_tree_num       表示该进程需要生成的决策树
            # X                  表示待预测数据
            # result_queue       表示用于存放预测结果的数据
            p = Process(target=signal_process_predict, args=(self.job_forest_queue, i, job_tree_num, X, result_queue))
            print('process Num. ' + str(i) + "  will start predict")
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        result = np.zeros(X.shape[0])

        #把每个进程的平均结果再一次加起来求平均， 得到最终结果
        for i in range(result_queue.qsize()):
            result =  result + result_queue.get()

        result = result / self.n_jobs
        print("Predict end")
        return  result

#单进程训练函数
def signal_process_train(job_forest_queue,process_num,job_tree_num,X, Y, dtr_args):
    #循环生成决策树， 并且把生成的决策树加入到job_forest_queue 队列中
    for i in range(0, job_tree_num):
        # 使用bootstrap 方法生成  1个 训练集

        len = int(Y.shape[0] * 0.7)
        indexs = np.arange(len)
        np.random.shuffle(indexs)
        x = []
        y = []
        for ind in indexs:
            x.append(X.values[ind])
            y.append(Y.values[ind])

        # 对这个样本 进行训练 并且根据传入的决策树参数 生成1棵决策树
        dtr = DecisionTreeRegressor(criterion=dtr_args['criterion'], max_depth=dtr_args['max_depth'],
                      max_features=dtr_args['max_features'],max_leaf_nodes=dtr_args['max_leaf_nodes'],
                      min_impurity_decrease=dtr_args['min_impurity_decrease'],
                      min_samples_leaf=dtr_args['min_samples_leaf'], min_samples_split=dtr_args['min_samples_split'],
                      min_weight_fraction_leaf=dtr_args['min_weight_fraction_leaf'],
                      random_state=dtr_args['random_state'], splitter=dtr_args['splitter'])

        dtr.fit(x,y)

        if (i% int(job_tree_num/10 or i<10) == 0):
            print ('process Num. ' + str(process_num) +  "  trained  " + str(i) + '  tree')

        # 决策树存进森林（决策树队列）
        job_forest_queue.put(dtr)
    print('process Num. ' + str(process_num) + '  train  Done!!')
#单进程预测函数
def signal_process_predict(job_forest_queue,process_num,job_tree_num,X,result_queue):

    # 生成结果矩阵
    result = np.zeros(X.shape[0])

    for i in range(job_tree_num):
        # 从队列中取出一颗树 进行预测
        tree = job_forest_queue.get()
        result_single = tree.predict(X)
        # 将得出的结果加到总结果中
        result = result +result_single


    # 算出平均结果  放入结果队列中
    result = result / job_tree_num
    result_queue.put(result)
    print('process ' + str(process_num) + ' predict Done!!')


if __name__ == '__main__':
    #读取数据
    train_all, label_all, test_all = ReadLittleData()

    plt.figure()
    fig_x = []
    fig_y = []

    for n_process in range(2,8,2):
        # 训练
        time_start = time.time()
        pa_tree = PA_RandomForestRegressor(n_jobs=n_process, n_estimators = 1000,min_samples_split=100,
                                             min_samples_leaf=100, max_depth=8, max_features='log2')
        pa_tree.fit(train_all, label_all)
        # 查看森林队列大小
        print(pa_tree.job_forest_queue.qsize())

        time_end=time.time()
        print('train time: ',time_end-time_start,'s')

        fig_y.append(time_end-time_start)
        fig_x.append(n_process)

        time_start = time.time()
        pre = pa_tree.predict(test_all)

        time_end=time.time()
        print('predict time: ',time_end-time_start,'s')

        # output.to_csv('predict.csv', index=False)

    plt.plot(fig_x, fig_y)  # 绘制曲线 y2
    # 设置横轴标签
    plt.xlabel("n_job")
    # 设置纵轴标签
    plt.ylabel("Train_time")
    plt.show()
    print (fig_x)
    print (fig_y)
