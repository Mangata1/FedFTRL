
import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'], params['lambda'])
        super(Server, self).__init__(params, learner, dataset)
        self.params=params

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        acc=[]
        loss=[]
        gd=[]
        sparse=[]
        select_device=[]

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))
                acc.append(np.sum(stats[3])*1.0/np.sum(stats[2]))
                loss.append(np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2]))
                

            model_len = process_grad(self.latest_model).size
            global_grads = np.zeros(model_len)
            client_grads = np.zeros(model_len)
            num_samples = []
            local_grads = []

            for c in self.clients:
                num, client_grad = c.get_grads(model_len)
                local_grads.append(client_grad)
                num_samples.append(num)
                global_grads = np.add(global_grads, client_grad * num)
            global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))
            global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

            difference = 0
            for idx in range(len(self.clients)):
                difference += np.sum(np.square(global_grads - local_grads[idx]))
            difference = difference * 1.0 / len(self.clients)
            tqdm.write('gradient difference: {}'.format(difference))
            gd.append(difference)

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)

            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.latest_model, self.client_model, self.show_grads())# 这里是模型的调用方法，将相关的参数调用给相关的模型When you call set_params, it will assign self.latest_model to v_star

            for idx, c in enumerate(selected_clients.tolist()):# 注意这里与FedAvg不同，当有被选中的不活跃设备，采取其他处理方式�?                # communicate the latest model
                c.set_params(self.latest_model)
                

                total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)

                # solve minimization locally
                if c in active_clients:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                else:
                    #soln, stats = c.solve_iters(num_iters=np.random.randint(low=1, high=total_iters), batch_size=self.batch_size)
                    soln, stats = c.solve_inner(num_epochs=np.random.randint(low=1, high=self.num_epochs), batch_size=self.batch_size)
                    
                # gather solutions from client
                csolns.append(soln)#获得所有本地模型的参数
        
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)
            '''  此块代码是对客户端所有阈值统计
            theall=0
            for (cnum, cweight) in csolns:# csolns 一共有78500个点                 
                for var in cweight:
                    c=np.sum(np.absolute(var)<=1e-2)
                    theall+=c  
            print('小于阈值数量点：',theall)
            sparse.append(theall)
            '''




            '''
            datanum=[]
            for (cnum, cweight) in csolns:
                for (cnum, cweight) in csolns:#每个cweight相当于一个客户端的所有参数，长度为2，分别包含一个784*10的梯度和784*1的bias
                    #fisher=np.dot(np.array(cweight[0]).flatten(),np.array(self.latest_model[0]).flatten())
                    #fisher=np.dot(np.append(np.array(cweight[0]).flatten(),np.array(cweight[1]).flatten()),np.append(np.array(cweight[0]).flatten(),np.array(cweight[1]).flatten()))

                    #增加一个新实验，bias也算入其中。计算fisher 完成，在nist上效果稍逊于不加
                    #调整数据集看fisher能够选择出来含有十种标签的，而不是两种标签的
                    #查看fisher是否会因为全局模型造成选择数据集有所偏好，偏好已经学到的就要适度调整，偏好未学到的那非常好，可以讲故事，设置成设备之间标签数量固定，前期先用单一标签训练，看设备选择策略。后期提供iid数据和单一标签看选择
                    #之所以选择数据量大的是因为数据多，梯度下降的快，信息量也多
                    #查看和全局模型做交集，效果怎么样

                    #fisher可以让数据量多的客户端学的差不多了，即梯度不再变化时，转向其他小任务

                    #查看文档 定时记录结果
                    #fishers.append(fisher)
                    datanum.append(cnum)
                select_csoln=[]
                select_device_oneround=[]
                findex=np.argsort(-np.array(datanum))
                #print('fishers信息:',fishers)
                for i1,fid in enumerate(findex):
                    #if(len(select_csoln)<10):
                    if(len(select_csoln)<10 and datanum[fid]<500000):
                        select_csoln.append(csolns[fid])
                        select_device_oneround.append(datanum[fid])
                        #print('选择的设备的fisher值为:',fishers[fid])
                        print('选择的设备数据集大小：',datanum[fid])#研究fisher是否和数据集大小存在一定关系
                    elif(len(select_csoln)==10):
                        break
                csolns=select_csoln
                select_device.append(select_device_oneround)
                #print('csolns的长度:',len(csolns))
            '''





            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)
            theall=0
            for  cweight in self.latest_model:# csolns 一共有7850个点                 
                for var in cweight:
                    c=np.sum(np.absolute(var)<=1e-1)
                    theall+=c  
            print('全局小于阈值数量点：',theall)
            sparse.append(theall)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()

        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)

        pd_acc=pd.DataFrame(data=list(acc))
        pd_acc.columns =['acc']
        
        pd_loss=pd.DataFrame(data=list(loss))
        pd_loss.columns =['loss']
        
        pd_gd=pd.DataFrame(data=list(gd))
        pd_gd.columns =['gd']

        #pd_sparse=pd.DataFrame(data=list(sparse))
        #pd_sparse.columns =['num']
        print(pd_acc)
        pd_acc.to_csv(r'./res/acc/{}_drop{}_client{}_lr{}_acc_FedFTRL_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['lambda']), mode='a', index=None)
        pd_loss.to_csv(r'./res/loss/{}_drop{}_client{}_lr{}_loss_FedFTRL_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['lambda']), mode='a', index=None)
        #pd_gd.to_csv(r'./res/gd/{}_drop{}_client{}_lr{}_gd_FedFTRL_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['lambda']), mode='a', index=None)
        #pd_sparse.to_csv(r'./res/sparse/{}_drop{}_client{}_lr{}_sparse_FedFTRL_{}_l1.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['lambda']), mode='a', index=None)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
