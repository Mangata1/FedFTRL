
from http import client
from logging.config import DEFAULT_LOGGING_CONFIG_PORT
from xml.etree.ElementInclude import default_loader
import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
np.set_printoptions(threshold=1e6)

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        super(Server, self).__init__(params, learner, dataset)
        self.params=params

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        acc=[]
        loss=[]
        gd=[]
        maxacc=0
        clientNum=[]
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
            accloc=[]
            ii=0
            for ind in indices:
                if (maxacc<stats[3][ind]/stats[2][ind]):
                    maxacc=stats[3][ind]/stats[2][ind]
            for ind in indices:
                if(maxacc-stats[3][ind]/stats[2][ind]<0.05):
                    accloc.append(ii)
                    
                ii+=1
                if(len(accloc)==10):
                    break
            print('设备数：',len(accloc))
            clientNum.append(len(accloc))
            if(i>35):
                if(len(accloc)>0):
                    selected_clients=selected_clients[accloc]
                #现在是10个设备，可以采用设备选择算法，从20个里面挑10个最好的，性能肯定更好！
                else:
                    indices, selected_clients1 = self.select_clients(i, num_clients=10-len(accloc))
                    selected_clients=np.hstack((selected_clients[accloc],selected_clients1))
                    print('测试结果：',len(selected_clients))
            else:
                
                indices, selected_clients1 = self.select_clients(i, num_clients=10)
  

                

            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            #active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)
            active_clients = np.random.choice(selected_clients, round(len(accloc) * (1 - self.drop_percent)), replace=False)

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
                self.metrics.update(rnd=i, cid=c.id, stats=stats)#

            with open('weight.txt', 'w') as f:
                f.write(str(csolns))
                print(11)
            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

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

        pd_cn=pd.DataFrame(data=list(clientNum))
        pd_cn.columns =['Cient Number']
        print(np.array(pd_cn))
        pd_acc.to_csv(r'./res/test/{}_drop{}_client{}_lr{}_acc_FedFTRL_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['mu']), mode='a', index=None)
        pd_loss.to_csv(r'./res/test/{}_drop{}_client{}_lr{}_loss_FedFTRL_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['mu']), mode='a', index=None)
        pd_gd.to_csv(r'./res/test/{}_drop{}_client{}_lr{}_gd_FedFTRL_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['mu']), mode='a', index=None)
        pd_cn.to_csv(r'./res/test/{}_drop{}_client{}_lr{}_ClientNum_FedFTRL_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['mu']), mode='a', index=None)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
