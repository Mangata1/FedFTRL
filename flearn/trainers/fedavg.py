import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import pandas as pd
from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
np.set_printoptions(threshold=1e6)

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.compat.v1.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.params=params

    
    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        acc=[]
        loss=[]
        sparse=[]

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))
                acc.append(np.sum(stats[3]) * 1.0 / np.sum(stats[2]))
                loss.append(np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2]))

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            csolns = []  # buffer for receiving client solutions

            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)
            '''
            theall=0
            for (cnum, cweight) in csolns:# csolns 一共有78500个点                 
                for var in cweight:
                    c=np.sum(np.absolute(var)<=1e-2)
                    theall+=c  
            print('小于阈值数量点：',theall)
            sparse.append(theall)
            '''
            
            #with open('weight.txt', 'w') as f:
                #f.write(str(csolns))
            np.save('weight.npy',csolns)
            # update models
            self.latest_model = self.aggregate(csolns)
            theall=0
            for  cweight in self.latest_model:# csolns 一共有7850个点                 
                for var in cweight:
                    c=np.sum(np.absolute(var)<=1e-1)#sum(var**2)
                    theall+=c  
            print('小于阈值数量点：',theall)
            sparse.append(theall)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        
        pd_loss=pd.DataFrame(data=list(loss))
        pd_loss.columns =['loss']
        pd_loss.to_csv('./res/loss/{}_drop{}_client{}_lr{}_loss_FedAvg_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['mu']), mode='a', index=None)
        
        pd_acc=pd.DataFrame(data=list(acc))
        pd_acc.columns =['acc']
        pd_acc.to_csv(r'./res/acc/{}_drop{}_client{}_lr{}_acc_FedAvg_{}.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['mu']), mode='a', index=None)
        #pd_sparse=pd.DataFrame(data=list(sparse))
        #pd_sparse.columns =['num']
        #pd_sparse.to_csv(r'./res/sparse/server{}_drop{}_client{}_lr{}_sparse_FedAvg_{}_l1.csv'.format(self.params['dataset'],self.params['drop_percent'],self.params['clients_per_round'],self.params['learning_rate'],self.params['mu']), mode='a', index=None)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
