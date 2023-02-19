import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import numpy as np

#How to customize the optimizer
#https://towardsdatascience.com/custom-optimizer-in-tensorflow-d5b41f75644a

class PerturbedGradientDescent(optimizer.Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer"""
    def __init__(self, learning_rate=0.001, mu=0.01, xlambda=0.01, use_locking=False, name="PGD"):
        super(PerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._mu = mu
        self.all_grad=None
        self._lambda=xlambda
       
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._mu_t = None
        self._lambda_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._mu_t = ops.convert_to_tensor(self._mu, name="ftrl_mu")
        self._lambda_t = ops.convert_to_tensor(self._lambda, name="ftrl_lambda")

    def _create_slots(self, var_list):
        #Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)#var is local model，grad_all is global model，vstar is bar(g)^t
            self._zeros_slot(v, "grad_all", self._name)
    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        lambda_t = math_ops.cast(self._lambda_t, var.dtype.base_dtype)
        grad_all=self.get_slot(var,"grad_all")
        vstar = self.get_slot(var, "vstar")

        var_update = state_ops.assign_sub(var, lr_t*(grad + lambda_t*(vstar)+mu_t*(var-grad_all)))#var is local model，grad_all is global model，vstar is bar(g)^t

        return control_flow_ops.group(*[var_update,])

    
    def _apply_sparse_shared(self, grad, var, indices, scatter_add):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        v_diff = state_ops.assign(vstar, mu_t * (var - vstar), use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = state_ops.assign_sub(var, lr_t * scaled_grad)

        return control_flow_ops.group(*[var_update,])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(x, i, v))
    #'''   

    def set_params(self, cog, client,grad_list):#
        #grad_all=self.get_slot(grad_list[-1],"grad_all")
        
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            i=0
            global_grads=grad_list[-1]
            for variable, value in zip(all_vars, cog):

                '''
                if(i==0):
                    value1=np.array(grad_list[-1][0:7840]).reshape(784,10)  #numpy.array(a).reshape(len(a),1) 
                else:
                    value1=np.array(grad_list[-1][7840:]).reshape(10)
                '''
                
                
                value1=np.zeros_like(value)
                if(len(value1.shape)==1):
                    value1=np.array(global_grads[0:value1.shape[0]]).reshape(value1.shape[0])
                    global_grads=np.delete(global_grads,np.s_[0:value1.shape[0]])
                    
                elif(len(value1.shape)==2):
                    value1=np.array(global_grads[0:value1.shape[0]*value1.shape[1]]).reshape(value1.shape[0],value1.shape[1])
                    global_grads=np.delete(global_grads,np.s_[0:value1.shape[0]*value1.shape[1]])
                else:
                    raise Exception('Length less than 1 or greater than 3 cannot be processed')
                
                grad_all = self.get_slot(variable, "grad_all")
                grad_all.load(value, client.sess)        

                vstar = self.get_slot(variable, "vstar")
                vstar.load(value1, client.sess)
                i+=1
    '''
    def set_params(self, cog, client,grad_list):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, cog):
                vstar = self.get_slot(variable, "vstar")
                vstar.load(value, client.sess)
    '''