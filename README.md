# Improving Federated Learning Across Heterogeneous Clients with Follow the Regularized Leader



**Federated Learning** (FL) has two key implementation challenges in real networks due to its fundamental design assumptions. First, the assumption that the datasets across different devices follow the  independent and identical distribution (IID) (statistical heterogeneity) does not hold in many practical networks. Second, devices are seldom homogeneous rather they almost always have different capabilities (system heterogeneity).

In recent years, there have been several attempts to resolve these challenges. Some recent works employ restrictive metrics (e.g., local dissimilarity), while the others focus on minimizing the objective inconsistency. In this work, we take a major departure from existing works and propose  **FedFTRL** , to approximate the contribution estimates of local parameters and address the loss of other local participating devices, so that both performance and stability can be improved substantially. At the heart of our solution is the **FTRL** (a regularizer of FedFTRL), which can be plugged into several existing FL techniques to improve their optimization and stability. Our design assumptions and findings are validated extensively with experiments, and demonstrate that the proposed FedFTRL outperforms existing state of the art  FL solutions.

## General Guidelines

Note that if you would like to use FedFTRL as a baseline and run our code:

* If you are using different datasets, then at least the learning rates, lambda parameter and the mu parameter need to be tuned based on your metric. 

* If you are using the same datasets as those used here, then need to use the same learning rates lambda and mu reported in `README` files.

* Note that all parameter settings are based on the original papers of  [FedProx](https://arxiv.org/abs/1812.06127) and FedFTRL.


## Preparation

### Dataset generation

We use the dataset provided by [FedProx code](https://github.com/litian96/FedProx). FedProx  provide four synthetic datasets that are used in the paper under corresponding folders. For all datasets, see the `README` files in separate `data/$dataset` folders for instructions on preprocessing and/or sampling data.

The statistics of real federated datasets are summarized as follows.

<center>

| Dataset       | Devices         | Samples|Samples/device <br> mean (stdev) |
| ------------- |-------------| -----| ---|
| MNIST      | 1,000 | 69,035 | 69 (106)| 
| FEMNIST     | 200      |   18,345 | 92 (159)|
| Shakespeare | 143    |    517,106 | 3,616 (6,808)|
| Sent140| 772      |    40,783 | 53 (32)|

</center>

### Downloading dependencies

```
pip install -r requirements.txt  
```

## Run on synthetic federated data 
(1) You don't need a GPU to run the synthetic data experiments:

```
export CUDA_VISIBLE_DEVICES=
```

(2) The  results in the paper are obtained by running the instructions as follows, and the results will be stored as a csv file in the corresponding directory under the `res/` folder.

To make sure that FedFTRL as a generalized version of FedProx does improve performance, all parameters are set according to those provided by FedProx, except for those specific (lambda parameter) to FedFTRL.

### synthetic iid tasks

```python
#FedAvg.
python  -u main.py --dataset=synthetic_iid --optimizer=fedavg  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=0 --lambda=0
#FedProx. When lambda=0, FedFTRL degenerates to FedProx
python  -u main.py --dataset=synthetic_iid --optimizer=fedftrl  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 --lambda=0
#FedFTRL
python  -u main.py --dataset=synthetic_iid --optimizer=fedftrl  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 --lambda=2.0
```

### synthetic non-iid tasks


```python
python  -u main.py --dataset=synthetic_0_0 --optimizer=fedavg  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=0 --lambda=0
python  -u main.py --dataset=synthetic_0_0 --optimizer=fedftrl  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 --lambda=0
python  -u main.py --dataset=synthetic_0_0 --optimizer=fedftrl  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 --lambda=1.2
```
```python
python  -u main.py --dataset=synthetic_0.5_0.5 --optimizer=fedavg  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=0 --lambda=0
python  -u main.py --dataset=synthetic_0.5_0.5 --optimizer=fedftrl  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 --lambda=0
python  -u main.py --dataset=synthetic_0.5_0.5 --optimizer=fedftrl  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 --lambda=0.1
```
```python
python  -u main.py --dataset=synthetic_1_1 --optimizer=fedavg  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=0 --lambda=0
python  -u main.py --dataset=synthetic_1_1 --optimizer=fedftrl  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 --lambda=0
python  -u main.py --dataset=synthetic_1_1 --optimizer=fedftrl  --learning_rate=0.01 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 --lambda=0.04
```

## Run on real federated datasets
(1) Specify a GPU id if needed:

```
export CUDA_VISIBLE_DEVICES=available_gpu_id
```
Otherwise just run to CPUs [might be slow if testing on Neural Network models]:

```
export CUDA_VISIBLE_DEVICES=
```

(2) The  results in the paper are obtained by running the instructions as follows, and the results will be stored as a csv file in the corresponding directory under the `res/` folder.

To make sure that FedFTRL as a generalized version of FedProx does improve performance, all parameters are set according to those provided by FedProx, except for those specific (lambda parameter) to FedFTRL.

### Minst tasks

```python
python  -u main.py --dataset=mnist --optimizer=fedavg  --learning_rate=0.03 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=0 --lambda=0

python  -u main.py --dataset=mnist --optimizer=fedftrl  --learning_rate=0.03 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=0.8 -lambda=0

python  -u main.py --dataset=mnist --optimizer=fedftrl  --learning_rate=0.03 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=0.8 -lambda=0.8
```
### FeMinst tasks

```python

python  -u main.py --dataset=nist --optimizer=fedavg  --learning_rate=0.003 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=0 --lambda=0

python  -u main.py --dataset=nist --optimizer=fedftrl  --learning_rate=0.003 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 ---lambda=0

python  -u main.py --dataset=nist --optimizer=fedftrl  --learning_rate=0.003 --num_rounds=200 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=mclr  --drop_percent=0 --mu=1 ---lambda=0.04

```
### Sent140 tasks
```python
python  -u main.py --dataset=sent140 --optimizer=fedavg  --learning_rate=0.3 --num_rounds=800 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=stacked_lstm  --drop_percent=0 --mu=0 --lambda=0

python  -u main.py --dataset=sent140 --optimizer=fedftrl  --learning_rate=0.3 --num_rounds=800 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=stacked_lstm  --drop_percent=0 --mu=0.01 --lambda=0

python  -u main.py --dataset=sent140 --optimizer=fedftrl  --learning_rate=0.3 --num_rounds=800 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=stacked_lstm  --drop_percent=0 --mu=0.01 --lambda=0.01
```
### Shakespare tasks

The Shakespare task is the next word prediction task in NLP, and it is important to note that the task takes a long time to complete training before running. It is recommended to set a small number of num_rounds parameter.

```python
python  -u main.py --dataset=shakespeare --optimizer=fedavg  --learning_rate=0.8 --num_rounds=40 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=stacked_lstm  --drop_percent=0 --mu=0 --lambda=0

python  -u main.py --dataset=shakespeare --optimizer=fedftrl  --learning_rate=0.8 --num_rounds=40 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=stacked_lstm  --drop_percent=0 --mu=0.001 --lambda=0

python  -u main.py --dataset=shakespeare --optimizer=fedftrl  --learning_rate=0.8 --num_rounds=40 --clients_per_round=10  --eval_every=1 --batch_size=10  --num_epochs=20  --model=stacked_lstm  --drop_percent=0 --mu=0.001 --lambda=0.0001
```



