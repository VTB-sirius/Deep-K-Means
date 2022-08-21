# Deep-K-Means
## Description
Realization of deep k-means on pytorch https://arxiv.org/abs/1806.10069 </br>
Calculation of NMI, ARI, Cluster Accuracy, Purity metrics of Deep K-Means, K-Means on AutoEncoder (in epoch 0), K-Means on Ebeddings models.
## Installation
Run the command:\
``pip install -r requirements.txt``
## Usage
Deep k-Means (DKM) is run using the following command:
```dkm.py [-h] [-d <strig>] [-c] [-l <float>] [-a <int>] [-e <int>] [-b <int>] [-t <int>]```
The meaning of each argument is detailed below:
* ``-h``, ``--help``:           show this help message and exit
* ``-d <strig>``, ``--dataset <strig>``:
                       Dataset on which DKM will be run (one of DBPEDIA,
                       YELP, TREC, AGNEWS, SHORT, TWEETS), default: TWEETS
* ``-c``, ``--cpu``             Force the program to run on CPU
* ``-l <float>``, ``--lambda <float>``:
                       Value of the hyperparameter weighing the clustering
                       loss against the reconstruction loss, default: 0.0001
* ``-a <int>``, ``--ae_epochs <int>``:
                       Number of autoencoder training epochs, default: 50
* ``-e <int>``, ``--cls_epochs <int>``:
                       Number of train clustering epochs per alpha value,
                       default: 10
* ``-b <int>``, ``--batch_size <int>``:
                       Size of batch, default: 100
* ``-t <int>``, ``--max_len <int>``: 
                        Size of max token len, default: 128
## Contacts
* Telegram - [Yury_Sokolov](https://t.me/Yury_Sokolov)