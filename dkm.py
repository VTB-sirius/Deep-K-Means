import argparse
import torch
from clustering_datasets import ClusteringDataset
from deep_k_means import DeepKMeans
from transformers import AutoModel
from metrics import Metric as metric
import numpy as np
from sklearn.cluster import KMeans


def calc_kmeans(loader, n_clusters, bert):
    with torch.no_grad():
        y_true = []
        X = []
        for i, batch in enumerate(loader):
            input_, mask, validation_target = batch
            y_true.append(validation_target.cpu().detach().numpy())
            X.append(DeepKMeans.mean_pooling(
                bert(input_.to(device),
                mask.to(device)).last_hidden_state,
                mask.to(device)).cpu().detach().numpy()
            )
        y_pred = KMeans(n_clusters=n_clusters, init="k-means++").fit_predict(np.concatenate(X))
        print("Validation ACC", metric.cluster_accuracy(y_true, y_pred))
        print("Validation ARI", metric.ar(y_true, y_pred))
        print("Validation NMI", metric.nmi(y_true, y_pred))
        print("Validation purity", metric.calculate_purity(y_true, y_pred))

parser = argparse.ArgumentParser(description="Deep k-means algorithm")
parser.add_argument("-d", "--dataset", type=str.upper, default='TWEETS',
                    help="Dataset on which DKM will be run (one of DBPEDIA, YELP, TREC, AGNEWS, SHORT, TWEETS)",
                    required=False)
parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true', required=False)
parser.add_argument("-l", "--lambda", type=float, default=0.0001, dest="lambda_", required=False,
                    help="Value of the hyperparameter weighing the clustering loss against the reconstruction loss")
parser.add_argument("-a", "--ae_epochs", type=int, default=50, required=False,
                    help="Number of autoencoder training epochs")
parser.add_argument("-e", "--cls_epochs", type=int, default=5, required=False,
                    help="Number of train clustering epochs per alpha value")
parser.add_argument("-b", "--batch_size", type=int, default=100, required=False, help="Size of batch")
parser.add_argument("-t", "--max_len", type=int, default=128, required=False, help="Size of max token len")
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if parser.cpu:
    device = torch.device('cpu')
pre = ClusteringDataset(parser.max_len, parser.batch_size)
bert = AutoModel.from_pretrained('bert-base-uncased').to(device)
DeepKMeans.set_grad(bert)

if parser.dataset == 'DBPEDIA':
    print('============dbpedia============')
    loader, n_clusters = pre.init_dbpedia()
    calc_kmeans()
    model1 = DeepKMeans(bert,  '', device,   n_clusters, n_clusters, parser.ae_epochs, parser.cls_epochs).to(device)
    model1.fit(loader,  val_lambda=parser.lambda_, val=True)
elif parser.dataset == 'YELP':
    print('============yelp============')
    loader, n_clusters = pre.init_yelp()
    calc_kmeans(loader, n_clusters, bert)
    model = DeepKMeans(bert,  '', device,   n_clusters, n_clusters, parser.ae_epochs, parser.cls_epochs).to(device)
    model.fit(loader,  val_lambda=parser.lambda_, val=True)
elif parser.dataset == 'TREC':
    print('============trec============')
    loader, n_clusters = pre.init_trec()
    calc_kmeans(loader, n_clusters, bert)
    model = DeepKMeans(bert,  '', device,   n_clusters, n_clusters, parser.ae_epochs, parser.cls_epochs).to(device)
    model.fit(loader, val_lambda=parser.lambda_, val=True)
elif parser.dataset == 'AGNEWS':
    print('============agnews============')
    loader, n_clusters = pre.init_agnews()
    calc_kmeans(loader, n_clusters, bert)
    model = DeepKMeans(bert,  '', device,   n_clusters, n_clusters, parser.ae_epochs, parser.cls_epochs).to(device)
    model.fit(loader,  val_lambda=parser.lambda_, val=True)
elif parser.dataset == 'SHORT':
    print('============short============')
    loader, n_clusters = pre.init_short()
    calc_kmeans(loader, n_clusters, bert)
    model = DeepKMeans(bert,  '', device,  n_clusters, n_clusters, parser.ae_epochs, parser.cls_epochs).to(device)
    model.fit(loader,  val_lambda=parser.lambda_, val=True)
elif parser.dataset == 'TWEETS':
    print('============tweets============')
    loader, n_clusters = pre.init_tweets()
    calc_kmeans(loader, n_clusters, bert)
    model = DeepKMeans(bert, '', device,  n_clusters, n_clusters, parser.ae_epochs, parser.cls_epochs).to(device)
    model.fit(loader,  val_lambda=parser.lambda_, val=True)
else:
    parser.error("Unknown dataset!")
    exit()