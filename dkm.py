import argparse
import torch
from clustering_datasets import ClusteringDataset
from deep_k_means import DeepKMeans
from transformers import AutoModel


def calc_metrics(loader, n_clusters, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.cpu:
        device = torch.device('cpu')
    bert = AutoModel.from_pretrained('bert-base-uncased').to(device)
    DeepKMeans.set_grad(bert)
    print('K-Means on Embeddings:')
    model = DeepKMeans(bert, '', device, n_clusters, n_clusters, args.ae_epochs, args.cls_epochs).to(device)
    model.calc_kmeans(loader)
    model.fit(loader, val_lambda=args.lambda_, val=True)


parser = argparse.ArgumentParser(description="Deep k-means algorithm")
parser.add_argument("-d", "--dataset", type=str.upper, default='TWEETS',
                    help="Dataset on which DKM will be run\
                          (one of DBPEDIA, YELP, TREC, AGNEWS, SHORT, TWEETS), default: TWEETS",
                    required=False)
parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true', required=False)
parser.add_argument("-l", "--lambda", type=float, default=0.0001, dest="lambda_", required=False,
                    help="Value of the hyperparameter weighing the clustering\
                          loss against the reconstruction loss, default: 0.0001")
parser.add_argument("-a", "--ae_epochs", type=int, default=50, required=False,
                    help="Number of autoencoder training epochs, default: 50")
parser.add_argument("-e", "--cls_epochs", type=int, default=10, required=False,
                    help="Number of train clustering epochs per alpha value, default: 10")
parser.add_argument("-b", "--batch_size", type=int, default=100,
                    required=False, help="Size of batch, default: 100")
parser.add_argument("-t", "--max_len", type=int, default=128, required=False,
                    help="Size of max token len, default: 128")
args = parser.parse_args()

pre = ClusteringDataset(args.max_len, args.batch_size)

if args.dataset == 'DBPEDIA':
    print('============dbpedia============')
    loader, n_clusters = pre.init_dbpedia()
    calc_metrics(loader, n_clusters, args)

elif args.dataset == 'YELP':
    print('============yelp============')
    loader, n_clusters = pre.init_yelp()
    calc_metrics(loader, n_clusters, args)

elif args.dataset == 'TREC':
    print('============trec============')
    loader, n_clusters = pre.init_trec()
    calc_metrics(loader, n_clusters, args)

elif args.dataset == 'AGNEWS':
    print('============agnews============')
    loader, n_clusters = pre.init_agnews()
    calc_metrics(loader, n_clusters, args)

elif args.dataset == 'SHORT':
    print('============short============')
    loader, n_clusters = pre.init_short()
    calc_metrics(loader, n_clusters, args)

elif args.dataset == 'TWEETS':
    print('============tweets============')
    loader, n_clusters = pre.init_tweets()
    calc_metrics(loader, n_clusters, args)

else:
    parser.error("Unknown dataset!")
    exit()
