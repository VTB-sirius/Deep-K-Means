import numpy as np
import torch
import torch.nn.functional as F
from autoencoder import AutoEncoder
from metrics import Metric as metric
from sklearn.cluster import KMeans
from torch import nn
from torch import optim


class DeepKMeans(nn.Module):
    def __init__(self, bert, tokenizer, device, embedding_size, n_clusters, n_epoch, cls_n_epoch):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_size = embedding_size
        self.n_clusters = n_clusters
        self.n_epoch = n_epoch
        self.cls_n_epoch = cls_n_epoch
        self.input_size = self.bert.config.hidden_size
        self.ae = AutoEncoder(self.input_size, embedding_size, n_clusters, [500, 500, 2000])
        self.optimizer = optim.AdamW(self.ae.parameters())
        self.criterion = nn.L1Loss().to(device)

    @staticmethod
    def set_grad(net, require=False):
        for param in net.parameters():
            param.requires_grad = require

    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        sentence_embedding = self.mean_pooling(output.last_hidden_state, attention_mask)
        ae_embedding, reconstructed = self.ae(sentence_embedding)
        return ae_embedding, sentence_embedding, reconstructed

    def get_centers(self, loader, val):
        auto_encoder_embeddings = []
        for i, batch in enumerate(loader):
            input_, mask, _ = batch
            auto_encoder_embedding, bert_output, x = self.forward(input_.to(self.device), mask.to(self.device))
            auto_encoder_embeddings.append(auto_encoder_embedding.cpu().detach().numpy())
        auto_encoder_embeddings = np.concatenate(auto_encoder_embeddings)
        k_means = KMeans(n_clusters=self.n_clusters, init="k-means++").fit(auto_encoder_embeddings)
        if val:
            print('K-Means on AutoEncoder')
            self.validation(loader, kmeans=k_means)
        self.ae.a_enc_centers.copy_(torch.as_tensor(k_means.cluster_centers_, dtype=torch.float).to(self.device))

    def get_cluster(self, input_, mask):
        auto_encoder_embedding, _, _ = self.forward(input_, mask)
        diff = auto_encoder_embedding.unsqueeze(1) - self.ae.a_enc_centers.unsqueeze(0)
        diff = torch.sum(diff ** 2, dim=-1).unsqueeze(0)
        min_dist = torch.min(diff, dim=-1).indices
        return min_dist, F.softmax(diff, dim=-1)

    def get_clusters(self, loader):
        y_pred = []
        embs = []
        diffs = []
        y_true = []
        for batch in loader:
            input_, mask, validation_target = batch
            y_true.append(validation_target.cpu().detach().numpy())
            cls, diff = self.get_cluster(input_.to(self.device), mask.to(self.device))
            diffs.append(torch.squeeze(diff).cpu().detach().numpy())
            y_pred.append(cls.cpu().detach().numpy()[0])
            _, emb, _ = self.forward(input_.to(self.device), mask.to(self.device))
            embs.append(emb.cpu().detach().numpy())
        return np.concatenate(y_pred), np.concatenate(embs), np.concatenate(y_true)

    def validation(self, val_loader, kmeans=None):

        with torch.no_grad():
            self.eval()
            if kmeans is None:
                y_pred, _, y_true = self.get_clusters(val_loader)
            else:
                y_pred = []
                y_true = []
                for i, batch in enumerate(val_loader):
                    input_, mask, validation_target = batch
                    y_true.append(validation_target.cpu().detach().numpy())
                    auto_encoder_embedding, _, _ = self.forward(input_.to(self.device), mask.to(self.device))
                    y_pred.append(kmeans.predict(auto_encoder_embedding.cpu().detach().numpy()))
                y_pred = np.concatenate(y_pred)
                y_true = np.concatenate(y_true)
            print("Validation ACC", metric.cluster_accuracy(y_true, y_pred))
            print("Validation ARI", metric.ar(y_true, y_pred))
            print("Validation NMI", metric.nmi(y_true, y_pred))
            print("Validation purity", metric.calculate_purity(y_true, y_pred))
            if len(np.unique(y_pred)) != len(np.unique(y_true)):
                print(f'Есть вырожденный кластер. Всего кластеров {len(np.unique(y_pred))}')
            self.train()

    def cls_train(self, loader, val_lambda, val):
        for epoch in range(self.cls_n_epoch):
            print(f'Epoch: {epoch}')
            for i, batch in enumerate(loader):
                input_, mask, _ = batch
                auto_encoder_embedding, bert_output, auto_encoder_output =\
                    self.forward(input_.to(self.device), mask.to(self.device))
                diff = auto_encoder_embedding.unsqueeze(1) - self.ae.a_enc_centers.unsqueeze(0)
                diff = torch.sum(diff ** 2, dim=-1)
                kmeans_loss = torch.mean(torch.max(diff, dim=0).values)
                ae_loss = self.criterion(auto_encoder_output, bert_output)
                loss = ae_loss + val_lambda * kmeans_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1, norm_type=2)
                self.optimizer.step()
                self.optimizer.zero_grad()
            if val:
                self.validation(loader)
        return self.get_clusters(loader)

    def fit(self, loader, val_lambda=0.0001, val=False):
        print('Autoencoder training start')
        for epoch in range(self.n_epoch):
            print(f'Epoch: {epoch}')
            mean_loss = 0
            for i, batch in enumerate(loader):
                input_, mask, _ = batch
                auto_encoder_embedding, bert_output, auto_encoder_output =\
                    self.forward(input_.to(self.device), mask.to(self.device))
                loss = self.criterion(auto_encoder_output, bert_output)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                mean_loss += loss.item()
        print('Getting centers')
        with torch.no_grad():
            self.eval()
            self.get_centers(loader, val=val)
            self.train()
        print('Clusterer trainig start')
        self.cls_train(loader, val_lambda, val=val)
