import torch
import torch.nn as nn
from model.encoder import Encoder
import torch.nn.functional as F
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型

class Risk_Classifier(nn.Module):
    def __init__(self, d_emb: int, p_hid: float, n_layers: int):
        super().__init__()
        
        self.l0 = nn.Linear(d_emb, 1)
        # self.l1 = nn.Linear(d_emb, 1)
        # self.l0 = nn.Linear(d_emb, 2)
        # self.l1 = nn.Linear(380, 1)
        self.dropout = nn.Dropout(p_hid)
        self.relu=nn.ReLU()
    def forward(self, document: torch.Tensor) -> torch.Tensor:
        output = document
        # print(output.size())
        output = self.dropout(output)
        # print(output.size())
        output = self.l0(output)
        # output=self.relu(output)
        # output = self.l1(output)
        # print(output.size())
        # output = output.squeeze(-1)
        # output = self.l1(output)

        return output.squeeze(-1)
        # return output

class risk_model(nn.Module):
    def __init__(self, embedding_path: str, d_emb: int, n_layers: int, p_hid: float):
        super().__init__()
        self.risk = Risk_Classifier(d_emb, p_hid, n_layers)
        self.encoder = Encoder(d_emb, p_hid)
        self.l0 = nn.Linear(d_emb, 1)
        self.relu=nn.ReLU()
        # Encode = SentenceTransformer('paraphrase-distilroberta-base-v1')
    def forward(self, document):
        # print(document)
        # print(document.size())
        # sys.exit()
        # temp = []
        # for i in range(document.shape[0]):
        #     temp.append(document[i])
        # doc = torch.stack(document)
        # print(document.size())
        # print(type(document))
        # Document embedding
        #(batch,380,786)

        # sm=nn.Softmax(dim=-1)
        # document_mask = self.l0(document)
        # document_mask=sm(document_mask)
        # # print(document_mask.size(),document.size())
        # document=torch.mul(document,document_mask)
        # # document=self.relu(document)
        # # print(document.size())
        # # sys.exit()

        smask=self.create_mask(document)
        document = self.encoder(document, smask)
        document=self.relu(document)
        risk_output = self.risk(document.float())

        risk_output=torch.sigmoid(risk_output)
        
        return risk_output
    def create_mask(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        # Create padding self attention masks.
        # Shape: [B, `max_doc_len`, `max_sent_len`, 1]
        # Output dtype: `torch.bool`.
        # w_pad_mask = batch_prev_tkids == 0
        # w_pad_mask = w_pad_mask.unsqueeze(-1)

        s_pad_mask = batch_prev_tkids.sum(dim=-1)
        s_pad_mask = s_pad_mask == 0
        s_pad_mask = s_pad_mask.unsqueeze(-1)

        return s_pad_mask
    def loss_fn(self, document, risk):
        pred_risk = self(document)
        pred_risk = pred_risk.reshape(-1)
        risk = risk.reshape(-1)
        # print(F.binary_cross_entropy(pred_risk, risk))
        return F.binary_cross_entropy(pred_risk, risk)
        # print(pred_risk.size(),risk.size())
        # print(risk)
        # loss=nn.CrossEntropyLoss()
        # return loss(pred_risk,risk.bool().long())
