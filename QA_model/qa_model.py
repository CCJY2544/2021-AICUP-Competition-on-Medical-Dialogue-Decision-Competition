import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer,BertModel,XLNetTokenizer, XLNetModel
from QA_model.encoder import Encoder
PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型

class QA_Classifier(nn.Module):
    def __init__(self, d_emb: int, p_hid: float, n_layers: int):
        super().__init__()
        self.l1 = nn.Linear(3*d_emb, d_emb)
        self.dropout = nn.Dropout(p_hid)

        hid = []
        for _ in range(n_layers):
            hid.append(nn.Linear(in_features=d_emb, out_features=d_emb))
            hid.append(nn.ReLU())
            hid.append(nn.Dropout(p=p_hid))
        self.hid = nn.Sequential(*hid)
        self.l2 = nn.Linear(d_emb, 1)

    def forward(
        self,
        document: torch.Tensor,
        question: torch.Tensor,
        choice: torch.Tensor
    ) -> torch.Tensor:
        # Concatenates `document embedding`, `question embedding`
        # and `choice embeding`
        # Input shape: `(B, E)`, `(B, E)`, `(B, E)`
        # Ouput shape: `(B, 3*E)`
        output = torch.cat((document, question, choice), -1)

        #　Linear layer
        # Input shape: `(B, 3*E)`
        # Ouput shape: `(B, E)`
        output = F.relu(self.l1(output))

        #　Dropout
        # Input shape: `(B, E)`
        # Ouput shape: `(B, E)`
        output = self.dropout(output)

        # Hidden layer
        output = self.hid(output)

        #　Linear layer
        # Input shape: `(B, E)`
        # Ouput shape: `(B, 1)`
        output = torch.sigmoid(self.l2(output))

        return output


class qa_model(nn.Module):
    def __init__(self, d_emb: int, n_layers: int, p_hid: float):
        super().__init__()
        self.encoder = Encoder(d_emb, p_hid)
        self.qa = QA_Classifier(d_emb, p_hid, n_layers)

    def forward(self, document, question, choice):

        
        # print(type(document))
        # print(document.dtype)
        # Document embedding
        s_mask = self.create_mask(document)
        # print(type(s_mask))
        # print(s_mask.dtype)
        doc = self.encoder(document, s_mask)

        # Sentence embedding
        s_mask = self.create_mask(question)
        qst = self.encoder(question, s_mask)
        
        # Shape: [3, B, E]
        print(choice.size())
        choice = choice.transpose(0, 1)
        print(choice.size())
        s_mask = self.create_mask(choice)
        qa_output = []
        for i in range(choice.shape[0]):
            chs_temp = self.encoder(choice[i], s_mask[i])
            qa_output.append(self.qa(doc, qst, chs_temp))
        qa_output = torch.cat(qa_output, dim=-1)

        return qa_output

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

    def loss_fn(self, document, question, choice, qa):
        pred_qa = self(document, question, choice)
        pred_qa = pred_qa.reshape(-1)
        qa = qa.reshape(-1)
        return F.binary_cross_entropy(pred_qa, qa)
