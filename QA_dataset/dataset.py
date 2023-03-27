import numpy as np
import csv
import json
import unicodedata
import re
import pickle
import random
import os
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

# def split_sent(sentence: str):
#     result =[]
#     first_role_idx = re.search(':', sentence).end(0)
#     out = [sentence[:first_role_idx]]
#     tmp = sentence[first_role_idx:]
#     while tmp:
#         res = re.search( r'(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:)', tmp)
#         if res is None:
#             break
#         idx = res.start(0)
#         idx_end = res.end(0)
#         result.append(out[-1] + tmp[:idx])
#         out[-1] = list(out[-1] + tmp[:idx])
#         out.append(tmp[idx:idx_end])
#         tmp = tmp[idx_end:]
#     return result
def split_sent(sentence: str):
    first_role_idx = re.search(':', sentence).end(0)
    out = [sentence[:first_role_idx]]

    tmp = sentence[first_role_idx:]
    while tmp:
        res = re.search(
            r'(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:)', tmp)
        if res is None:
            break

        idx = res.start(0)
        idx_end = res.end(0)
        out[-1] = out[-1] + tmp[:idx]
        out.append(tmp[idx:idx_end])
        tmp = tmp[idx_end:]

    out[-1] = out[-1] + tmp

    return out

def _read_qa(qa_file: str):
    qa = {}
    # [Question, [[Choice_1, Answer_1], [Choice_2, Answer_2], [Choice_3, Answer_3]]]
    for data in json.loads(unicodedata.normalize("NFKC", open(qa_file, "r", encoding="utf-8").read())):
        question = data["question"]
        if "answer" not in  data:
            choice_ans = [(
                choice["text"],
                int(choice["label"].strip() == "A")
            ) for choice in question["choices"]]
        else:
            choice_ans = [(
                    choice["text"],
                    int(choice["label"].strip() == data["answer"].strip())
                ) for choice in question["choices"]]
        question_text = list(question["stem"])
        aid = data["article_id"]
        if aid in qa:
            qa[aid][1].append((["".join(question_text)], choice_ans))
        else:
            qa[aid] = (split_sent(data['text'].replace(" ", "")), [(["".join(question_text) ], choice_ans)])
            

    return zip(*[v for _,v in sorted(qa.items(),key=lambda x:x[0])])

def encode_sent( sentence: str):
    output = []
    Encod = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    output=Encod.encode(sentence)

    return output

def encode_articles(article_text, max_doc_len, max_sent_len):
    Encod = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    article = []

    for document in article_text:
        # article.append([])
        if not document:
            continue
        hold=[]
        d=document[:max_doc_len]
        # print(d)
        hold.append(Encod.encode(d))
        

        doc_padding_size = max_doc_len - len(hold[0])
        h=[]
        h.append([])
        for i in range(doc_padding_size):
            h[0].append([np.float32(0)]*768)


        if doc_padding_size>0:
            a=np.concatenate((np.array(hold), np.array(h)), axis=1)
            article.append(a.squeeze(0))
        else:
            print(np.array(hold).shape)
            article.append(np.array(hold).squeeze(0))
            print(np.array(article).shape)
            
    return np.array(article)
# def encode_articles(article_text, max_doc_len):
#     Encod = SentenceTransformer('distiluse-base-multilingual-cased-v1')
#     article = []
#     for document in article_text:
#         article.append([])
#         for i, sentence in enumerate(document):
#             if i >= max_doc_len:
#                 break
#             article[-1].append([])
#             article[-1][-1] = Encod.encode(sentence)
            
#         doc_padding_size = max_doc_len - len(article[-1])
#         for i in range(doc_padding_size):
#             article[-1].append([])
#             article[-1][-1] = [np.float32(0)]*512
#             article[-1][-1] = np.array(article[-1][-1])
            
#     return np.array(article)

# def _read_qa(qa_file: str):
#     qa = {}
#     count = 0
#     # [Question, [[Choice_1, Answer_1], [Choice_2, Answer_2], [Choice_3, Answer_3]]]
#     for data in json.loads(unicodedata.normalize("NFKC", open(qa_file, "r", encoding="utf-8").read())):
#         question = data["question"]
#         if "answer" not in  data:
#             choice_ans = [(
#                 list(choice["text"]),
#                 int(choice["label"].strip() == "A")
#             ) for choice in question["choices"]]
#         else:
#             choice_ans = [(
#                 choice["text"],
#                 int(choice["label"].strip() == data["answer"].strip())
#             ) for choice in question["choices"]]
#         question_text =question["stem"]
#         aid = data["article_id"]
#         if aid in qa:
#             qa[aid][1].append((question_text, choice_ans))
#         else:
#             article_text = split_sent(data['text'].replace(" ", ""))
#             qa[aid] = (article_text, [(question_text, choice_ans)])
#     #         if len(article_text) > count:
#     #             count = len(article_text)
#     # print(count)

#     return zip(*[v for _,v in sorted(qa.items(),key=lambda x:x[0])])


# class dataset_qa(Dataset):
#     def __init__(
#         self,
#         qa_file: str,
#         max_doc_len: int = 170
#     ):
#         super().__init__()
#         Encod = SentenceTransformer('distiluse-base-multilingual-cased-v1')
#         article_text, qa_pairs = _read_qa(qa_file)
#         self.article = encode_articles(article_text, max_doc_len)

#         self.QA = []
#         for idx, qa_pair in enumerate(qa_pairs):
#             #print(idx)
#             for question, choice_ans in qa_pair:
#                 choice, ans = zip(*choice_ans)
#                 self.QA.append({
#                     "article_id":idx,
#                     "article": self.article[idx],
#                     "question": np.array(Encod.encode(question)),
#                     "choice": np.array([Encod.encode(x) for x in choice]),
#                     "qa_answer": np.array(ans),
#                 })


#     def __getitem__(self, idx: int):
#         return self.QA[idx]

#     def __len__(self):
#         return len(self.QA)

class dataset_qa(Dataset):
    def __init__(
        self,
        qa_file: str,
        max_sent_len: int = 60,
        max_doc_len: int = 380
    ):
        super().__init__()
        article_text, qa_pairs = _read_qa(qa_file)
        
        if os.path.exists('%s.pkl'%qa_file):
            with open('%s.pkl'%qa_file, 'rb') as f:
                self.QA = pickle.load(f)
        else:
            self.QA = []
            self.article = encode_articles(article_text, max_doc_len, max_sent_len)
            for idx, qa_pair in enumerate(qa_pairs):
                print(idx)
                for question, choice_ans in qa_pair:
                    choice, ans = zip(*choice_ans)
                    self.QA.append({
                        "article_id":idx,
                        "article": self.article[idx],
                        "question": np.array(encode_sent(question)),
                        "choice": np.array([encode_sent(x) for x in choice]),
                        "qa_answer": np.array(ans),
                    })
            random.shuffle(self.QA)
            with open('%s.pkl'%qa_file, 'wb') as f:
                pickle.dump(self.QA, f)

    def __len__(self):
        return len(self.QA)

    def __getitem__(self, idx: int):
        return self.QA[idx]