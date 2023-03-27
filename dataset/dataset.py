import numpy as np
import csv
import json
import unicodedata
import re
import os 
import pickle
import torch 
import time
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

import sys 
def split_sent(sentence: str):
    result =[]
    first_role_idx = re.search(':', sentence).end(0)
    out = [sentence[:first_role_idx]]
    tmp = sentence[first_role_idx:]

    while tmp:
        res = re.search( r'(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:)', tmp)
        if res is None:
            break
        idx = res.start(0)
        idx_end = res.end(0)
        result.append(out[-1] + tmp[:idx])
        # print(result[-1])
        out[-1] = list(out[-1] + tmp[:idx])
        out.append(tmp[idx:idx_end])
        tmp = tmp[idx_end:]
    # print(result)
    return result
    
# def encode_articles(article_text, max_doc_len):
#     # Encod = SentenceTransformer('distiluse-base-multilingual-cased-v1')
#     Encod = SentenceTransformer('distiluse-base-multilingual-cased-v1')
#     article = []
#     print(len(article_text))
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
#             # article[-1][-1] = Encod.encode("[PAD]")
#             article[-1][-1] = [np.float32(0)]*512
#             # article[-1][-1] = Encod.encode("")
#             article[-1][-1] = np.array(article[-1][-1])
#             # article[-1][-1]=double(article[-1][-1])
        
#     # print(article[0][0])
#     # print(article[0][-1])
#     # print(type(article))
#     # print(len(article))
#     # print(article[0][0].dtype)
#     # print(article[0][-1].dtype)
#     # for a in article:
#     #     for l in a:
#     #         if l.dtype!=article[0][0].dtype:
#     #             print(l.dtype)
#     # print(len(article[0]))
#     # print(len(article[0][-1]))
#     # sys.exit()
#     # print(type(article))
#     print(np.array(article).shape) 
#     print(np.array(article).dtype) 
#     return np.array(article)
#     # return torch.from_numpy(np.ndarray(article))

def encode_articles(article_text, max_doc_len):
    # Encod = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    # Encod = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    Encod = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    article = []
    c=0
    for document in article_text:
        # article.append([])
        print(c)
        c+=1
        if not document:
            continue
        hold=[]
        d=document[:max_doc_len]
        # print(d)
        hold.append(Encod.encode(d))
        # print(hold[-1])
        # sys.exit()
        # for i, sentence in enumerate(document):

        #     if i >= max_doc_len:
        #         break
        #     hold.append(Encod.encode(sentence))

        # print(np.array(hold).shape)
        

        doc_padding_size = max_doc_len - len(hold[0])
        h=[]
        h.append([])
        for i in range(doc_padding_size):
            h[0].append([np.float32(0)]*768)

        # print(np.array(h).shape)
        if doc_padding_size>0:
            a=np.concatenate((np.array(hold), np.array(h)), axis=1)
        

        
            article.append(a.squeeze(0))
        else:
            # print(np.array(hold).shape)
            article.append(np.array(hold).squeeze(0))
            # print(np.array(article).shape)



    # print(np.array(article).shape)
    return np.array(article)

def _read_risk(risk_file: str):
    article = []
    risk = []
    # [[Sent_1], [Sent_2], ..., [Sent_n]]
    for i, line in enumerate(csv.reader(open(risk_file, "r", encoding="utf-8"))):
        if i == 0:
            continue
        text = unicodedata.normalize("NFKC", line[2]).replace(" ", "")
        text = text.replace(":嗯哼。", ":好。").replace(":OK。", ":好。").replace(":齁。", ":好。")
        text = text.replace(":欸。", ":好。").replace(":嘿。", ":好。").replace("OK", "好")
        text = text.replace("恩哼", "好").replace("恩亨", "好").replace("嗯亨", "好").replace("嗯哼", "好")
        text = text.replace("好，好", "好").replace("好啦", "好").replace("好喔", "好").replace("好好", "好")
        text = text.replace("對，對", "對").replace("對對", "對").replace("痾", "").replace("阿", "")
        text = text.replace("是是", "是").replace("對對", "對").replace("...", "").replace("阿", "")
        text = text.replace("嘿，", "，").replace("嘿嘿", "嘿").replace("哈哈", "哈").replace("嗯嗯", "嗯")
        text = text.replace("齁齁", "齁").replace("有有", "有").replace("，。", "。").replace("，，", "，")
        text = text.replace("欸，", "，").replace("，欸", "，").replace("欸。", "。").replace("。欸", "。")
        # print(text)
        article.append(split_sent(text))
        # print(len(line))
        if len(line)<4:
            risk.append(0)
        else:
            risk.append(int(line[3]))
    #print(article[0])
    return article, risk

# def _read_risk(risk_file: str):
#     article = []
#     risk = []
#     dic={"感染","性行為","血","痛","有沒有","抽血","檢查","狀況","關係","固定","PrEP","藥物","會不會","回診","擔心","指數","舒服","功能","差","預防","肝","梅毒","發炎","治療","病毒","任務型","伴侶","發燒","報告","抗生素",
#     "忘記","打針","血糖","腎","住院","腎臟","抗體","過敏","控制","細菌","嚴重","要不要","追蹤","以後","結果","到時候","穩定","血壓","感冒","接下來","特別","隔天","超音波","自費","B肝","學名藥","影響","累","要不然","匿篩","濃度","疫苗","免疫","神經","情況","不一定","門診","HIV","學名","機會","主要","處理","完全",
#     "單子","胃","固炮","保護力","效果","戴套","有效","愛滋","增加","注意","抽血單","症狀","原因","副作用","體重","病毒量","運動","藥膏","腫","超過","咳嗽","風險","測","發現","狀態","肚子","漏","皮膚","開刀","喉嚨","劑","眼睛","細胞","肝炎","免疫力","痰","吐","流感","急診","白血球","胖","體質","篩檢","C肝","傷口",
#     "性病","公斤","胃藥","咳","檢驗","A肝","癢","血管","疾病","飲食","接觸","喘","頻率","HPV","X光","預計","復發","精神","調整","連續","暴露","觀察","反映","尿","心臟","類固醇","百分之百","掛號","菜花","影響到","B型","原廠藥","傳染","頭暈","成分","手術","止痛","證明","抽完","睡眠","抵抗力","體溫","注射","血液",
#     "劑量","個案","看診","預約","空腹","痠","氣喘","膽固醇","肌肉","息肉","鼻塞","斷層","水泡","變化","骨頭","血色素","慢性","長期","疹子","抗藥性","健保","糖尿病","傳染病","暈","吸收","循環","脹氣","尿酸","黴菌","血脂","評估","敏感","肺","病房","出院","服藥","痘痘","皮膚科","積水","普拿疼","糖化","酒","心跳",
#     "病人","拉肚子","肝臟","肺炎","復健","頭痛","淋病","Ｂ肝","陽性","紅色","定期","疫情","保險","健保卡","麻醉","核磁共振","三酸甘油脂","降低","後天","明天","預期","泌尿科","維他命","甲狀腺","益生菌","診斷","危險","鼻水","鼻涕","口交","典型","膠囊","同意書","代謝","頻繁","消炎","紅紅","小便","血小板","照舊","回升","輕微",
#     "安排","貧血","蜂窩性","尿道","顏色","異常","骨質","高蛋白","營養","內科","CD4","上升","焦慮","鑑定","嘴巴","處方簽","障礙","芒果","陰性","中樞","瘀青","退燒","白色","組織炎","雷射","因素","安眠藥","破皮","退掛","膽","淋巴球","胰島素","抗原","大腸鏡","鼻子","無套","高血壓","驗血","數量","癌症","療程","瓣膜","膽囊",
#     "感染科","鼻竇炎","止痛藥","腰椎","器官","黏膜","根治","藥效","急性","腹水","血紅素","增生","良性","留意","昏倒","刺激","泌尿道","肛門","嗅覺","結石","四環素","胸椎","疲勞","流膿","化療","磷","精神科","慢性病","分泌物","帶原者","局部","結核菌","腹部","膝蓋","淋巴結","大腸癌","蕁麻疹","抗組織胺","眼科","白內障","玻尿酸",
#     "披衣菌","漏藥","基因","電燒","腸胃","消毒","婦產科","心臟科","紅斑性","骨科","單核球","腦部","痔瘡","肺結核","萎縮","排便","腹腔鏡","肌肝酸","關節","胃潰瘍","視力","曲張","分枝桿菌","葡萄糖","肝膿瘍","內視鏡","康瑞斯","維生素","懷孕","尿管","胃乳","肝藥","轉移","突變","口腔","冷凍","活性","A型","腫腫","支氣管","蛋白尿",
#     "坐骨","C型","扁桃腺","骨髓","殺菌","異狀","切掉","化膿","葡萄球菌","病變","轉銜單","強烈","檢疫","鼻炎","攝護腺","肌酐酸","脾臟","肝硬化","肝病","長效型","抑制","靜脈","三酸甘油酯","惡化","眼球","脫皮","減肥","節食","骨髓炎","復健科","軟便","滲水","恙蟲病","胃食道","弱視","肝片","毒品"}
#     # [[Sent_1], [Sent_2], ..., [Sent_n]]
#     flag=0
#     another=0
#     count =0
#     for i, line in enumerate(csv.reader(open(risk_file, "r", encoding="utf-8"))):
#         if i == 0:
#             continue
#         text = unicodedata.normalize("NFKC", line[2]).replace(" ", "")
#         # print(text)
#         # print(split_sent(text))
#         hold=split_sent(text)
#         hold_ar=[]
#         for sen in hold:

#             for i in dic:
#                 # print(sen)
#                 if i in sen:
#                     flag=1
#                     another=1
#                     break
#             # print(flag,another)
#             if flag==1 or another==1:
#                 hold_ar.append(sen)
                
#             if flag==0:
#                 another=0
#             flag=0
#         if len(hold_ar)>count:
#             count = len(hold_ar)
#         article.append(hold_ar)
#         # print(hold)
#         if not line[3]:
#             risk.append(0)
#         else:
#             risk.append(int(line[3]))
#         # print(int(line[3]))
#         # sys.exit()
#     print(count)
#     return article, risk

class dataset_risk(Dataset):
    def __init__(
        self,
        risk_file: str,
        max_doc_len: int = 380,
    ):
        super().__init__()
        article_text, risk = _read_risk(risk_file)
        print(risk_file)
        if os.path.exists('%s.pkl'%risk_file):
            print("good")
            with open('%s.pkl'%risk_file, 'rb') as f:
                self.article = pickle.load(f)
        else:
            self.article = encode_articles(article_text, max_doc_len)
            # with open('%s.pkl'%risk_file, 'wb') as f:
            #     pickle.dump(self.article, f)

        # self.article = encode_articles(article_text, max_doc_len)
        self.risk = np.array(risk, dtype=np.float32)
        
    def __getitem__(self, idx: int):
        return {"article": self.article[idx], "risk_answer": self.risk[idx]}

    def __len__(self):
        return len(self.risk)