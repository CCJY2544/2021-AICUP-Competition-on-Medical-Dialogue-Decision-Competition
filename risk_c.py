import torch
import json
import re
import numpy as np
import random
import os
import csv
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from model import risk_model
from dataset import dataset_risk
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer

def risk_train():
    # Hyperparameters
    args = {
        "batch_size": 4,
        "learning_rate": 1e-5,
        "random_seed": 42,
        "n_epoch": 47,
        "log_step": 5,
        "save_step": 100,
        "d_emb": 312,
        "n_cls_layers": 2,
        "p_drop": 0.001,
        "weight_decay": 0.0,
        "model_path": os.path.join("exp", "_risk", "2_14_47"),
        "embedding_path": os.path.join("data", "embeddings.npy"),
        "vocab_path": os.path.join("data", "vocab.json"),
        "log_path": os.path.join("log", "_risk", "_1"),
        "qa_data": os.path.join("data", "Train_qa_ans.json"),
        "risk_data": os.path.join("data", "Train_risk_classification_ans.csv"),
    }

    # Save training configuration
    if not os.path.isdir(args["model_path"]):
        os.makedirs(args["model_path"])
    with open(os.path.join(args["model_path"], "cfg.json"), "w") as f:
        json.dump(args, f)

    # Random seed
    random_seed = args["random_seed"]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:1')

    # Model
    model = risk_model(args['embedding_path'], args["d_emb"], args["n_cls_layers"], args["p_drop"])
    model = model.train()
    model = model.to(device)

    # Remove weight decay on bias and layer-norm.
    no_decay = ['bias', 'LayerNorm.weight']
    optim_group_params = [
        {
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': args["weight_decay"],
        },
        {
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]

    # Optimizer
    optimizer = torch.optim.AdamW( optim_group_params, lr=args["learning_rate"] )

    # Data
    data = dataset_risk(args['vocab_path'], args["risk_data"])
    print(data[0])
    dataldr = torch.utils.data.DataLoader(data, batch_size=args["batch_size"], shuffle=True)

    # Log writer
    if not os.path.isdir(args["log_path"]):
        os.makedirs(args["log_path"])
    writer = SummaryWriter(log_dir=args["log_path"])

    # Train loop
    step = 0
    avg_loss = 0
    for epoch in range(args["n_epoch"]):
        tqdm_dldr = tqdm(dataldr)

        for batch_data in tqdm_dldr:
            optimizer.zero_grad()
            print(type(batch_data))
            batch_document = batch_data["article"].to(device)
            batch_risk = batch_data["risk_answer"].to(device)
            # print(batch_document.size())
            # print(batch_risk.size())
            loss = model.loss_fn(batch_document, batch_risk)
            
            loss.backward()
            optimizer.step()

            step = step + 1
            avg_loss = avg_loss + loss

            if step % args["log_step"] == 0:
                avg_loss = avg_loss/args["log_step"]
                tqdm_dldr.set_description(f"epoch:{epoch}, loss:{avg_loss}")
                writer.add_scalar("loss", avg_loss, step)
                avg_loss = 0

            if step % args["save_step"] == 0:
                if not os.path.isdir(args["model_path"]):
                    os.makedirs(args["model_path"])
                torch.save(model.state_dict(), os.path.join(
                    args["model_path"], f"model-{step}.pt"))

    if not os.path.isdir(args["model_path"]):
        os.makedirs(args["model_path"])

    torch.save(model.state_dict(), os.path.join(
        args["model_path"], f"model-{step}.pt"))


def risk_test(exp_path: str):
    # Hyperparameters
    batch_size = 4
    with open(os.path.join(exp_path, "cfg.json"), "r") as f:
        args = json.load(f)
    # output_path = args["log_path"].replace("log", "output","4-14-47")
    output_path = "./output/_risk/2-14-47"
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:1')

    # Load checkpoints
    ckpts = []
    for ckpt in os.listdir(args["model_path"]):
        match = re.match(r'model-(\d+).pt', ckpt)
        if match is None:
            continue
        ckpts.append(int(match.group(1)))
    ckpts = sorted(ckpts)
    
    # Log writer
    writer = SummaryWriter(log_dir=args["log_path"])

    # Evaluate on training set
    print("Evaluate on training set...")

    # Data
    data = dataset_risk(args['vocab_path'], "data/dev_class.csv")
    #data = dataset_risk(args['vocab_path'], "data/test_class.csv")
    
    for ckpt in ckpts:
        # Model
        model = risk_model(args['embedding_path'], args["d_emb"], args["n_cls_layers"], 0.0)
        model.load_state_dict(torch.load(os.path.join(
            args["model_path"], f"model-{ckpt}.pt")))
        model = model.eval()
        model = model.to(device)

        dataldr = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False)
        tqdm_dldr = tqdm(dataldr)
        answer = {}
        answer["risk"] = data.risk
        pred = {"risk": []}

        for batch_data in tqdm_dldr:
            batch_document = batch_data["article"].to(device)
            pred_risk = model(batch_document).tolist()
            for temp in pred_risk:
                pred["risk"].append(temp)

        print(f"risk: {roc_auc_score(answer['risk'], pred['risk'])}")
        writer.add_scalar("train", roc_auc_score(answer['risk'], pred['risk']), ckpt)
        #write2file(output_path, pred['risk'], ckpt)

def write2file(output_path: str, data: list, ckpt: int):
    output = []
    output.append(["article_id", "label"])
    for i, label in enumerate(data):
        temp = [i+1, label]
        output.append(temp)
    with open(os.path.join(output_path, f"decision_{ckpt}.csv"), 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(output)

if __name__ == "__main__":
    risk_train()
    exp_path = os.path.join("exp", "_risk", "_1")
    #risk_test(exp_path)
