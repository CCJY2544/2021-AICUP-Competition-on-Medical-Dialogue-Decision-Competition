import torch
import json
import re
import numpy as np
import random
import os
import csv
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from QA_model import qa_model
from QA_dataset import dataset_qa
from torch.utils.tensorboard import SummaryWriter

'''
Here is the part of training the model, you need to set hyperparameters by yourself.
'''


def qa_train():
    # Hyperparameters
    args = {
        "batch_size": 4,
        "learning_rate": 1e-3,
        "random_seed": 42,
        "n_epoch": 30,
        "log_step": 15,
        "save_step": 100,
        "d_emb": 512,
        "n_cls_layers": 2,
        "p_drop": 0.0,
        "weight_decay": 0.0,
        "model_path": os.path.join("exp", "_qa", "0608"),
        "log_path": os.path.join("log", "_qa", "0608"),
        "qa_data": os.path.join("data", "QA_combine.json")
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
    # device = torch.device('cpu')
    # if torch.cuda.is_available():
    device = torch.device('cuda:1')

    print("Prepare model.............")
    # Model
    model = qa_model(args["d_emb"], args["n_cls_layers"], args["p_drop"])
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

    print("Prepare data.............")
    optimizer = torch.optim.AdamW(optim_group_params, lr=args["learning_rate"])
    print(args["qa_data"])
    data = dataset_qa(args["qa_data"])
    dataldr = torch.utils.data.DataLoader(data, batch_size=args["batch_size"], shuffle=True)

    # Log writer
    if not os.path.isdir(args["log_path"]):
        os.makedirs(args["log_path"])
    writer = SummaryWriter(log_dir=args["log_path"])

    print("Train loop.............")
    # Train loop
    step = 0
    avg_loss = 0
    for epoch in range(args["n_epoch"]):
        tqdm_dldr = tqdm(dataldr)

        for batch_data in tqdm_dldr:
            optimizer.zero_grad()
            batch_document = []
            #print(batch_data["article_id"])
            for idx in batch_data["article_id"]:
                batch_document.append(data.article[idx])

            batch_document = torch.LongTensor(batch_document).to(device)
            batch_question = batch_data["question"].to(device)
            batch_choice = batch_data["choice"].to(device)
            batch_qa_answer = batch_data["qa_answer"].float().to(device)

            loss = model.loss_fn(batch_document, batch_question,batch_choice, batch_qa_answer)
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


def qa_test(exp_path: str, Data_name:str):
    # Hyperparameters
    batch_size = 4
    with open(os.path.join(exp_path, "cfg.json"), "r") as f:
        args = json.load(f)
    output_path = args["log_path"]
    output_path = args["log_path"].replace("log", "output")
    
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # Device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:2')

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
    print("Evaluate on develop set...")
    data = dataset_qa(Data_name)

    for ckpt in ckpts:
        # Model
        model = qa_model(args['embedding_path'], args["d_emb"], args["n_cls_layers"], 0.0)
        model.load_state_dict(torch.load(os.path.join(
            args["model_path"], f"model-{ckpt}.pt")))
        model = model.eval()
        model = model.to(device)

        dataldr = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False)
        tqdm_dldr = tqdm(dataldr)
        answer = {"qa": []}
        pred = {"qa": []}

        for batch_data in tqdm_dldr:
            batch_document = []

            for idx in batch_data["article_id"]:
                #print(idx)
                batch_document.append(data.article[idx])

            batch_document = torch.LongTensor(batch_document).to(device)
            batch_question = batch_data["question"].to(device)
            batch_choice = batch_data["choice"].to(device)

            answer["qa"] = answer["qa"] + \
                batch_data["qa_answer"].argmax(dim=-1).tolist()

            pred_qa = model(batch_document, batch_question, batch_choice)

            pred["qa"] = pred["qa"] + pred_qa.argmax(dim=-1).tolist()
        if Data_name =="data/new_QA_dev.json":
            print(f"qa: {accuracy_score(answer['qa'], pred['qa'])}")
            writer.add_scalar("train", accuracy_score( answer['qa'], pred['qa']), ckpt)
        else:
            write2file(output_path, pred["qa"], ckpt)
        
def write2file(output_path: str, data: list, ckpt: int):
    output = []
    output.append(["article_id", "label"])
    for i, label in enumerate(data):
        if label == 0:
            label = "A"
        elif label ==1:
            label = "B"
        elif label ==2:
            label="C"
        temp = [i+1, label]
        output.append(temp)
    with open(os.path.join(output_path, f"decision_{ckpt}.csv"), 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(output)


if __name__ == "__main__":
    qa_train()
    qa_test(os.path.join("exp", "_qa", "0608"), "data/new_QA_dev.json")

