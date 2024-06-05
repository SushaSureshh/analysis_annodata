import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification, AdamW, BertConfig
from AnnoMI_data import AnnoMI_Dataset, collate_func
from model import Bert_wrapper



train_dataset = AnnoMI_Dataset(filename="AnnoMI-processed_train.csv")
test_dataset = AnnoMI_Dataset(filename="AnnoMI-processed_test.csv")
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_func)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False, collate_fn=collate_func)
# Load BertForSequenceClassification, the pretrained BERT model with a single 
# classification layer on top. 
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
#     num_labels = 4, # The number of expected labels.   
#     output_attentions = False, # If the model returns attentions weights.
#     output_hidden_states = False, # If the model returns all hidden-states.
# )
# Call the BERT wrapper
model = Bert_wrapper()

# Without weights for the class
CE_loss = nn.CrossEntropyLoss()

# With weights for the class
# CE_loss = nn.CrossEntropyLoss(weight=torch.Tensor([116/544, 69/544, 145/544, 214/544])) # input1: batch x num_classes, input2: batch

# SGD with mopmentum
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
 
 # optimizer 

# Note: AdamW is a class from the huggingface library 
# with weight decal regularization
# optimizer = AdamW(model.parameters(),
#                   lr = 2e-5, # default is 5e-5 as seen in the litrature, choosing
#                              # 2e-5
#                   eps = 1e-8 # default is 1e-8.
#                 )
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
EPOCH=10

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(e):
    optimizer.zero_grad()
    Avg_acc, Avg_loss = [], []
    for ind, (x,y) in enumerate(train_dataloader):
        # print(x)
        # print(x[0].shape, x[1].shape, x[2].shape)
        o = model((x[0], x[1]))
        # print(o)
        # print("output shape", o.shape)
        loss = CE_loss(o, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        top1 = accuracy(o, y)
        Avg_acc.append(top1)
        Avg_loss.append(loss.item())
        print("Train Epoch %s Step %s Loss %s Top1 step acc %s Top1 Avg acc %s" %(e, ind, loss.item(), top1, np.mean(Avg_acc)))
        # print("loss at step",loss.item(), ind)
        # print("Accurace at step:", train_accuracy)

    return Avg_loss, Avg_acc
#TODO Test the F1 scores for each class not the accuracy
def test(e):
    Avg_acc, Avg_loss = [], []
    for ind, (x,y) in enumerate(test_dataloader):
        with torch.no_grad():
            o = model((x[0], x[1]))
            loss = CE_loss(o, y)
        top1 = accuracy(o, y)
        Avg_acc.append(top1)
        Avg_loss.append(loss.item())
        print("Test Epoch %s Step %s Loss %s Top1 step acc %s Top1 Avg acc %s" %(e, ind, loss.item(), top1, np.mean(Avg_acc)))
    return Avg_loss, Avg_acc

def loop():
    writer = SummaryWriter(log_dir=os.path.join('./bert_exp_without_weights_2', 'tensorboard'))
    for e in range(EPOCH):
        Loss, Acc = train(e)
        writer.add_scalar("Train_Loss", np.mean(Loss), e)
        writer.add_scalar("Train_Accuracy", np.mean(Acc), e)
        Loss, Acc = test(e)
        writer.add_scalar("Test_Loss", np.mean(Loss), e)
        writer.add_scalar("Test_Accuracy", np.mean(Acc), e)


if __name__ == "__main__":
    loop()
