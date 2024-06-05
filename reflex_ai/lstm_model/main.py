import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from AnnoMI_data import AnnoMI_Dataset_lstm
from model import LSTMClassifier


train_dataset = AnnoMI_Dataset_lstm(filename="AnnoMI-processed_train.csv")
test_dataset = AnnoMI_Dataset_lstm(filename="AnnoMI-processed_test.csv")

# We are feeding one transaction at a time to LSTM
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
model_lstm = LSTMClassifier()
# Without weights for the class
# CE_loss = nn.CrossEntropyLoss(ignore_index=0)
# With weights for the class so as to mitigate the issue with the class imbalance
CE_loss = nn.CrossEntropyLoss(weight=torch.Tensor([116/544, 69/544, 145/544, 214/544])) # input1: batch x num_classes, input2: batch
# print(CE_loss)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.AdamW(model.parameters(), lr=0.0001)
optimizer = optim.AdamW(model_lstm.parameters(), lr=0.0005)
# optimizer = optim.SGD(model_lstm.parameters(), lr=0.01, momentum=0.9)
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

def train(epoch_i):
    optimizer.zero_grad()
    Avg_acc, Avg_loss = [], []
    for ind, (x,y) in enumerate(train_dataloader):
        o = model_lstm(x[0])
        loss = CE_loss(o.squeeze(0), y.squeeze(0))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), 0.01)
        #run this
        torch.nn.utils.clip_grad_value_(model_lstm.parameters(), 0.01)

        optimizer.step()
        optimizer.zero_grad()
        top1 = accuracy(o.squeeze(0), y.squeeze(0))
        Avg_acc.append(top1)
        Avg_loss.append(loss.item())
        print("Train Epoch %s Step %s Loss %s Top1 step acc %s Top1 Avg acc %s" %(epoch_i, ind, loss.item(), top1, np.mean(Avg_acc)))

    return Avg_loss, Avg_acc

def test(epoch_i):
    Avg_acc, Avg_loss = [], []
    for ind, (x,y) in enumerate(test_dataloader):
        with torch.no_grad():
            # o = model((x[0], x[1]))
            o = model_lstm(x[0])
            loss = CE_loss(o.squeeze(0), y.squeeze(0))
        top1 = accuracy(o.squeeze(0), y.squeeze(0))
        Avg_acc.append(top1)
        Avg_loss.append(loss.item())
        print("Test Epoch %s Step %s Loss %s Top1 step acc %s Top1 Avg acc %s" %(epoch_i, ind, loss.item(), top1, np.mean(Avg_acc)))
    return Avg_loss, Avg_acc

def loop():
    writer = SummaryWriter(log_dir=os.path.join('./lstm_exp_2_garbage', 'tensorboard'))
    for epoch_i in range(EPOCH):
        Loss, Acc = train(epoch_i)
        writer.add_scalar("Train_Loss", np.mean(Loss), epoch_i)
        writer.add_scalar("Train_Accuracy", np.mean(Acc), epoch_i)
        Loss, Acc = test(epoch_i)
        writer.add_scalar("Test_Loss", np.mean(Loss), epoch_i)
        writer.add_scalar("Test_Accuracy", np.mean(Acc), epoch_i)


if __name__ == "__main__":
    loop()
