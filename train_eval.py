# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter

def init_network(model, method='xavier', exclude='embedding', seed=42):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter, test_iter,flod_i):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    total_batch = 0
    dev_best_loss = float('inf')
    flag = False
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    bsestacc=[0,0,0,0,0]
    for epoch in range(config.num_epochs):
        lossSum=0.0
        model.train()
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains).to(config.device)
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            lossSum += loss.item()
            if total_batch % 50 == 0 :
                model.eval()
                with torch.no_grad():
                    true = labels.data.cpu()
                    lable_outputs = outputs.cpu()
                    for x in range(len(outputs)):
                        max_o = (torch.max(outputs[x]))

                        for idx in range(len(outputs[x])):
                            if outputs[x][idx] > 0 or outputs[x][idx] > max_o - 0.00001:
                                #print('outputs[x][idx] outputs[x]',outputs[x][idx],torch.min(outputs[x]))
                                lable_outputs[x][idx] = 1
                            else:
                                lable_outputs[x][idx] = 0
                    predic = lable_outputs
                    train_acc = Accuracy(np.array(predic.cpu()),np.array(true.cpu()))
                    tuple_acc_con_absf_abst_aim, dev_loss, Predication, lable_outputs_raw,sig = evaluate(config, model, dev_iter)

                    if tuple_acc_con_absf_abst_aim[2] > bsestacc[2]:
                        bsestacc=tuple_acc_con_absf_abst_aim
                        Best_Predicaation=Predication

                    if epoch==18 :
                        Best_Predicaation = Predication
                        bsestacc = tuple_acc_con_absf_abst_aim

                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0},  Train Loss: {1},  Train Acc: {2},  Val Loss: {3},  Val Acc: {4},  Time: {5} {6}'

                    if total_batch % 500 == 0:
                        print('Iter:',total_batch,'T-Loss',loss.item(), 'V-Loss',dev_loss, (time_dif))

                    if bsestacc[2] - 0.99 >= 0:
                        flag=True
                        break
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("acc/dev", tuple_acc_con_absf_abst_aim[0], total_batch)
                    model.train()
            total_batch += 1

        scheduler.step()
        if flag:
            break

    recoderFile = open('Res.csv', 'a')
    recoderFile.write(
        str(bsestacc[0]) + ',' +
        str(bsestacc[1]) + ',' +
        str(bsestacc[2]) + ',' +
        str(bsestacc[3]) + ',' +
        str(bsestacc[4]) +
        '\n')
    recoderFile.close()
    writer.close()

def Aiming(y_hat, y):
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y_hat[v])
    return sorce_k/n

def Coverage(y_hat, y):
    import numpy as np
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / sum(y[v])
    return sorce_k/n

def Accuracy(y_hat, y):
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        sorce_k += intersection / union
    return sorce_k/n

def AbsoluteTrue(y_hat, y):
    import numpy as np
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            sorce_k += 1
    return sorce_k/n

def AbsoluteFalse(y_hat, y):
    n, m = y_hat.shape
    sorce_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v,h] == 1 or y[v,h] == 1:
                union += 1
            if y_hat[v,h] == 1 and y[v,h] == 1:
                intersection += 1
        sorce_k += (union-intersection)/m
    return sorce_k/n

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    Predication_out=None
    predict_all = []
    true_labels_all = []
    lable_outputs_raw=None
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
            loss_total += loss
            labels = np.array(labels.data.cpu()).tolist()
            lable_outputs = outputs.cpu()
            lable_outputs_raw=outputs.cpu()
            sig=F.sigmoid(outputs.cpu())
            Sig_lable_outputs_raw=F.sigmoid(outputs.cpu())
            for x in range(len(sig)):
                max_o = (torch.max(sig[x]))
                for idx in range(len(sig[x])):
                    if sig[x][idx] > 0.5 or sig[x][idx] > max_o - 0.00001:
                        lable_outputs[x][idx] = 1
                    else:
                        lable_outputs[x][idx] = 0
            lable_outputs = np.array(lable_outputs).tolist()
            Predication_out=lable_outputs
            if len(labels)==len(lable_outputs):
                for i in range(len(labels)):
                    true_labels_all.append(labels[i])
                    predict_all.append(lable_outputs[i])
            else:
                print('erro evl!')

    acc = Accuracy(np.array(predict_all),np.array(true_labels_all))
    con = Coverage(np.array(predict_all),np.array(true_labels_all))
    absf = AbsoluteFalse(np.array(predict_all),np.array(true_labels_all))
    aim = Aiming(np.array(predict_all),np.array(true_labels_all))
    abst = AbsoluteTrue(np.array(predict_all),np.array(true_labels_all))
    if test:
        pass
    return (aim,con,acc,abst,absf), loss_total / len(data_iter),Predication_out,lable_outputs_raw,Sig_lable_outputs_raw