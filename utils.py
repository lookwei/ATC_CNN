# coding: UTF-8
import os
import pickle as pkl
import time
from datetime import timedelta
import pandas as pd
import numpy as np
import torch
import requests, json

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

def get_splited_smis(filePath='./data/splited_smis.txt',smiList=None):
    if os.path.exists(filePath)==True:
        f = open(filePath,'r')
        splited_smis=[]
        while True:
            line = f.readline()
            if not line:
                break
            lineList=eval(line.split('\n')[0])
            splited_smis.append(lineList)
        f.close()
        return splited_smis
    else:
        gaode_url = 'http://47.243.95.243:8090/tokenizer/'
        data = json.dumps({'smi': str(smiList)})
        header = {"Accept": "application/json",
                  'Content-Type': 'application/json',
                  'charset': 'utf-8'}
        response = requests.get(url=gaode_url, data=data, headers=header)
        data_json = json.loads(response.text)
        splited_smis = (eval(data_json['smi']))
        return splited_smis

def getLableFromtxt1(config):
    lab=pd.read_csv(config.allfile_path,sep=',')['Lable']
    strArr=np.array(lab)
    alllableList=[]
    for s in strArr:
        lableList=eval(s)
        alllableList.append(lableList)
    return alllableList

def getLableFromtxt(config):
    lab=pd.read_csv(config.allfile_path,sep=',')['Lable']
    strArr=np.array(lab)
    alllableList=[]
    for s in strArr:
        lableList=eval(s)
        for ss in range(len(lableList)):
            lableList[ss]=float(lableList[ss])
        alllableList.append(lableList)
    return alllableList

def build_dataset(config,cross_validation_flag=True,i_th_flod=0,total_K_fold=10):

    allLableList = getLableFromtxt(config)
    smi_SpliVocList=get_splited_smis()

    f = open(config.vocab_path, 'rb')
    vocab_dic = pkl.load(f)

    sequenceWord2id_lable_seqLen=[]
    sequence_id=[]
    seq_len=[]

    if len(allLableList) != len(smi_SpliVocList):
        print('data erro!')
    for i in range(len(smi_SpliVocList)):
        smi_id=[]
        for j in range(len(smi_SpliVocList[i])):
            smi_id.append(vocab_dic[smi_SpliVocList[i][j]])
        while len(smi_id) < config.pad_size:#padding
            smi_id.append(vocab_dic['<PAD>'])
        sequence_id.append(smi_id)
        seq_len.append(len(smi_SpliVocList[i]))
        sequenceWord2id_lable_seqLen.append((smi_id,allLableList[i],len(smi_SpliVocList[i])))
    indices = list(range(len(sequenceWord2id_lable_seqLen)))

    train_indexList = None
    val_indexList = None
    test_indexList = None
    if cross_validation_flag == True:
        data_len = len(sequenceWord2id_lable_seqLen)
        fold_size = data_len // total_K_fold
        for j in range(total_K_fold):
            idx = slice((j * fold_size), ((j + 1) * fold_size))
            if data_len - ((j + 1) * fold_size) < 1:
                idx = slice((j * fold_size), data_len)
            partList = indices[idx]
            if j == i_th_flod:
                val_indexList = partList
            elif train_indexList is None:
                train_indexList = partList
            else:
                train_indexList = train_indexList + partList
        test_indexList = val_indexList.copy()
        print('val idx =',test_indexList)
    else:
        pass
        # train_indexList = indices[:3644]
        # val_indexList = indices[3644:4555]
        # test_indexList = indices[3644:4555]

    def get_train_val_test(indicesList,sequenceWord2id_lable_seqLen):
        returnList=[]
        for i in indicesList:
            returnList.append(sequenceWord2id_lable_seqLen[i])
        return returnList
    train = get_train_val_test(train_indexList,sequenceWord2id_lable_seqLen)
    val = get_train_val_test(val_indexList,sequenceWord2id_lable_seqLen)
    test = get_train_val_test(test_indexList,sequenceWord2id_lable_seqLen)
    return vocab_dic, train, val, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if self.n_batches==0:
            self.residue = True
        elif len(batches) % self.n_batches != 0:
            self.residue = True
        else:
            pass
        self.index = 0
        self.device = device
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
