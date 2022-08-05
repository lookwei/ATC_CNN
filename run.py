# coding: UTF-8
import time
from train_eval import train, init_network
from importlib import import_module
import pandas as pd

if __name__ == '__main__':
    dataset = '.'
    embedding = 'embedding_SMILES_Vocab.npz'
    model_name = 'TextCnn'
    from utils import build_dataset, build_iterator, get_time_dif
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    K=config.K
    for i in range(K):
        start_time = time.time()
        vocab, train_data, dev_data, test_data = build_dataset(config,cross_validation_flag=True,i_th_flod=i,total_K_fold=K)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        init_network(model)
        train(config, model, train_iter, dev_iter, test_iter,flod_i=i)
    df=pd.read_csv('Res.csv',sep=',',header=None)
    ResList=[]
    for idx in range(5):
        ResList.append(df[idx].mean())
    print('(aim,con,acc,abst,absf)= ',ResList)