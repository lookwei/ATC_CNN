## [Identifying The Kind Behind SMILES – Anatomical Therapeutic Chemical Classification using Structure-Only Representations](https://doi.org/10.1093/bib/bbac346) 

ATC (Anatomical Therapeutic Chemical) classification for compounds/drugs plays an important role in drug development and basic research. However, previous methods depend on interactions extracted from STITCH dataset which may make it depends on lab experiments.

We present a pilot study to explore the possibility of conducting the ATC prediction solely based on the molecular structures. 

The motivation is to eliminate the reliance on the costly lab experiments so that the characteristics of a drug can be pre-assessed for better decision-making and effort-saving before the actual development.

1. we construct a new benchmark consisting of 4545 compounds which is with larger scale than the one used in previous study. 
2. A light-weight prediction model is proposed. The model is with better explainability in the sense that it is consists of a straightforward tokenization that extracts and embeds statistically and physicochemically meaningful tokens, and a deep network backed by a set of pyramid kernels to capture multi-resolution chemical structural characteristics. 

![image](https://github.com/lookwei/ATC_CNN/blob/main/ATC-CNN.png)



## Quick tour

This is a pytorch implementation of the ATC-CNN proposed in our paper [1].

To run the code, please make sure you have prepared canonical SMILES data ( *Computed* *by* *OEChem* *2.3.0* ) following the same structure as follows (you can also refer to our ATC-SMILES dataset in this repository):

../data/ATC_SMILES.csv        (ATC-SMILES dataset)



## Tokenization

SMILES Tokenization:

```
from Utils import get_splited_smis
#Use our ATC-SMILES dataset, the splited file has already existed,You can set the parameters as follows:
get_splited_smis(filePath='./data/splited_smis.txt',smiList=None)
#Use other canonical SMILES sequence, Please prepare the SMILES list as follows:
SMILES_List=['CCCC(=O)',...,'CCCC(=O)C']
get_splited_smis(filePath='',smiList=SMILES_List)
```



## Embedding

We adopt the Word2Vec and Skip-gram model to train token embeddings:

```
../models/TextCNN.py
self.embedding_pretrained=torch.tensor(np.load(dataset+'/data/embedding_SMILES_Vocab.npz')["embeddings"].astype('float32')) if embedding != 'random' else None
```



## Start training

To train a model with our ATC-SMILES dataset:

```
../run.py
python run.py
```



## Dataset

ATC-SMILES proposed in our paper [1].

```
../data/ATC_SMILES.csv
```

|      | KEGG_Drug_ID |                          CanSmiles                           |                   Lable                    |
| :--: | :----------: | :----------------------------------------------------------: | :----------------------------------------: |
|  0   |    C00018    |               CC1=NC=C(C(=C1O)C=O)COP(=O)(O)O                | [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
|  1   |    C00220    |   CN(C)C1=CC2=C(C=C1)N=<br/>C3C=CC(=[N+](C)C)C=C3S2.[Cl-]    | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] |
| ...  |     ...      |                             ...                              |                                            |
| 4544 |    D11817    | CNC(=O)C1=NN=C(C=<br/>C1NC2=CC=CC(=C2OC)C3=NN<br/>(C=N3)C)NC(=O)C4CC4 | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] |



## Citation

[1] Yi Cao, Zhen-Qun Yang, Xu-Lu Zhang, Wenqi Fan, Yaowei Wang, Jiajun Shen, Dong-Qing Wei, Qing Li, and Xiao-Yong Wei. Identifying The Kind Behind SMILES – Anatomical Therapeutic Chemical Classification using Structure-Only Representations,  *Briefings in Bioinformatics*, 2022, DOI:10.1093/bib/bbac346.

```bibtex
@ARTICLE{  
author={Yi Cao, Zhen-Qun Yang, Xu-Lu Zhang, Wenqi Fan, Yaowei Wang, Jiajun Shen, Dong-Qing Wei, Qing Li, and Xiao-Yong Wei.},  
journal={Briefings in Bioinformatics},   
title={Identifying The Kind Behind SMILES – Anatomical Therapeutic Chemical Classification using Structure-Only Representations},   
year={2022},  
volume={},  
number={},  
pages={},  
doi={DOI:10.1093/bib/bbac346}}
```
