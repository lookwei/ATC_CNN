## [Identifying The Kind Behind SMILES – Anatomical Therapeutic Chemical Classification using Structure-Only Representations](https://doi.org/10.1093/bib/bbac346) 

ATC (Anatomical Therapeutic Chemical) classification for compounds/drugs plays an important role in drug development and basic research. However, previous methods depend on interactions extracted from STITCH dataset which may make it depends on lab experiments.

We present a pilot study to explore the possibility of conducting the ATC prediction solely based on the molecular structures. The motivation is to eliminate the reliance on the costly lab experiments so that the characteristics of a drug can be pre-assessed for better decision-making and effort-saving before the actual development.

Our contributions are as follows:

1. We construct a new benchmark consisting of 4545 compounds which is with larger scale than the one used in previous study.
2. A light-weight prediction model is proposed. The model is with better explainability in the sense that it is consists of a straightforward tokenization that extracts and embeds statistically and physicochemically meaningful tokens, and a deep network backed by a set of pyramid kernels to capture multi-resolution chemical structural characteristics. 

For details, please refer to our paper [1], which will be available in *briefings in Bioinformatics* soon. Besides, an online ATC-codes predictor is available now.  http://aimars.net:8090/ATC_SMILES/

![image](/tokens.png)



## Quick tour

This is a pytorch implementation of the ATC-CNN proposed in our paper [1]. If you don't care details, just start training in ./run.py.

**[Step 1: Prepare dataset](#Datasets)**

To run the code, please make sure you have prepared canonical SMILES data ( *Computed* *by* *OEChem* *2.3.0* ) following the same data structure in ./data/ATC_SMILES.csv.

**[Step 2: Tokenization](#Tokenization)**

We have provided tokenized ATC-SMILES dataset, if you want to use other canonical SMILES, please use tokenization API in ./utils.py.

**[Step 3: Embedding](#Embedding)**

We adopt the Word2Vec and Skip-gram model to train token embeddings, you can use our pre-trained embeddings in ./data/embedding_SMILES_Vocab.npz.

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
./models/TextCNN.py
self.embedding_pretrained=torch.tensor(np.load(dataset+'/data/embedding_SMILES_Vocab.npz')["embeddings"].astype('float32')) if embedding != 'random' else None
```



## Start training

To train a model with our ATC-SMILES dataset:

```
./run.py
python run.py
```



## Datasets

**ATC-SMILES dataset proposed in our paper [1] with 4545 drugs/compounds.**

If you plan to use other smiles files, please refer to the data structure in [ATC_SMILES.csv](./data/ATC_SMILES.csv).

```
./data/ATC_SMILES.csv
```

|      | KEGG_Drug_ID |                          CanSmiles                           |                   Lable                    |
| :--: | :----------: | :----------------------------------------------------------: | :----------------------------------------: |
|  0   |    C00018    |               CC1=NC=C(C(=C1O)C=O)COP(=O)(O)O                | [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
|  1   |    C00220    |   CN(C)C1=CC2=C(C=C1)N=<br/>C3C=CC(=[N+](C)C)C=C3S2.[Cl-]    | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] |
| ...  |     ...      |                             ...                              |                    ...                     |
| 4544 |    D11817    | CNC(=O)C1=NN=C(C=<br/>C1NC2=CC=CC(=C2OC)C3=NN<br/>(C=N3)C)NC(=O)C4CC4 | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] |

**Chen-2012 dataset proposed in paper[2] with 3883 drugs/compounds.**

Chen-2012 dataset is available in  [Chen-2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0035254) .

**ATC-SMILES-Aligned dataset proposed in our paper [1] with 3785 drugs/compounds.**

ATC-SMILES is with a larger scale than that of Chen-2012 [2], but there are some mis-aligned items. We remove these items to generate a subset consisting of 3785 drugs/compounds, and we call this set ATC-SMILES-Aligned. ATC-SMILES-Aligned can be downloaded in [ATC-SMILES-Aligned.csv](./data/ATC-SMILES-Aligned.csv).


## Citation

[1] Yi Cao, Zhen-Qun Yang, Xu-Lu Zhang, Wenqi Fan, Yaowei Wang, Jiajun Shen, Dong-Qing Wei, Qing Li, and Xiao-Yong Wei. Identifying The Kind Behind SMILES – Anatomical Therapeutic Chemical Classification using Structure-Only Representations,  *Briefings in Bioinformatics*, 2022, DOI:10.1093/bib/bbac346.

```bibtex
@ARTICLE{  
author={Yi Cao, Zhen-Qun Yang, Xu-Lu Zhang, Wenqi Fan, Yaowei Wang, Jiajun Shen, Dong-Qing Wei, Qing Li, and Xiao-Yong Wei.},  
journal={Briefings in Bioinformatics},   
title={Identifying The Kind Behind SMILES – Anatomical Therapeutic Chemical Classification using Structure-Only Representations},   
year={2022},   
doi={DOI:10.1093/bib/bbac346}}
```

## Reference

[2] Chen L, Zeng WM, Cai YD, Feng KY, Chou KC. Predicting Anatomical Therapeutic Chemical (ATC) classification of drugs by integrating chemical-chemical interactions and similarities. *PLoS One*. 2012;7(4):e35254.
