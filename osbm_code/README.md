## Qualitative
The python notebook `Community_Detection_NIPS12.ipynb` demonstrates how DGLFRM embeddings can be used to readily extract overlapping communities. 

## Quantitative Experiments 

- Change the dataset using the `--dataset` option. We have included 10 random splits for three different datasets (cora, citeseer, and nips12). 
- To choose the random split set `--split_idx` option to a number (0-9).
- Two different versions of the framework (DGLFRM and DGLFRM-B) can be run using the `--model` option.
- You may directly use the following commands to train the model.


### DGLFRM
---------------------------------
```python train.py --dataset cora --hidden 32_50 --alpha0 10 --split_idx 0 --early_stopping 0 --deep_decoder 1 --model dglfrm --epochs 500```
- Test ROC score: 0.9279189425663147
- Test AP score: 0.9313888813717796
- Test Z Activated: 14.623781388478582
---------------------------------

---------------------------------
```python train.py --dataset cora --hidden 32_50 --alpha0 10 --split_idx 1 --early_stopping 0 --deep_decoder 1 --model dglfrm --epochs 500```
- Test ROC score: 0.9263454662638759
- Test AP score: 0.9348284353038857
- Test Z Activated: 14.575701624815363
---------------------------------

---------------------------------
```python train.py --dataset cora --hidden 32_50 --alpha0 10 --split_idx 2 --early_stopping 0 --deep_decoder 1 --model dglfrm --epochs 500```
- Test ROC score: 0.930140532677538
- Test AP score: 0.9292958502878256
- Test Z Activated: 14.235524372230428
---------------------------------

### DGLFRM-B

---------------------------------
```python train.py --dataset cora --hidden 32_100 --alpha0 20 --split_idx 0 --early_stopping 0 --deep_decoder 1 --model dglfrm_b --epochs 500```
- Test ROC score: 0.9164509287830943
- Test AP score: 0.9180554129495934
- Test Z Activated: 24.823929098966026
---------------------------------

#### Directory structure

##### 3 Datasets 
- cora 
- citeseer
- NIPS12

##### Source dir and contents
- `train.py`:  Trains one of the three models (DGLFRM and DGLFRM-B). All the three models are instances of the framework presented in the paper. 
- `model.py`:  The model definition of the three models.
- `utils.py`:  Common functions including the computation of KL divergence of beta-kumarswamy and bernoulli-concrete distributions. 
- `optimizer.py`: Loss formulation and optimization.
- `layers.py`:  Definition of the layers
