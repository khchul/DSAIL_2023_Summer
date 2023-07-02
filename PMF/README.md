# Pytorch Implementation of PMF
Paper : Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. Advances in neural information processing systems, 20.
Dataset : https://grouplens.org/datasets/movielens/100k/

## Train dataset 
* datasets/ml-100k/modified_u1.base  
    : Modification of u1.base, User 1 ~ 50 are modified to have only 15 ratings in order to evaluate Constrained PMF,  
    Dropped data are added in u1_1-50
     
## Test dataset
* datasets/ml-100k/u1_1-50.test  
    : Consists of user 1 ~ 50
* datasets/ml-100k/u1_51-.test  
    : Consists of user 51 ~


### Training PMF

```
python train.py --config configs/mod_u1.txt
```

### Training PMFA

```
python train.py --config configs/mod_u1.txt --adaptive --N_a 15
```

### Training constrained PMF

```
python train.py --config configs/mod_u1.txt --constrained --lu=0.002 --lv=0.002 --lw=0.002
```