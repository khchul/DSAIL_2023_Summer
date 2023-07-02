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
![pmf](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/f80f4529-e43d-4b3b-b751-f7f684337dc1)

### Training PMFA

```
python train.py --config configs/mod_u1.txt --adaptive --N_a 15
```
![pmfa](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/cf3608ec-33f6-4a66-b903-500bff88e84f)

### Training constrained PMF

```
python train.py --config configs/mod_u1.txt --constrained --lu=0.002 --lv=0.002 --lw=0.002
```
![cpmf](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/f04cf7e9-e349-488c-baac-65c20a834885)

### Test Result
![test_result](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/b19a5067-3bfa-40a4-a9bd-45897b0a4c13)
