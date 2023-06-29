# Pytorch Implementation of PMF
Paper : Mnih, A., & Salakhutdinov, R. R. (2007). Probabilistic matrix factorization. Advances in neural information processing systems, 20.
Dataset : https://grouplens.org/datasets/movielens/100k/

## Train dataset 
    * datasets/ml-100k/modified_u1.base for train dataset
        : Modification of u1.base, User 1 ~ 50 are modified to have only 15 ratings in order to evaluate Constrained PMF,
       Dropped data are added in u1_1-50
     
## Test dataset
    * datasets/ml-100k/u1_1-50.test
        : Consists of user 1 ~ 50
    * datasets/ml-100k/u1_51-.test
        : Consists of user 51 ~


