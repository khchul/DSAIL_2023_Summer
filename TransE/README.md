# Pytorch Implementation of TransE
Paper : Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. Advances in neural information processing systems, 26.

Dataset : From PyG

### Training TransE

```
python train.py --config configs/WordNet
```
![Screenshot from 2023-07-14 19-44-11](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/e5800bda-5a82-44c3-b883-f86195a87459)

### Testing TransE

Unlike the paper, normalizing the relation tensors seemed to have better accuracy

#### With normalization

```
python test.py --config configs/WordNet --testname with_lnorm_1000.pt --hit_size 10
```
![Screenshot from 2023-07-17 19-46-58](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/812d06a7-53b1-4d5a-8203-27f837af1ce8)

#### Without normalization
```
python test.py --config configs/WordNet --testname without_lnorm_1000.pt --hit_size 10
```
![Screenshot from 2023-07-17 19-47-10](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/e6e3d1aa-b14f-4b5d-9e22-923dade65f13)

