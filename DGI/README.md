# Pytorch Implementation of DGI
Paper : Veličković, Petar, et al. "Deep graph infomax." arXiv preprint arXiv:1809.10341 (2018).
Dataset : From PyG

### Training DGI

```
python train.py --config configs/Cora_train.txt
```
![Screenshot from 2023-11-01 17-36-24](https://github.com/khchul/DSAIL_2023_Summer/assets/49011793/5f616349-543b-4ba1-a481-afb7c0a9ca30)

### Visualizing Embeddings

```
python visualize.py --config configs/Cora_test.txt
```

![Screenshot from 2023-11-01 14-35-42](https://github.com/khchul/DSAIL_2023_Summer/assets/49011793/3585b49e-2f42-4a61-8f7d-d210fdb3c6ea)
