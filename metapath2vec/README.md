# Pytorch Implementation of metapath2vec
Paper : Dong, Y., Chawla, N. V., & Swami, A. (2017, August). metapath2vec: Scalable representation learning for heterogeneous networks. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 135-144).

Dataset : AMiner, with reduced size

### Training metapath

```
python train.py --config configs/AMiner.txt --is_meta
```

### Visualizing metapath

```
python visualize.py --config configs/AMiner.txt
```

