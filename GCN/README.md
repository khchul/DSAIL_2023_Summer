# Pytorch Implementation of GCN
Paper : Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
Dataset : From PyG

Tested only for Cora Dataset due to PyG version unmatches 
(PyG did not support CUDA >= 12.0)

### Training GCN

```
python train.py --config configs/Cora.txt
```
![Screenshot from 2023-07-05 20-55-09](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/d590b264-3a38-4916-81f5-921e37f7b42b)

![Screenshot from 2023-07-05 20-55-13](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/50f7daa0-59b7-4763-9a5a-4800664ebf1a)

![Screenshot from 2023-07-05 20-55-19](https://github.com/hhhhnFe/DSAIL_2023_Summer/assets/49011793/893c93fb-af91-4782-a13f-d91881d0ab3b)