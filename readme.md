# RRF-GZSL

Code for the CVPR 2020 paper: Learning the Redundancy-free Features for Generalized Zero-Shot Object Recognition. [[paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Han_Learning_the_Redundancy-Free_Features_for_Generalized_Zero-Shot_Object_Recognition_CVPR_2020_paper.pdf)]



![generation_framework](./images/generation_framework.jpg)

## Learn the Redundancy-free Features for GZSL

### Dependencies
This code requires the following:
- Python 3.6
- Pytorch 1.1.0
- scikit-learn

### Datasets

Download the dataset (AWA1/CUB/SUN/FLO) from the work of [Xian et al. (CVPR2017)](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), and save correspongding data into directory `./data/`.

### Train and Test

Run `python RFF_GZSL.py` with the following args:

* `--dataset`: datasets, e.g: CUB.

* `--syn_num`: number synthetic features for each class.

* `--preprocessing`: preprocess the raw visual features or not.

* `--attSize`: size of semantic features.

* `--nz`: size of the Gaussian noise z.

* `--nepoch`: number of the epoch for training.

* `--i_c`: information constrain, corresponds to b in Eq. (5) and (9).

* `--manualSeed`: manual seed.

* `--nclass_all`: number of all classes.

* `--nclass_seen`: number of seen classes

* `--lr_dec`: enable lr decay or not.

* `--lr_dec_ep`: the period of conducting lr decay.

* `--lr_dec_rate`: lr decay rate.

* `--final_classifier`: the classifier for final classification, e.g. softmax or knn.

* `--k`: k for knn. If `--final_classifier` is `softmax`, this is not needed.

* `--center_margin`: margin in the center loss.

* `--center_weight`: weight for the center loss.

For example:

*  Softmax as the final classifier:
```python
python3 RRF_GZSL.py --dataset CUB --syn_num 400 --preprocessing --batch_size 512 --attSize 312 --nz 312 --nepoch 208 --i_c 0.1 --cls_weight 0.2 --lr 0.0001 --manualSeed 3483 --nclass_all 200 --nclass_seen 150 --lr_dec --lr_dec_ep 100 --lr_dec_rate 0.95 --center_margin 190 --center_weight 0.1 --final_classifier softmax
```

*  knn as the final classifier:
```python
python3 RRF_GZSL.py --dataset CUB --syn_num 600 --preprocessing --batch_size 512 --attSize 312 --nz 312 --nepoch 851 --i_c 0.1 --cls_weight 0.2 --lr 0.0001 --manualSeed 3483 --nclass_all 200 --nclass_seen 150 --lr_dec --lr_dec_ep 100 --lr_dec_rate 0.95 --center_margin 190 --center_weight 0.1 --final_classifier knn --k 5
```

### Citation
If you find it useful, please cite:
```
@InProceedings{Han_2020_CVPR,
author = {Han, Zongyan and Fu, Zhenyong and Yang, Jian},
title = {Learning the Redundancy-Free Features for Generalized Zero-Shot Object Recognition},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```

