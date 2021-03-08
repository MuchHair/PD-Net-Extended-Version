# Official Source code for PD-Net 
Polysemy Deciphering Network for Human-Object Interaction Detection （[[ECCV2020 paper]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650069.pdf) [[Code]](https://github.com/MuchHair/PD-Net)）

###  Polysemy Deciphering Network for Robust Human-Object Interaction Detection （[[IJCV paper]](https://arxiv.org/pdf/2008.02918.pdf))
<img src="https://github.com/MuchHair/PD-Net-Extended-Version/blob/master/Paper_Images/overview.png" width="999" >

### Train, Test and Eval Model
```
# train
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/train_net_pd.py

# test(use tensorboard to choose the best model and the precoss will generate a .hdf5 file used for eval)
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/test_net_pd.py

# eval (use the .hdf5 generated above to eval)
bash eval/compute_mAP.sh
```

### HOI-VP Dataset
The Images are provided by [VG](http://visualgenome.org/api/v0/api_home.html) and the annotations (based on [HCVRD](https://github.com/bohanzhuang/HCVRD-a-benchmark-for-large-scale-Human-Centered-Visual-Relationship-Detection)) can be obtained from [this link](https://pan.baidu.com/s/1LCDtjDNbIqJFDLsoPqZOsg).



## Citation
Please consider citing this paper in your publications if it helps your research. The following is a BibTeX reference. 
```
@article{zhong2020polysemy,
  title={Polysemy Deciphering Network for Robust Human-Object Interaction Detection},
  author={Zhong, Xubin and Ding, Changxing and Qu, Xian and Tao, Dacheng},
  journal={arXiv preprint arXiv:2008.02918},
  year={2020}
}
```
