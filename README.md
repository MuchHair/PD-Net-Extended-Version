# Official Source code for PD-Net 
Polysemy Deciphering Network for Human-Object Interaction Detection （[[ECCV2020 paper]](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650069.pdf) [[Code]](https://github.com/MuchHair/PD-Net)）

###  Polysemy Deciphering Network for Robust Human-Object Interaction Detection （[[IJCV paper]](https://arxiv.org/pdf/2008.02918.pdf))
<img src="https://github.com/MuchHair/PD-Net-Extended-Version/blob/master/Paper_Images/overview.png" width="999" >

### Train, Test and Eval Model on HICO-DET
#### Preprocess data
1. Please prepare these [files](https://pan.baidu.com/s/1pcRqDsFzMP1C9Frgag7Ygw) (pwd:1111) 
 Put them in data/hico/hico_processed dir.
2. Prepare faster_rcnn_fc7.hdf5 (Step 1 in [No-frills](https://github.com/BigRedT/no_frills_hoi_det#evaluate-model)) and 
 put it in data/hico/hico_processed dir.
3. Please follow [No-frills](https://github.com/BigRedT/no_frills_hoi_det#evaluate-model) to obtain the 
"hoi_candidates_subset.hdf5" "hoi_candidates_box_feats_subset.hdf5", "hoi_candidate_labels_subset.hdf5" files. 
Put them in  data/hico/hoi_candidates dir.
4. Prepare pose

```
# prepare input file for AlphaPose
python -m lib.data_process.prepare_for_pose
```
use [AlphaPose](https://github.com/SherlockHolmes221/AlphaPose) to obtain the pose results

```
# convert and generate features
python -m lib.data_process_hico.convert_pose_result
python -m lib.data_process_hico.cache_alphapose_features
```
Please put the final .hdf5 pose file in data/hico/hoi_candidates dir.

5 .If evaling PD-Net with INet, download a pre-trained INet [preditions](https://pan.baidu.com/s/10NYRHthOR53iZInraAxoDQ) (pwd:1111) and put this .hdf5 file in output/hico-det/INet/ dir
```
 # train
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/train_net_pd.py

# test(choose a model  MODEL_NUM to test and the precoss will generate a .hdf5 file used for eval)
CUDA_VISIBLE_DEVICES=0 python tools/vcoco/test_net_pd.py --model_num MODEL_NUM --eval_with_INet True

# eval (use the .hdf5 generated above to eval)
bash eval/compute_mAP.sh
```
#### [Pretrained model](https://pan.baidu.com/s/1gm6DQaQmr-ai1U2JIfbOfA) (22.37 mAP on HICO-DET)
### HOI-VP Dataset
The Images are provided by [VG](http://visualgenome.org/api/v0/api_home.html) and the annotations (based on [HCVRD](https://github.com/bohanzhuang/HCVRD-a-benchmark-for-large-scale-Human-Centered-Visual-Relationship-Detection)) can be obtained from [this link](https://pan.baidu.com/s/14aYOJk6Fi4KihVsGhweKjQ) (pwd:1111).



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
