import sys, argparse, h5py

sys.path.insert(0, "lib")
from dataset.dataset import Features_HICO_DET, FeatureConstant
from net.model import PD_Net, BaseModelConst
import os
import itertools
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

tqdm.monitor_interval = 0
import torch.optim as optim
from torch.autograd import Variable
from tensorboard_logger import configure, log_value
from torch.utils.data.sampler import RandomSampler
import utils.io as io

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_num',
    type=int)

parser.add_argument(
    '--eval_with_INet', type=bool, default=False)


def eval_model(model, dataset, model_num, pred_hdf5_save_dir, eval_with_INet):
    model.eval()
    if eval_with_INet:
        print('eval with iNet')
        pred_INet_dets_hdf5 = 'output/hico-det/INet/pred_hoi_dets_test.hdf5'
        INet = h5py.File(pred_INet_dets_hdf5, 'r')
        combine_h5py_dir = os.path.join('output/hico-det/PD/pred_hdf5/', 'combine_INet')
        io.mkdir_if_not_exists(combine_h5py_dir)
        pred_hdf5_save_dir = combine_h5py_dir

    pred_hoi_dets_hdf5 = os.path.join(pred_hdf5_save_dir, f'pred_hoi_dets_test_{model_num}.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5, 'w')
    sampler = RandomSampler(dataset)
    torch.manual_seed(0)
    for sample_id in tqdm(sampler):
        data = dataset[sample_id]
        feats = {
            'global_id': data['global_id'],
            'prob_mask': Variable(torch.cuda.FloatTensor(data['prob_mask'])),
            'human_prob_vecs': Variable(torch.cuda.FloatTensor(data['human_prob_vecs'])),
            'object_prob_vecs': Variable(torch.cuda.FloatTensor(data['object_prob_vecs'])),
            'object_word2vec': Variable(torch.cuda.FloatTensor(data['object_word2vec'])),
            'human_det_prob': Variable(torch.cuda.FloatTensor(data['human_det_prob'])),
            'object_det_prob': Variable(torch.cuda.FloatTensor(data['object_det_prob'])),
            'hoi_labels': Variable(torch.cuda.FloatTensor(data['hoi_labels'])),
            'hoi_idx': Variable(torch.cuda.LongTensor(data['hoi_idx'])),
            'human_feats': Variable(torch.cuda.FloatTensor(data['human_app'])),
            'object_feats': Variable(torch.cuda.FloatTensor(data['object_app'])),
            'spatial_feats': Variable(torch.cuda.FloatTensor(data['spatial_feat'])),
            'pose_feats': Variable(torch.cuda.FloatTensor(data['pose_feats'])),
            'hoi_labels_vec': Variable(torch.cuda.FloatTensor(data['hoi_labels_vec'])),
            'basic_verb_human_prob_vecs': Variable(torch.cuda.FloatTensor(data['new_hoi_human_prob_vecs'])),
            'basic_verb_obj_prob_vecs': Variable(torch.cuda.FloatTensor(data['new_hoi_obj_prob_vecs'])),
            'basic_verb_labels': Variable(torch.cuda.FloatTensor(data['new_hoi_labels_vec'])),
            'mask_for_basic_verb': Variable(torch.cuda.FloatTensor(data['new_hoi_mask'])),
            'phrase': Variable(torch.cuda.FloatTensor(data['phrase'])),

        }
        hoi_scores, human_embedding_score, object_embedding_score = model(feats)
        hoi_scores = hoi_scores.data.cpu().numpy()
        hoi_scores = hoi_scores[np.arange(hoi_scores.shape[0]), data['new_hoi_idx']]
        scores = hoi_scores * human_embedding_score.data.cpu().numpy() * object_embedding_score.data.cpu().numpy()

        global_id = data['global_id']
        ## we find it gets higher performance in HICO-DET after multiplying embedding scores
        if eval_with_INet:
            a = INet[global_id]['human_obj_boxes_scores']
            assert (a[:, :4] == data['human_bbox']).all()
            INet_human_obj_boxes_scores = np.copy(a)
            INet_scores = INet_human_obj_boxes_scores[:, 8]
            scores = scores * INet_scores

        human_obj_boxes_scores = np.concatenate((
            data['human_bbox'],
            data['object_bbox'],
            scores.reshape(-1, 1)), 1)


        pred_hois.create_group(global_id)
        pred_hois[global_id].create_dataset(
            'human_obj_boxes_scores',
            data=human_obj_boxes_scores)
        pred_hois[global_id].create_dataset(
            'start_end_ids',
            data=data['start_end_ids'])

    pred_hois.close()


def main_PD_net(args):

    # dataset
    dataset_train_const = FeatureConstant()
    dataset_train_const.use_sample_balance = True
    dataset_test_const = FeatureConstant()
    dataset_train = Features_HICO_DET(dataset_train_const, 'train')
    dataset_test = Features_HICO_DET(dataset_test_const, 'test')

    # model
    model_const = BaseModelConst()
    model_const.attention_fc_num = 2
    model_const.use_pam = True
    model_const.use_channel_attention = True
    model_const.pam_type = ['phrase', 'object_word2vec'][0]
    model_const.lpa_type = ['phrase', 'object_word2vec'][0]

    model_const.pam_dim = dataset_train[0][model_const.pam_type].shape[1]
    model_const.lca_dim = dataset_train[0][model_const.lca_type].shape[1]
    model_const.pose_dim = dataset_train[0]['pose_feats'].shape[1] + dataset_train[0][model_const.lpa_type].shape[1]
    model_const.spatial_feats_dim = dataset_train[0]['spatial_feat'].shape[1] + \
                                    dataset_train[0][model_const.lpa_type].shape[1]
    verb_cls = int(np.max(dataset_train.new_hoi_id) + 1)
    model = PD_Net(model_const, verb_cls=verb_cls).cuda()

    model_dir = "output/hico-det/PD/"
    pred_hdf5_save_dir = os.path.join(model_dir, 'pred_hdf5')
    io.mkdir_if_not_exists(pred_hdf5_save_dir)
    model_path = os.path.join(model_dir, 'model', f'hoi_classifier_{args.model_num}')
    model.load_state_dict(torch.load(model_path))
    print(model)

    eval_model(model,  dataset_test, args.model_num, pred_hdf5_save_dir, args.eval_with_INet)


if __name__ == "__main__":
    args = parser.parse_args()
    main_PD_net(args)

