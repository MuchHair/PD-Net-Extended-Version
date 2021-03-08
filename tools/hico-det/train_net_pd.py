import sys, argparse

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
    '--GPU',
    type=str,
    default='0',
    help='GPU id')

parser.add_argument(
    '--embedding_loss_weight',
    type=float,
    default=1.0)



def train_model(model, dataset_train, lr, num_epochs, model_dir, args, no_embed_loss=False, parm_need_train=None, begin_step=0):
    if parm_need_train is None:
        params = model.parameters()
        optimizer = optim.Adam(params, lr=lr)
    else:
        optimizer = optim.Adam(parm_need_train, lr=lr)

    criterion = nn.BCELoss()
    model.train()

    step = begin_step
    optimizer.zero_grad()
    for epoch in range(0, num_epochs):
        sampler = RandomSampler(dataset_train)
        for i, sample_id in enumerate(sampler):
            data = dataset_train[sample_id]
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

            loss1 = criterion(hoi_scores, feats['basic_verb_labels'])
            loss2 = criterion(human_embedding_score, feats['hoi_labels'])
            loss3 = criterion(object_embedding_score, feats['hoi_labels'])

            if no_embed_loss:
                loss = loss1
            else:
                loss = loss1 + args.embedding_loss_weight * loss2 + args.embedding_loss_weight * loss3
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            max_prob = hoi_scores.max().data[0]
            max_prob_tp = torch.max(hoi_scores * feats['basic_verb_labels']).data[0]

            if step % 20 == 0 and step != 0:
                num_tp = np.sum(data['hoi_labels'])
                num_fp = data['hoi_labels'].shape[0] - num_tp
                log_str = \
                    'Epoch: {} | Iter: {} | Step: {} | ' + \
                    'Train Loss: {:.8f} | TPs: {} | FPs: {} | ' + \
                    'Max TP Prob: {:.8f} | Max Prob: {:.8f} | lr:{}'
                log_str = log_str.format(
                    epoch,
                    i,
                    step,
                    loss.data[0],
                    num_tp,
                    num_fp,
                    max_prob_tp,
                    max_prob,
                    optimizer.param_groups[0]["lr"])
                print(log_str)


            if step % 5000 == 0 and step > 50000:
                hoi_classifier_pth = os.path.join(
                    model_dir, "model",
                    f'hoi_classifier_{step}')
                torch.save(
                    model.state_dict(),
                    hoi_classifier_pth)

            step += 1



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
    lr = 1e-3
    num_epochs = 6
    model_dir = "output/hico-det/PD"
    io.mkdir_if_not_exists(model_dir, recursive=True)
    io.mkdir_if_not_exists(os.path.join(model_dir, "log"))
    io.mkdir_if_not_exists(os.path.join(model_dir, "model"))

    configure(os.path.join(model_dir, "log"))
    print(model)

    train_model(model, dataset_train, lr, num_epochs, model_dir, args, no_embed_loss=False, parm_need_train=None, begin_step=0)


if __name__ == "__main__":
    args = parser.parse_args()
    main_PD_net(args)

