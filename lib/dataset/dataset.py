import h5py, copy, random, time, sys
import numpy as np, os, json, pickle
from torch.utils.data import Dataset

sys.path.insert(0, "lib")

import utils.io as io


def load_gt_dets(anno_list_json,global_ids):
    global_ids_set = set(global_ids)

    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list = io.load_json_object(anno_list_json)

    gt_dets = {}
    for anno in anno_list:
        if anno['global_id'] not in global_ids_set:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets

class FeatureConstant():
    data_dir = '/data/xubing/Human-Object_Interactions/code/no_frills/no_frills_hoi_det-release_v1/data_symlinks/hico_processed/'
    hoi_cand_data_dir = '/data/xubing/Human-Object_Interactions/code/no_frills/no_frills_hoi_det-release_v1/data_symlinks/hico_exp/hoi_candidates'

    use_pose_type = 'alpha'
    sample_num =1000
    fp_to_tp_ratio = 1000
    data_for_NIS = False
    use_sample_balance = False
    use_human_3parts_agu = False
    use_bert_word = False
    use_relative_loc_between_kpts = False

    zsl_mask = 'VCL_sample_hoi_120_rare_first_mask'
    coco_word2vec_path = os.path.join(data_dir,'coco_word2vec_numpy.npy')
    hico_det_verb_word2vec_path = os.path.join(data_dir, 'verb_wordvec_numpy.npy')
    hico_det_hoi_word2vec_path = os.path.join(data_dir, 'hico_det_hoi_word2vec_numpy.npy')

    hoi_id_to_verb_id_path = os.path.join(data_dir, 'hoi_id_to_verb_id_list.json')
    hoi_id_to_coco_id_path = os.path.join(data_dir, 'hoi_id_to_coco_obj_id_list.json')
    use_object_one_hot_or_wordvec = 'wordvec'
    hoi_list_json = os.path.join(data_dir, 'hoi_list.json')
    verb_list_json = os.path.join(data_dir, 'verb_list.json')
    new_hoi_id_type = 'new_hoi_id_use_word'
    hoi_idx_to_hoi_id_path = os.path.join(data_dir, 'hoi_idx_to_hoi_id.json')


class Features_VCOCO(Dataset):
    def __init__(self, subset="trainval", fp_to_tp_ratio=1000, dir="vcoco"):
        self.subset = subset
        self.fp_to_tp_ratio = fp_to_tp_ratio
        assert subset == "trainval" or subset == "test"

        if self.subset == "trainval":
            self.hoi_cands_train = self.load_hdf5_file(f"data/{dir}/hoi_candidates_train.hdf5")
            self.hoi_cands_val = self.load_hdf5_file(f"data/{dir}/hoi_candidates_val.hdf5")

            self.hoi_cand_labels_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_labels_train.hdf5")
            self.hoi_cand_labels_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_labels_val.hdf5")
            self.hoi_cand_nis_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_nis_train.hdf5")
            self.hoi_cand_nis_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_nis_val.hdf5")

            self.box_feats_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_box_feats_train.hdf5")
            self.box_feats_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_box_feats_val.hdf5")

            self.human_pose_feat_train = self.load_hdf5_file(
                f"data/{dir}/human_pose_feats_train_bbox.hdf5")
            self.human_pose_feat_val = self.load_hdf5_file(
                f"data/{dir}/human_pose_feats_val_bbox.hdf5")

            self.hoi_cands_union_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_union_feats_train.hdf5")
            self.hoi_cands_union_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_union_feats_val.hdf5")

            self.human_new_features_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_human_feats_train.hdf5")
            self.human_new_features_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_human_feats_val.hdf5")
            self.object_new_features_train = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_object_feats_train.hdf5")
            self.object_new_features_val = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_object_feats_val.hdf5")

            self.global_ids_train = self.load_subset_ids("train")
            self.global_ids_val = self.load_subset_ids("val")
            self.global_ids = self.global_ids_train + self.global_ids_val
            print(len(self.global_ids))

        else:
            self.hoi_cands_test = self.load_hdf5_file(f"data/{dir}/hoi_candidates_test.hdf5")

            self.hoi_cand_labels_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_labels_test.hdf5")
            self.hoi_cand_nis_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidate_nis_test.hdf5")

            self.box_feats_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_box_feats_test.hdf5")

            self.human_pose_feat_test = self.load_hdf5_file(
                f"data/{dir}/human_pose_feats_test_bbox.hdf5")

            self.hoi_cands_union_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_union_feats_test.hdf5")

            self.human_new_features_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_human_feats_test.hdf5")
            self.object_new_features_test = self.load_hdf5_file(
                f"data/{dir}/hoi_candidates_object_feats_test.hdf5")

            self.global_ids = self.load_subset_ids("test")
            print(len(self.global_ids))

        self.hoi_234_to_44 = io.load_json_object("data/vcoco/annotations/hoi_id_to_cluster_idx_45.json")
        assert len(self.hoi_234_to_44) == 234

        self.hoi_to_vec_numpy = np.load("data/vcoco/annotations/vcoco_hoi_600.npy")
        assert len(self.hoi_to_vec_numpy) == 234
        assert len(self.hoi_to_vec_numpy[0]) == 600

        self.hoi_dict = self.get_hoi_dict("data/vcoco/annotations/hoi_list_234.json")
        assert len(self.hoi_dict) == 234

        self.obj_to_hoi_ids = self.get_obj_to_hoi_ids()

        self.table_list = io.load_json_object("data/vcoco/annotations/table_list.json")

        self.obj_to_id = self.get_obj_to_id("data/vcoco/annotations/object_list_80.json")
        assert len(self.obj_to_id) == 80

        self.verb_to_id = self.get_verb_to_id("data/vcoco/annotations/verb_list_25.json")
        assert len(self.verb_to_id) == 25

    def get_anno_dict(self, anno_list_json):
        anno_list = io.load_json_object(anno_list_json)
        anno_dict = {anno['global_id']: anno for anno in anno_list}
        return anno_dict

    def load_hdf5_file(self, hdf5_filename, mode='r'):
        return h5py.File(hdf5_filename, mode)

    def get_hoi_dict(self, hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {str(hoi['id']).zfill(3): hoi for hoi in hoi_list}
        return hoi_dict

    def get_obj_to_id(self, object_list_json):
        object_list = io.load_json_object(object_list_json)
        obj_to_id = {obj['name']: obj['id'] for obj in object_list}
        return obj_to_id

    def get_verb_to_id(self, verb_list_json):
        verb_list = io.load_json_object(verb_list_json)
        verb_to_id = {verb['verb'] + "_" + verb['role']: verb['id'] for verb in verb_list}
        return verb_to_id

    def get_obj_to_hoi_ids(self):
        obj_to_hoi_ids = {}
        for hoi_id, hoi in self.hoi_dict.items():
            obj = hoi['object']
            if obj in obj_to_hoi_ids:
                obj_to_hoi_ids[obj].append(hoi_id)
            else:
                obj_to_hoi_ids[obj] = [hoi_id]
        return obj_to_hoi_ids

    def load_subset_ids(self, subset):
        split_ids = io.load_json_object("data/vcoco/annotations/split_ids.json")
        subset_list_ = split_ids[subset]
        subset_list = []
        for id in subset_list_:
            subset_list.append(subset + "_" + id)
        return sorted(subset_list)

    def __len__(self):
        return len(self.global_ids)

    def get_labels(self, global_id):
        # hoi_idx: number in [0,29]
        if self.subset == "trainval":
            if global_id in self.hoi_cands_train.keys():
                hoi_cands = self.hoi_cands_train[global_id]
                hoi_idxs = hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, -1]
                hoi_idxs = hoi_idxs.astype(np.int)
                # label: 0/1 indicating if there was a match with any gt for that hoi
                labels = self.hoi_cand_labels_train[global_id][()]

                num_cand = labels.shape[0]
                hoi_label_vecs = np.zeros([num_cand, len(self.hoi_dict)])
                hoi_label_vecs[np.arange(num_cand), hoi_idxs] = labels

                hoi_ids = [None] * num_cand
                for i in range(num_cand):
                    hoi_ids[i] = str(hoi_idxs[i] + 1).zfill(3)
                return hoi_ids, labels, hoi_label_vecs
            else:
                hoi_cands = self.hoi_cands_val[global_id]
                hoi_idxs = hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, -1]
                hoi_idxs = hoi_idxs.astype(np.int)
                # label: 0/1 indicating if there was a match with any gt for that hoi
                labels = self.hoi_cand_labels_val[global_id][()]

                num_cand = labels.shape[0]
                hoi_label_vecs = np.zeros([num_cand, len(self.hoi_dict)])
                hoi_label_vecs[np.arange(num_cand), hoi_idxs] = labels

                hoi_ids = [None] * num_cand
                for i in range(num_cand):
                    hoi_ids[i] = str(hoi_idxs[i] + 1).zfill(3)
                return hoi_ids, labels, hoi_label_vecs
        else:
            hoi_cands = self.hoi_cands_test[global_id]
            hoi_idxs = hoi_cands['boxes_scores_rpn_ids_hoi_idx'][:, -1]

            hoi_idxs = hoi_idxs.astype(np.int)
            # label: 0/1 indicating if there was a match with any gt for that hoi
            labels = self.hoi_cand_labels_test[global_id][()]

            num_cand = labels.shape[0]
            hoi_label_vecs = np.zeros([num_cand, len(self.hoi_dict)])
            hoi_label_vecs[np.arange(num_cand), hoi_idxs] = labels

            hoi_ids = [None] * num_cand
            for i in range(num_cand):
                hoi_ids[i] = str(hoi_idxs[i] + 1).zfill(3)
            return hoi_ids, labels, hoi_label_vecs

    def get_faster_rcnn_prob_vecs(self, hoi_ids, human_probs, object_probs):
        num_hois = len(self.hoi_dict)
        num_cand = len(hoi_ids)
        human_prob_vecs = np.tile(np.expand_dims(human_probs, 1), [1, num_hois])

        object_prob_vecs = np.zeros([num_cand, num_hois])
        for i, hoi_id in enumerate(hoi_ids):
            obj = self.hoi_dict[hoi_id]['object']
            obj_hoi_ids = self.obj_to_hoi_ids[obj]
            for obj_hoi_id in obj_hoi_ids:
                object_prob_vecs[i, int(obj_hoi_id) - 1] = object_probs[i]
        return human_prob_vecs, object_prob_vecs

    def sample_cands(self, hoi_labels):
        num_cands = hoi_labels.shape[0]
        indices = np.arange(num_cands)
        tp_ids = indices[hoi_labels == 1.0]
        fp_ids = indices[hoi_labels == 0]
        num_tp = tp_ids.shape[0]
        num_fp = fp_ids.shape[0]
        if num_tp == 0:
            num_fp_to_sample = self.fp_to_tp_ratio
        else:
            num_fp_to_sample = min(num_fp, self.fp_to_tp_ratio * num_tp)
        sampled_fp_ids = np.random.permutation(fp_ids)[:num_fp_to_sample]
        sampled_ids = np.concatenate((tp_ids, sampled_fp_ids), 0)
        return sampled_ids

    def get_obj_one_hot(self, hoi_ids):
        num_cand = len(hoi_ids)
        assert len(self.obj_to_id) == 80
        obj_one_hot = np.zeros([num_cand, len(self.obj_to_id)])
        for i, hoi_id in enumerate(hoi_ids):
            obj_id = self.obj_to_id[self.hoi_dict[hoi_id]['object']]
            obj_idx = int(obj_id) - 1
            obj_one_hot[i, obj_idx] = 1.0
        return obj_one_hot

    def get_verb_one_hot(self, hoi_ids):
        num_cand = len(hoi_ids)
        verb_one_hot = np.zeros([num_cand, len(self.verb_to_id)])
        for i, hoi_id in enumerate(hoi_ids):
            action = self.hoi_dict[hoi_id]['verb']
            object = self.hoi_dict[hoi_id]['object']

            d = self.table_list[action]

            verb_name = ""
            if "obj" in d and object in d["obj"]:
                verb_name = action + "_" + "obj"
            elif "instr" in d and object in d["instr"]:
                verb_name = action + "_" + "instr"
            assert verb_name != ""

            verb_id = self.verb_to_id[verb_name]
            verb_idx = int(verb_id) - 1
            verb_one_hot[i, verb_idx] = 1.0
        return verb_one_hot

    def get_prob_mask(self, hoi_idx):
        num_cand = len(hoi_idx)
        prob_mask = np.zeros([num_cand, len(self.hoi_dict)])
        prob_mask[np.arange(num_cand), hoi_idx] = 1.0
        return prob_mask

    def get_features(self, file, global_id, key="features"):
        features = file[global_id][key][()]
        features_indexs = file[global_id]['indexs'][()].tolist()
        ans = []
        for index in features_indexs:
            ans.append(features[int(index)])
        return np.array(ans)

    def get_verb_role_vec_list(self, hoi_idx):
        num = len(hoi_idx)
        vec_list = np.zeros([num, 600])
        for i, index in enumerate(hoi_idx):
            assert len(self.hoi_to_vec_numpy[index]) == 600
            vec_list[i] = self.hoi_to_vec_numpy[index, :]
        return vec_list

    def __getitem__(self, i):
        global_id = self.global_ids[i]

        if self.subset == "trainval":

            if global_id in self.hoi_cands_train.keys():
                start_end_ids = self.hoi_cands_train[global_id]['start_end_ids'][()]
                assert len(start_end_ids) == 234
                hoi_cands_ = self.hoi_cands_train[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
                hoi_ids_, hoi_labels_, hoi_label_vecs_, = self.get_labels(global_id)

                box_feats_ = self.box_feats_train[global_id][()]
                absolute_pose_feat_ = self.human_pose_feat_train[global_id]['absolute_pose'][()]
                relative_pose_feat_ = self.human_pose_feat_train[global_id]['relative_pose'][()]
                nis_labels_ = self.hoi_cand_nis_train[global_id][()]

                verb_obj_vec_ = self.get_verb_role_vec_list(hoi_cands_[:, -1].astype(np.int))
                human_features_ = self.get_features(self.human_new_features_train, global_id)
                object_features_ = self.get_features(self.object_new_features_train, global_id)
                union_features_ = self.get_features(self.hoi_cands_union_train, global_id)

                cand_ids = self.sample_cands(hoi_labels_)

                human_features = human_features_[cand_ids]
                object_features = object_features_[cand_ids]
                verb_obj_vec = verb_obj_vec_[cand_ids]
                hoi_cands = hoi_cands_[cand_ids]
                union_features = union_features_[cand_ids]
                hoi_ids = np.array(hoi_ids_)[cand_ids].tolist()
                hoi_labels = hoi_labels_[cand_ids]
                hoi_label_vecs = hoi_label_vecs_[cand_ids]
                box_feats = box_feats_[cand_ids]
                absolute_pose_feat = absolute_pose_feat_[cand_ids]
                relative_pose_feat = relative_pose_feat_[cand_ids]
                nis_labels = nis_labels_[cand_ids]

            else:
                start_end_ids = self.hoi_cands_val[global_id]['start_end_ids'][()]
                hoi_cands_ = self.hoi_cands_val[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
                hoi_ids_, hoi_labels_, hoi_label_vecs_ = self.get_labels(global_id)

                box_feats_ = self.box_feats_val[global_id][()]

                absolute_pose_feat_ = self.human_pose_feat_val[global_id]['absolute_pose'][()]
                relative_pose_feat_ = self.human_pose_feat_val[global_id]['relative_pose'][()]
                nis_labels_ = self.hoi_cand_nis_val[global_id][()]

                verb_obj_vec_ = self.get_verb_role_vec_list(hoi_cands_[:, -1].astype(np.int))
                human_features_ = self.get_features(self.human_new_features_val, global_id)
                object_features_ = self.get_features(self.object_new_features_val, global_id)
                union_features_ = self.get_features(self.hoi_cands_union_val, global_id)

                cand_ids = self.sample_cands(hoi_labels_)

                human_features = human_features_[cand_ids]
                object_features = object_features_[cand_ids]
                verb_obj_vec = verb_obj_vec_[cand_ids]
                hoi_cands = hoi_cands_[cand_ids]
                union_features = union_features_[cand_ids]
                hoi_ids = np.array(hoi_ids_)[cand_ids].tolist()
                hoi_labels = hoi_labels_[cand_ids]
                hoi_label_vecs = hoi_label_vecs_[cand_ids]
                box_feats = box_feats_[cand_ids]
                nis_labels = nis_labels_[cand_ids]

                absolute_pose_feat = absolute_pose_feat_[cand_ids]
                relative_pose_feat = relative_pose_feat_[cand_ids]

        else:
            start_end_ids = self.hoi_cands_test[global_id]['start_end_ids'][()]
            hoi_cands_ = self.hoi_cands_test[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
            hoi_ids, hoi_labels, hoi_label_vecs = self.get_labels(global_id)
            nis_labels = self.hoi_cand_nis_test[global_id][()]

            box_feats = self.box_feats_test[global_id][()]

            absolute_pose_feat = self.human_pose_feat_test[global_id]['absolute_pose'][()]
            relative_pose_feat = self.human_pose_feat_test[global_id]['relative_pose'][()]

            hoi_cands = hoi_cands_
            verb_obj_vec = self.get_verb_role_vec_list(hoi_cands_[:, -1].astype(np.int))

            human_features = self.get_features(self.human_new_features_test, global_id)
            object_features = self.get_features(self.object_new_features_test, global_id)
            union_features = self.get_features(self.hoi_cands_union_test, global_id)

        to_return = {
            'global_id': global_id,
            'human_box': hoi_cands[:, :4],
            'object_box': hoi_cands[:, 4:8],
            'human_prob': hoi_cands[:, 8],
            'object_prob': hoi_cands[:, 9],
            'human_rpn_id': hoi_cands[:, 10].astype(np.int),
            'object_rpn_id': hoi_cands[:, 11].astype(np.int),
            'hoi_idx': hoi_cands[:, -1].astype(np.int),
            'hoi_id': hoi_ids,
            'hoi_label': hoi_labels,
            'hoi_label_vec': hoi_label_vecs,
            'box_feat': box_feats,
            'absolute_pose': absolute_pose_feat,
            'relative_pose': relative_pose_feat,
            'hoi_cands_': hoi_cands_,
            'start_end_ids_': start_end_ids.astype(np.int),
            "union_features": union_features,
            "verb_obj_vec": verb_obj_vec,
            "human_feat": human_features,
            "object_feat": object_features,
            "nis_labels": nis_labels
        }

        human_prob_vecs, object_prob_vecs = self.get_faster_rcnn_prob_vecs(
            to_return['hoi_id'],
            to_return['human_prob'],
            to_return['object_prob'])

        to_return['human_prob_vec'] = human_prob_vecs
        to_return['object_prob_vec'] = object_prob_vecs

        to_return['object_one_hot'] = self.get_obj_one_hot(to_return['hoi_id'])
        to_return['verb_one_hot'] = self.get_verb_one_hot(to_return['hoi_id'])
        to_return['prob_mask'] = self.get_prob_mask(to_return['hoi_idx'])

        return to_return


class Features_HICO_DET(Dataset):
    def __init__(self, const, subset):
        super(Features_HICO_DET, self).__init__()
        self.const = copy.deepcopy(const)
        if self.const.use_sample_balance:
            print(subset, 'use samples')
        else:
            print(subset, 'no_use samples')
        if self.const.data_for_NIS:
            print('data for NIS')
        else:
            print('data not for NIS')

        self.data_dir = self.const.data_dir
        global_ids_set_test, self.global_ids = self.get_global_ids(subset=subset)
        self.hoi_cands_set = self.load_h5py_file(
            os.path.join(self.const.hoi_cand_data_dir, f'hoi_candidates_{subset}.hdf5'))

        self.hoi_labels = self.load_h5py_file(
            os.path.join(self.const.hoi_cand_data_dir, f'hoi_candidate_labels_{subset}.hdf5'))
        self.faster_rcnn_feats = self.load_h5py_file(os.path.join(self.const.data_dir, f'faster_rcnn_fc7.hdf5'))
        self.coco_wordvec_numpy = np.load(self.const.coco_word2vec_path)
        self.hico_det_verb_wordvec_numpy = np.load(self.const.hico_det_verb_word2vec_path)
        self.hico_det_hoi_wordvec_numpy = np.load(self.const.hico_det_hoi_word2vec_path)
        if self.const.use_bert_word:
            self.coco_wordvec_numpy = np.load(self.const.coco_bert_path)
            self.hico_det_verb_wordvec_numpy = np.load(self.const.verb_bert_path)
            self.hico_det_hoi_wordvec_numpy = np.load(self.const.hoi_bert_path)
        self.hoi_id_to_verb_id_list = np.array(io.load_json_object(self.const.hoi_id_to_verb_id_path))
        self.hoi_id_to_coco_id = np.array(io.load_json_object(self.const.hoi_id_to_coco_id_path))
        self.spatial_feats = self.load_h5py_file(os.path.join(self.const.hoi_cand_data_dir, f'hoi_candidates_box_feats_{subset}.hdf5'))
        self.hoi_dict = self.get_hoi_dict(self.const.hoi_list_json)
        self.obj_to_hoi_ids = self.get_obj_to_hoi_ids(self.hoi_dict)
        self.gt_dets_ = load_gt_dets(self.const.data_dir + '/anno_list.json', global_ids_set_test)
        print('new_hoi_id_type', self.const.new_hoi_id_type)
        self.new_hoi_id = np.load(os.path.join(self.const.data_dir, self.const.new_hoi_id_type+'.npy')).astype(np.float64)

        self.human_alphapose_feats = self.load_h5py_file(
            os.path.join(self.const.hoi_cand_data_dir, f'human_pose_feats_{subset}_{self.const.use_pose_type}.hdf5'))
        self.hoi_idx_to_hoi_id = np.array(io.load_json_object(self.const.hoi_idx_to_hoi_id_path))

    def get_hoi_dict(self, hoi_list_json):
        hoi_list = io.load_json_object(hoi_list_json)
        hoi_dict = {hoi['id']: hoi for hoi in hoi_list}
        return hoi_dict

    def load_pkl_file(self, path):
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print('load', path, 'ends!!!')
        return data

    def get_global_ids(self, subset):
        split_ids_json = os.path.join(self.data_dir, 'split_ids_my.json')
        with open(split_ids_json, 'r') as f:
            split_ids = json.load(f)
        global_ids = split_ids[subset]

        global_ids_set = sorted(global_ids)

        return global_ids_set, list(global_ids_set)

    def load_h5py_file(self, path, mode='r'):
        if not os.path.exists(path):
            print(path)
            assert False
        data = h5py.File(path, mode)
        print('load ', path, ' end!!!')
        return data

    def sample_hoi_cands(self, hoi_lables):
        hoi_lables = np.array(hoi_lables)
        tp = np.where(hoi_lables == 1.0)[0]
        fp = np.where(hoi_lables == 0.0)[0]
        tp_num = tp.shape[0]
        if self.const.data_for_NIS:
            self.const.fp_to_tp_ratio = 3
        if tp_num == 0:
            np.random.shuffle(fp)
            inds = fp[:self.const.sample_num]
        else:
            fp_num = min(fp.shape[0], tp_num * self.const.fp_to_tp_ratio)
            np.random.shuffle(fp)
            fp_inds = fp[:fp_num]
            inds = np.concatenate([tp, fp_inds])
        return inds

    def get_simple_spatial_feats(self, human_bbox, obj_bbox):

        h_x1, h_y1, h_x2, h_y2 = [human_bbox[:, i].reshape(-1, 1) for i in range(4)]
        o_x1, o_y1, o_x2, o_y2 = [obj_bbox[:, i].reshape(-1, 1) for i in range(4)]

        x1 = np.min(np.concatenate([h_x1, o_x1], 1), 1).reshape(-1, 1)
        y1 = np.min(np.concatenate([h_y1, o_y1], 1), 1).reshape(-1, 1)
        x2 = np.max(np.concatenate([h_x2, o_x2], 1), 1).reshape(-1, 1)
        y2 = np.max(np.concatenate([h_y2, o_y2], 1), 1).reshape(-1, 1)

        def normal(x, x_coordinate, w):
            out = (x - x_coordinate) / (w + 1e-6)
            return out

        w = x2 - x1
        h = y2 - y1
        h_x1 = normal(h_x1, x1, w)
        h_x2 = normal(h_x2, x1, w)
        h_y1 = normal(h_y1, y1, h)
        h_y2 = normal(h_y2, y1, h)

        o_x1 = normal(o_x1, x1, w)
        o_x2 = normal(o_x2, x1, w)
        o_y1 = normal(o_y1, y1, h)
        o_y2 = normal(o_y2, y1, h)
        spatial_feats = np.concatenate([h_x1, h_x2, h_y1, h_y2, o_x1, o_x2, o_y1, o_y2], 1)

        return spatial_feats

    def get_obj_to_hoi_ids(self, hoi_dict):
        obj_to_hoi_ids = {}
        for hoi_id, hoi in hoi_dict.items():
            obj = hoi['object']
            if obj in obj_to_hoi_ids:
                obj_to_hoi_ids[obj].append(hoi_id)
            else:
                obj_to_hoi_ids[obj] = [hoi_id]
        return obj_to_hoi_ids

    def get_faster_rcnn_prob_vecs(self, hoi_ids, human_probs, object_probs):
        num_hois = len(self.hoi_dict)
        num_cand = len(hoi_ids)
        human_prob_vecs = np.tile(np.expand_dims(human_probs, 1), [1, num_hois])
        object_prob_vecs = np.zeros([num_cand, num_hois])

        for i, hoi_id in enumerate(hoi_ids):
            obj = self.hoi_dict[hoi_id]['object']
            obj_hoi_ids = self.obj_to_hoi_ids[obj]
            for obj_hoi_id in obj_hoi_ids:
                object_prob_vecs[i, int(obj_hoi_id) - 1] = object_probs[i]
        return human_prob_vecs, object_prob_vecs

    def get_no_frills_openpose_feats(self, global_id):
        # pose feats
        absolute_pose_feat_ = self.human_pose_feats[global_id]['absolute_pose'][()]
        relative_pose_feat_ = self.human_pose_feats[global_id]['relative_pose'][()]
        pose_feats_ = np.concatenate((absolute_pose_feat_, relative_pose_feat_), 1)
        log_pose_feats = np.log(np.abs(pose_feats_) + 1e-6)
        pose_feats = np.concatenate((pose_feats_, log_pose_feats), 1)
        assert pose_feats.shape[1] == 288
        return pose_feats

    def get_new_pose_fetas(self, global_id):
        absolute_pose_feat_ = self.human_pose_feats[global_id]['absolute_pose'][()]
        # absolute_pose_feat_ = absolute_pose_feat_[:, :14*3]
        relative_pose_feat_ = self.human_pose_feats[global_id]['relative_pose'][()]
        # relative_pose_feat_ = relative_pose_feat_[:, :14*5]
        kypts_num = 18
        absolute_pose_feat_ = absolute_pose_feat_.reshape(-1, kypts_num, 3)
        samples_num = absolute_pose_feat_.shape[0]
        relative_loc_between_kypts = np.zeros([samples_num, 0, 4])  # C(n, 2)
        for i in range(kypts_num - 1):
            kypts_i = absolute_pose_feat_[:, i, :2].reshape(-1, 1, 2)
            kypts_no_i = absolute_pose_feat_[:, i + 1:, :2]

            def compute_relative_loc(loc1, loc2):
                loc1 = np.tile(loc1, (1, loc2.shape[1], 1))
                assert loc1.shape == loc2.shape
                out = np.zeros([loc1.shape[0], loc1.shape[1], 4])
                out[:, :, 0] = loc2[:, :, 0] - loc1[:, :, 0]
                out[:, :, 1] = loc2[:, :, 1] - loc1[:, :, 1]
                return out

            out = compute_relative_loc(kypts_i, kypts_no_i)
            out[:, :, 2] = np.tile(absolute_pose_feat_[:, i, 2].reshape(-1, 1), (1, kypts_num - (i + 1)))
            out[:, :, 3] = absolute_pose_feat_[:, i + 1:, 2]
            relative_loc_between_kypts = np.concatenate((relative_loc_between_kypts, out), 1)
        assert relative_loc_between_kypts.shape[1] == kypts_num * (kypts_num - 1) / 2

        relative_loc_between_kypts = relative_loc_between_kypts.reshape(samples_num, -1)
        # absolute_pose_feat_ = absolute_pose_feat_.reshape(samples_num, -1)
        pose_feats_ = np.concatenate((relative_loc_between_kypts, relative_pose_feat_), 1)
        log_pose_feats = np.log(np.abs(pose_feats_) + 1e-6)
        pose_feats = np.concatenate((pose_feats_, log_pose_feats), 1)
        return pose_feats, relative_loc_between_kypts

    def get_no_frills_alphapose_feats(self, global_id):
        # pose feats
        absolute_pose_feat_ = self.human_alphapose_feats[global_id]['absolute_pose'][()]
        relative_pose_feat_ = self.human_alphapose_feats[global_id]['relative_pose'][()]
        pose_feats_ = np.concatenate((absolute_pose_feat_, relative_pose_feat_), 1)
        log_pose_feats = np.log(np.abs(pose_feats_) + 1e-6)
        pose_feats = np.concatenate((pose_feats_, log_pose_feats), 1)
        assert pose_feats.shape[1] == 288-16
        return pose_feats

    def get_features(self, file, global_id):
        features = file[global_id]['features'][()]
        indexs = file[global_id]['indexs'][()].astype(np.int)

        return features[indexs]

    def get_union_features(self, file, index, global_id):

        features = file[global_id]['union_features'][()]
        indx = index[global_id]['indexs'][()]
        indx = indx.astype(np.int)
        return features[indx]

    def __len__(self):
        return len(self.global_ids)

    def __getitem__(self, i):
        start_time = time.time()
        global_id = self.global_ids[i]
        start_end_ids = self.hoi_cands_set[global_id]['start_end_ids'][()]
        hoi_cands_ = self.hoi_cands_set[global_id]['boxes_scores_rpn_ids_hoi_idx'][()]
        hoi_labels_ = self.hoi_labels[global_id][()]

        # union bbox
        #union_features = self.get_features(self.union_features_gmp, global_id)

        # word vec
        hoi_idx = hoi_cands_[:, -1].astype(np.int)
        hoi_wordvec = self.hico_det_hoi_wordvec_numpy[hoi_idx, :]

        verb_idx_ = self.hoi_id_to_verb_id_list[hoi_idx]
        verb_wordvec = self.hico_det_verb_wordvec_numpy[verb_idx_, :]



        num = hoi_cands_.shape[0]

        coco_idx = self.hoi_id_to_coco_id[hoi_idx]
        obj_wordvec = self.coco_wordvec_numpy[coco_idx]



        # csp

        num_new_hoi_id = int(np.max(self.new_hoi_id) + 1)
        new_hoi_idx = self.new_hoi_id[hoi_idx].astype(np.int)

        new_hoi_mask = np.zeros([num, num_new_hoi_id])
        new_hoi_mask[np.arange(num), new_hoi_idx] = 1
        new_hoi_labels_vec = np.zeros([num, num_new_hoi_id])
        new_hoi_labels_vec[np.arange(num), new_hoi_idx] = hoi_labels_

        new_hoi_human_prob_vecs = np.zeros_like(new_hoi_mask)
        new_hoi_human_prob_vecs[np.arange(num), new_hoi_idx] = hoi_cands_[:, 8]

        new_hoi_obj_prob_vecs = np.zeros_like(new_hoi_mask)
        new_hoi_obj_prob_vecs[np.arange(num), new_hoi_idx] = hoi_cands_[:, 9]

        # human app

        human_app_feats = np.take(self.faster_rcnn_feats[global_id],
                                  hoi_cands_[:, 10].astype(np.int),
                                  axis=0)


        # obj app
        obj_feats = np.take(self.faster_rcnn_feats[global_id],
                            hoi_cands_[:, 11].astype(np.int),
                            axis=0)
        # spatial feats
        object_one_hot = np.zeros([num, 80])
        object_one_hot[np.arange(num), coco_idx] = 1
        box_feats = self.spatial_feats[global_id][()]
        box_log_feat = np.log(np.abs(box_feats) + 1e-6)

        spatial_feats = np.concatenate((box_feats, box_log_feat), 1)
        assert spatial_feats.shape[1] == 42


        # pose feats
        pose_feats = self.get_no_frills_alphapose_feats(global_id)


        # mask
        prob_mask_ = np.zeros([num, 600])
        prob_mask_[np.arange(num), hoi_idx] = 1
        hoi_ids_ = self.hoi_idx_to_hoi_id[hoi_idx].tolist()



        if self.const.use_sample_balance:
            # sample
            sample_inds = self.sample_hoi_cands(hoi_labels_)
            hoi_labels = hoi_labels_[sample_inds]

            hoi_cands = hoi_cands_[sample_inds]
            human_app_feats = human_app_feats[sample_inds]
            human_dets = hoi_cands[:, 8]
            obj_dets = hoi_cands[:, 9]
            obj_feats = obj_feats[sample_inds]
            spatial_feats = spatial_feats[sample_inds]
            obj_wordvec = obj_wordvec[sample_inds]

            verb_wordvec = verb_wordvec[sample_inds]
            hoi_wordvec = hoi_wordvec[sample_inds]
            verb_idx = verb_idx_[sample_inds]
            hoi_idx = hoi_idx[sample_inds]

            pose_feats = pose_feats[sample_inds]
            object_one_hot = object_one_hot[sample_inds]
            hoi_ids = (np.array(hoi_ids_)[sample_inds]).tolist()
            prob_mask = prob_mask_[sample_inds]


            new_hoi_idx = new_hoi_idx[sample_inds]
            new_hoi_mask = new_hoi_mask[sample_inds]
            new_hoi_labels_vec = new_hoi_labels_vec[sample_inds]
            new_hoi_obj_prob_vecs = new_hoi_obj_prob_vecs[sample_inds]
            new_hoi_human_prob_vecs = new_hoi_human_prob_vecs[sample_inds]



        else:
            hoi_labels = hoi_labels_
            hoi_cands = hoi_cands_
            human_dets = hoi_cands[:, 8]
            obj_dets = hoi_cands[:, 9]
            verb_idx = verb_idx_
            hoi_ids = hoi_ids_
            prob_mask = prob_mask_

        hoi_num = hoi_labels.shape[0]
        feats = {}
        feats['global_id'] = global_id
        feats['hoi_id'] = hoi_ids
        feats['human_bbox'] = hoi_cands[:, :4]
        feats['object_bbox'] = hoi_cands[:, 4:8]
        feats['hoi_cands'] = hoi_cands
        feats['spatial_feat'] = spatial_feats
        feats['human_app'] = human_app_feats
        feats['object_app'] = obj_feats
        feats['object_word2vec'] = obj_wordvec
        feats['hoi_word2vec'] = hoi_wordvec
        feats['verb_word2vec'] = verb_wordvec
        feats['human_det_prob'] = human_dets
        feats['object_det_prob'] = obj_dets
        feats['verb_idx'] = verb_idx
        feats['hoi_idx'] = hoi_idx
        feats['hoi_labels'] = hoi_labels
        feats['start_end_ids'] = start_end_ids.astype(np.int)

        hoi_labels_vec = np.zeros([hoi_num, 600])
        hoi_labels_vec[np.arange(hoi_num), hoi_idx] = hoi_labels
        feats['hoi_labels_vec'] = hoi_labels_vec
        feats['pose_feats'] = pose_feats

        feats['coco_idx'] = coco_idx
        human_prob_vecs, object_prob_vecs = self.get_faster_rcnn_prob_vecs(
            feats['hoi_id'],
            feats['human_det_prob'],
            feats['object_det_prob'])
        feats['human_prob_vecs'] = human_prob_vecs
        feats['object_prob_vecs'] = object_prob_vecs
        feats['prob_mask'] = prob_mask
        feats['phrase'] = np.concatenate((verb_wordvec, obj_wordvec), 1).astype(np.float64)
        feats['new_hoi_idx'] = new_hoi_idx
        feats['new_hoi_mask'] = new_hoi_mask
        feats['new_hoi_labels_vec'] = new_hoi_labels_vec
        feats['new_hoi_human_prob_vecs'] = new_hoi_human_prob_vecs
        feats['new_hoi_obj_prob_vecs'] = new_hoi_obj_prob_vecs

        return feats



if __name__ == "__main__":
    c = FeatureConstant()
    d = Features_HICO_DET(c, 'train')


