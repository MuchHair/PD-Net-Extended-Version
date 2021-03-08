import os, json, sys
import argparse
import time
import h5py
from tqdm import tqdm
import numpy as np
#import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import average_precision_score, precision_recall_curve
sys.path.insert(0, "../lib")

import utils.io as io
from utils.bbox_utils import compute_iou

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pred_hoi_dets_hdf5', 
    type=str, 
    default=None,
    required=True,
    help='Path to predicted hoi detections hdf5 file')
parser.add_argument(
    '--out_dir', 
    type=str, 
    default=None,
    required=True,
    help='Output directory')
parser.add_argument(
    '--proc_dir',
    type=str,
    default=None,
    required=True,
    help='Path to HICO processed data directory')
parser.add_argument(
    '--subset',
    type=str,
    default='test',
    choices=['train','test','val','train_val'],
    help='Subset of data to run the evaluation on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=12,
    help='Number of processes to parallelize across')

parser.add_argument(
    '--exp_name',
    type=str,
    required=True)

parser.add_argument(
    '--file_name',
    type=str,
    required=True)
parser.add_argument(
    '--model_num',
    type=int,
    required=True
)

parser.add_argument(
    '--mAP_dir',
    type=str,
    required=True)

parser.add_argument(
    '--mode',
    type=str,
    default='Default'
)

parser.add_argument(
    '--thres',
    type=float,
    required=False
)

def match_hoi(pred_det,gt_dets):
    is_match = False
    remaining_gt_dets = [gt_det for gt_det in gt_dets]
    for i,gt_det in enumerate(gt_dets):
        human_iou = compute_iou(pred_det['human_box'],gt_det['human_box'])
        if human_iou > 0.5:
            object_iou = compute_iou(pred_det['object_box'],gt_det['object_box'])
            if object_iou > 0.5:
                is_match = True
                del remaining_gt_dets[i]
                break
        #remaining_gt_dets.append(gt_det)

    return is_match, remaining_gt_dets


def compute_ap(precision,recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0,1.1,0.1): # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall>=t]
        if selected_p.size==0:
            p = 0
        else:
            p = np.max(selected_p)   
        ap += p/11.
    
    return ap


def compute_pr(y_true,y_score,npos):
    sorted_y_true = [y for y,_ in 
        sorted(zip(y_true,y_score),key=lambda x: x[1],reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos==0:
        recall = np.nan*tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall


def compute_normalized_pr(y_true,y_score,npos,N=196.45):
    sorted_y_true = [y for y,_ in 
        sorted(zip(y_true,y_score),key=lambda x: x[1],reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos==0:
        recall = np.nan*tp
    else:
        recall = tp / npos
    precision = recall*N / (recall*N + fp)
    nap = np.sum(precision[sorted_y_true]) / (npos+1e-6)
    return precision, recall, nap


def eval_hoi(hoi_id,global_ids,gt_dets,pred_dets_hdf5,out_dir, args):
    # print(f'{args.mode} Evaluating hoi_id: {hoi_id} ...')
    pred_dets = h5py.File(pred_dets_hdf5, 'r')
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    for global_id in global_ids:
        if global_id in loss_id:    # we find that some images have no ground truth so we delete them
            continue
        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)-1].astype(np.int)
        hoi_dets = \
            pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx, _ in sorted(
            zip(range(num_dets), hoi_dets[:, 8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_dets[i, :4],
                'object_box': hoi_dets[i, 4:8],
                'score': hoi_dets[i, 8]
            }
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id,i))

    # Compute PR
    if len(y_score)==0:
        if npos==0:
            ap = np.nan
        else:
            ap = 0.0

    else:
        precision, recall = compute_pr(y_true, y_score, npos)
        # Compute AP
        ap = compute_ap(precision,recall)
    #print(f'AP:{ap}')


    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap,hoi_id)


def eval_hoi_select_img_by_object_scores2(hoi_id, global_ids, gt_dets, pred_dets_hdf5, out_dir, args):
    print(f'{args.mode} Evaluating hoi_id: {hoi_id} ...')
    select_global_ids = io.load_json_object(os.path.join(args.proc_dir, f'test_global_id_dict_select_by_score_{args.thres}.json'))
    pred_dets = h5py.File(pred_dets_hdf5,'r')
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    for global_id in global_ids:
        hoi_idx = int(hoi_id)-1
        obj = hoi_id_to_obj_id_list[hoi_idx]  # the object idx in this hoi
        obj_str = str(obj)
        object_global_id = select_global_ids[obj_str]  # a list that contains all the imgs which have this kind of object

        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)-1].astype(np.int)
        hoi_dets = \
            pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

        if global_id not in object_global_id:
            hoi_dets[:, 8] = 0
        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx,_ in sorted(
            zip(range(num_dets),hoi_dets[:,8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_dets[i, :4],
                'object_box': hoi_dets[i, 4:8],
                'score': hoi_dets[i, 8]
            }
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id,i))

    # Compute PR
    if len(y_score)==0:
        if npos==0:
            ap = np.nan
        else:
            ap = 0.0

    else:
        precision, recall = compute_pr(y_true, y_score, npos)
        # Compute AP
        ap = compute_ap(precision,recall)
    print(f'AP:{ap}')


    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap, hoi_id)


def eval_hoi_select_img_by_object_scores(hoi_id, global_ids, gt_dets, pred_dets_hdf5, out_dir, args):
    print(f'{args.mode} Evaluating hoi_id: {hoi_id} ...')
    select_global_ids = io.load_json_object(os.path.join(args.proc_dir, f'test_global_id_dict_select_by_score_{args.thres}.json'))
    pred_dets = h5py.File(pred_dets_hdf5,'r')
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    for global_id in global_ids:
        if global_id in loss_id:    # we find that some images have no ground truth so we delete them
            continue
        hoi_idx = int(hoi_id)-1
        obj = hoi_id_to_obj_id_list[hoi_idx]  # the object idx in this hoi
        obj_str = str(obj)
        object_global_id = select_global_ids[obj_str]  # a list that contains all the imgs which have this kind of object

        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)
        if global_id not in object_global_id:
            continue
        start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)-1].astype(np.int)
        hoi_dets = \
            pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx,_ in sorted(
            zip(range(num_dets),hoi_dets[:,8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_dets[i, :4],
                'object_box': hoi_dets[i, 4:8],
                'score': hoi_dets[i, 8]
            }
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id,i))

    # Compute PR
    if len(y_score)==0:
        if npos==0:
            ap = np.nan
        else:
            ap = 0.0

    else:
        precision, recall = compute_pr(y_true, y_score, npos)
        # Compute AP
        ap = compute_ap(precision,recall)
    print(f'AP:{ap}')


    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap,hoi_id)


def eval_positive(hoi_id, global_ids, gt_dets, pred_dets_hdf5, out_dir, hoi_lal):
    pred_dets = h5py.File(pred_dets_hdf5, 'r')
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    hoi_labels = h5py.File(os.path.join(hoi_cand_data_dir, f'hoi_candidate_labels_test.hdf5'), 'r')
    for global_id in global_ids:
        if global_id in loss_id:  # we find that some images have no ground truth so we delete them
            continue
        hoi_idx = int(hoi_id) - 1
        obj = hoi_id_to_obj_id_list[hoi_idx]  # the object idx in this hoi
        obj_str = str(obj)

        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id) - 1].astype(np.int)
        hoi_dets = \
            pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

        hoi_label = hoi_labels[global_id][()]
        hoi_label = hoi_label[start_id:end_id]
        tp_idx = np.where(hoi_label==1)[0]
        hoi_dets = hoi_dets[tp_idx]
        if hoi_dets.shape[0]==0:
            continue

        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx, _ in sorted(
            zip(range(num_dets), hoi_dets[:, 8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_dets[i, :4],
                'object_box': hoi_dets[i, 4:8],
                'score': hoi_dets[i, 8]
            }
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id, i))

    # Compute PR
    if len(y_score) == 0:
        if npos == 0:
            ap = np.nan
        else:
            ap = 0.0

    else:
        precision, recall = compute_pr(y_true, y_score, npos)
        # Compute AP
        ap = compute_ap(precision, recall)
    # print(f'AP:{ap}')

    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir, f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap, hoi_id)



def eval_hoi_known_object(hoi_id,global_ids, gt_dets, pred_dets_hdf5, out_dir, args):
    #print(f'{args.mode} Evaluating hoi_id: {hoi_id} ...')
    pred_dets = h5py.File(pred_dets_hdf5,'r')
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    for global_id in global_ids:
        if global_id in loss_id:    # we find that some images have no ground truth so we delete them
            continue
        hoi_idx = int(hoi_id)-1
        obj = hoi_id_to_obj_id_list[hoi_idx]  # the object idx in this hoi
        obj_name = object_name_list[obj]
        object_global_id = object_global_ids[obj_name]  # a list of img ids which contain this kind of object

        if global_id not in object_global_id:
            continue

        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)-1].astype(np.int)
        hoi_dets = \
            pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx,_ in sorted(
            zip(range(num_dets),hoi_dets[:,8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_dets[i, :4],
                'object_box': hoi_dets[i, 4:8],
                'score': hoi_dets[i, 8]
            }
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id,i))

    # Compute PR
    if len(y_score)==0:
        if npos==0:
            ap = np.nan
        else:
            ap = 0.0

    else:
        precision, recall = compute_pr(y_true, y_score, npos)
        # Compute AP
        ap = compute_ap(precision,recall)
    #print(f'AP:{ap}')


    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap,hoi_id)


def eval_no_interaction(hoi_id, global_ids, gt_dets,pred_dets_hdf5, out_dir, args):
    print(f'{args.mode} Evaluating hoi_id: {hoi_id} ...')
    pred_dets = h5py.File(pred_dets_hdf5,'r')
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    verb_idx = hoi_id_to_verb_id_list[int(hoi_id)-1]
    for global_id in global_ids:
        if global_id in loss_id:    # we find that some images have no ground truth so we delete them
            continue
        if verb_idx != no_interaction_verb_idx:
            continue
        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)-1].astype(np.int)
        hoi_dets = \
            pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx,_ in sorted(
            zip(range(num_dets), hoi_dets[:,8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_dets[i, :4],
                'object_box': hoi_dets[i, 4:8],
                'score': -hoi_dets[i, 8]
            }
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id,i))

    # Compute PR
    if len(y_score)==0:
        if npos==0:
            ap = np.nan
        else:
            ap = 0.0

    else:
        precision, recall = compute_pr(y_true, y_score, npos)
        # Compute AP
        ap = compute_ap(precision,recall)
    # print(f'AP:{ap}')


    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap,hoi_id)


def eval_hoi_my(hoi_id,global_ids,gt_dets,pred_dets_hdf5,out_dir, global_ids_list):
    print(f'Evaluating hoi_id: {hoi_id} ...')
    pred_dets = h5py.File(pred_dets_hdf5,'r')

    y_true = []
    y_score = []
    det_id = []
    npos = 0
    for global_id in global_ids_list:

        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        start_id, end_id = pred_dets[global_id]['start_end_ids'][int(hoi_id)-1].astype(np.int)
        hoi_dets = \
            pred_dets[global_id]['human_obj_boxes_scores'][start_id:end_id]

        num_dets = hoi_dets.shape[0]
        sorted_idx = [idx for idx, _ in sorted(
            zip(range(num_dets),hoi_dets[:,8].tolist()),
            key=lambda x: x[1],
            reverse=True)]
        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_dets[i, :4],
                'object_box': hoi_dets[i, 4:8],
                'score': hoi_dets[i, 8]
            }
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id,i))

        # Compute PR
    if len(y_score) == 0:
        if npos == 0:
            ap = np.nan
        else:
            ap = 0.0

    else:
        precision, recall = compute_pr(y_true, y_score, npos)
        # Compute AP
        ap = compute_ap(precision, recall)
    print(f'AP:{ap}')

    # Plot PR curve
    # plt.figure()
    # plt.step(recall,precision,color='b',alpha=0.2,where='post')
    # plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('Precision-Recall curve: AP={0:0.4f}'.format(ap))
    # plt.savefig(
    #     os.path.join(out_dir,f'{hoi_id}_pr.png'),
    #     bbox_inches='tight')
    # plt.close()

    # Save AP data
    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir,f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap,hoi_id)


def load_gt_dets(proc_dir, global_ids_set):
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_list_json = os.path.join(proc_dir,'anno_list.json')
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


def eval_hoi_ap(args, hoi_id, pred_dets_hdf5, global_ids_list, subset='test'):

    print('Creating output dir ...')
    io.mkdir_if_not_exists(args.out_dir, recursive=True)

    # Load hoi_list
    hoi_list_json = os.path.join(args.proc_dir, 'hoi_list.json')
    hoi_list = io.load_json_object(hoi_list_json)

    # Load subset ids to eval on
    split_ids_json = os.path.join(args.proc_dir, 'split_ids.json')
    split_ids = io.load_json_object(split_ids_json)
    global_ids = split_ids[subset]
    global_ids_set = set(global_ids)

    # Create gt_dets
    print('Creating GT dets ...')
    gt_dets = load_gt_dets(args.proc_dir, global_ids_set)

    ap, _ = eval_hoi_my(hoi_id, global_ids, gt_dets, pred_dets_hdf5, args.out_dir, global_ids_list)
    return ap


proc_dir = '/data/xubing/Human-Object_Interactions/code/no_frills/no_frills_hoi_det-release_v1/data_symlinks/hico_processed'
hoi_cand_data_dir = '/data/xubing/Human-Object_Interactions/code/no_frills/no_frills_hoi_det-release_v1/data_symlinks/hico_exp/hoi_candidates/'
object_global_ids = io.load_json_object(os.path.join(proc_dir, f'object_test_global_id_dict.json'))
path = os.path.join(proc_dir, 'hoi_id_to_coco_obj_id_list.json')
hoi_id_to_obj_id_list = io.load_json_object(path)
path = os.path.join(proc_dir, 'hoi_id_to_verb_id_list.json')
hoi_id_to_verb_id_list = io.load_json_object(path)
verb_name_list = io.load_json_object(os.path.join(proc_dir, 'verb_name_list.json'))
no_interaction_verb_idx = verb_name_list.index('no_interaction')
path = os.path.join(proc_dir, 'test_global_ids_have_no_gt.json')
loss_id = io.load_json_object(path)
object_name_list = io.load_json_object(os.path.join(proc_dir, 'object_name_list.json'))


def main():
    args = parser.parse_args()
    mode = args.mode

    print('Creating output dir ...')
    io.mkdir_if_not_exists(args.out_dir,recursive=True)

    # Load hoi_list
    hoi_list_json = os.path.join(args.proc_dir, 'hoi_list.json')
    hoi_list = io.load_json_object(hoi_list_json)

    # Load subset ids to eval on
    split_ids_json = os.path.join(args.proc_dir, 'split_ids.json')
    split_ids = io.load_json_object(split_ids_json)
    global_ids = split_ids[args.subset]
    global_ids_set = set(global_ids)

    # Create gt_dets
    print('Creating GT dets ...')
    gt_dets = load_gt_dets(args.proc_dir, global_ids_set)

    eval_inputs = []
    for hoi in hoi_list:
        eval_inputs.append(
            (hoi['id'], global_ids, gt_dets, args.pred_hoi_dets_hdf5, args.out_dir, args))

    print(f'Starting a pool of {args.num_processes} workers ...')
    p = Pool(args.num_processes)

    print(f'Begin mAP computation ...')
    if mode=='Default':
        output = p.starmap(eval_hoi, eval_inputs)
    elif mode=='Known-Object':
        output = p.starmap(eval_hoi_known_object, eval_inputs)
    elif mode=='select_img_by_object_scores':
        output = p.starmap(eval_hoi_select_img_by_object_scores, eval_inputs)
    elif mode=='select_img_by_object_scores_set_zero':
        output = p.starmap(eval_hoi_select_img_by_object_scores2, eval_inputs)
    elif mode=='eval_no_interaction':
        output = p.starmap(eval_no_interaction, eval_inputs)
    elif mode=='eval_positive':
        output = p.starmap(eval_positive, eval_inputs)
    else:
        raise ValueError
    p.close()
    p.join()

    mAP = {
        'AP': {},
        'mAP': 0,
        'invalid': 0,
    }
    map_ = 0
    count = 0
    rare_count = 0
    rare_mAP = 0
    rare_id_list = io.load_json_object(os.path.join(proc_dir, 'rare_id.json'))

    for ap, hoi_id in output:
        mAP['AP'][hoi_id] = ap
        if not np.isnan(ap):
            count += 1
            map_ += ap
            if hoi_id in rare_id_list:
                rare_mAP +=ap
                rare_count +=1


    mAP['mAP'] = map_ / count
    mAP['rare_mAP'] = rare_mAP/rare_count
    mAP['non_rare_mAP'] = (map_-rare_mAP)/(count-rare_count)


    mAP['mAP'] = round(mAP['mAP'] * 100, 2)
    mAP['rare_mAP'] = round(mAP['rare_mAP'] * 100, 2)
    mAP['non_rare_mAP'] = round(mAP['non_rare_mAP'] * 100, 2)

    save_dir = os.path.join(args.mAP_dir, 'mAP_json_file')
    io.mkdir_if_not_exists(save_dir)
    save_dir = os.path.join(save_dir, mode)
    io.mkdir_if_not_exists(save_dir)
    save_dir = os.path.join(save_dir, args.exp_name)
    io.mkdir_if_not_exists(save_dir)
    mAP_json = os.path.join(save_dir, f'mAP_{args.file_name}_{args.model_num}.json')
    io.dump_json_object(mAP, mAP_json)

    print(f'mAP_{args.exp_name}_{args.model_num}')
    print(f'APs have been saved to {save_dir}')
    print(mAP['mAP'], mAP['rare_mAP'], mAP['non_rare_mAP'])


if __name__=='__main__':

    main()

