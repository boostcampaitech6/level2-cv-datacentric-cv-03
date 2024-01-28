import os
import cv2
import json
from metric import detect

def get_pred_bboxes_dict(data_dir, images, input_size, score_maps, geo_maps, split='valid'):
    image_list, image_fnames, by_sample_bboxes = [], [], []

    for image in images:
        image_fpath = os.path.join(data_dir, f'img/{split}/{image}')
        image_fnames.append(os.path.basename(image_fpath))
        image_list.append(cv2.imread(image_fpath)[:, :, ::-1])

    by_sample_bboxes.extend(detect(image_list, input_size, score_maps, geo_maps))

    pred_bboxes_dict = dict()
    for idx in range(len(image_fnames)):
        image_fname = image_fnames[idx]
        sample_bboxes = by_sample_bboxes[idx]
        pred_bboxes_dict[image_fname] = sample_bboxes

    return pred_bboxes_dict

def get_gt_bboxes_dict(ufo_dir, images):
    gt_bboxes_dict = dict()

    with open(ufo_dir, 'r') as f:
        ufo_file = json.load(f)

    ufo_file_images = ufo_file['images']
    for image in images:
        gt_bboxes_dict[image] = []
        for idx in ufo_file_images[image]['words'].keys():
            gt_bboxes_dict[image].append(ufo_file_images[image]['words'][idx]['points'])

    return gt_bboxes_dict