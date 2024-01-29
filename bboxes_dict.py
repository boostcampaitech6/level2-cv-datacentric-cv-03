from metric import detect

def get_pred_bboxes_dict(images, image_fnames, input_size, score_maps, geo_maps,):
    by_sample_bboxes = []

    by_sample_bboxes.extend(detect(images, input_size, score_maps, geo_maps))

    pred_bboxes_dict = dict()
    for idx in range(len(image_fnames)):
        image_fname = image_fnames[idx]
        sample_bboxes = by_sample_bboxes[idx]
        pred_bboxes_dict[image_fname] = sample_bboxes

    return pred_bboxes_dict

def get_gt_bboxes_dict(ufo_dir, images):
    gt_bboxes_dict = dict()

    ufo_file = ufo_dir

    ufo_file_images = ufo_file['images']
    for image in images:
        gt_bboxes_dict[image] = []
        for idx in ufo_file_images[image]['words'].keys():
            gt_bboxes_dict[image].append(ufo_file_images[image]['words'][idx]['points'])

    return gt_bboxes_dict