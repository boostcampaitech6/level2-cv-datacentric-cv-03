import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import wandb

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'datasets/data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])

    parser.add_argument('--name', type=str, default='exp')

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, name):
    run = wandb.init(
        project="OCR", 
        entity = "funfun_ocr",
        name=name,
        config = vars(args)
    )

    train_dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )
    valid_dataset = SceneTextDataset(
        data_dir,
        split='valid',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )

    train_dataset = EASTDataset(train_dataset)
    valid_dataset = EASTDataset(valid_dataset)

    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    valid_num_batches = math.ceil(len(valid_dataset) / (batch_size//2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size//2,
        shuffle=False,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    best_loss = 1e9
    for epoch in range(max_epoch):
        model.train()
        train_epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Train Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                train_epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        train_loss = train_epoch_loss / train_num_batches
        print('Train Mean loss: {:.4f} | Elapsed time: {}'.format(train_loss, timedelta(seconds=time.time() - epoch_start)))

        model.eval()
        with torch.no_grad():
            with tqdm(total=valid_num_batches) as pbar:
                valid_epoch_loss, epoch_start = 0, time.time()
                for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
                    pbar.set_description('[Valid Epoch {}]'.format(epoch + 1))

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                    loss_val = loss.item()
                    valid_epoch_loss += loss_val

                    pbar.update(1)
                    val_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }
                    pbar.set_postfix(val_dict)
            val_loss = valid_epoch_loss / valid_num_batches    
            print('Valid Mean loss: {:.4f} | Elapsed time: {}'.format(val_loss, timedelta(seconds=time.time() - epoch_start)))

            run.log({
                    'epochs':epoch,
                    'train_loss':train_loss,
                    'val_loss':val_loss
                    })

            if (epoch + 1) % save_interval == 0:
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                if best_loss > val_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), osp.join(model_dir, 'best.pth'))
                    artifact = wandb.Artifact('best.pth',type='model')
                    artifact.add_file(osp.join(model_dir, 'best.pth'))
                    run.log_artifact(artifact)

                torch.save(model.state_dict(), osp.join(model_dir, 'latest.pth'))
    
    run.finish()
            

def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
