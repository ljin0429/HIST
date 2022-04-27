import torch, math, time, argparse, os, sys
import random, dataset, utils, net
import numpy as np

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb


parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    default='cars',
                    help='Training dataset, e.g. cub, cars, SOP'
                    )

parser.add_argument('--embedding-size',
                    default=512,
                    type=int,
                    dest='sz_embedding',
                    help='Size of embedding that is appended to backbone model.'
                    )

parser.add_argument('--batch-size',
                    default=150,
                    type=int,
                    dest='sz_batch',
                    help='Number of samples per batch.'
                    )

parser.add_argument('--gpu-id',
                    default=0,
                    type=int,
                    help='ID of GPU that is used for training.'
                    )

parser.add_argument('--workers',
                    default=4,
                    type=int,
                    dest='nb_workers',
                    help='Number of workers for dataloader.'
                    )

parser.add_argument('--model',
                    default='resnet50',
                    help='Model for training'
                    )

parser.add_argument('--layer-norm',
                    default=1,
                    type=int,
                    help='layer normlization'
                    )

parser.add_argument('--model-path',
                    default='',
                    help='Path of the pre-trained model'
                    )

parser.add_argument('--remark',
                    default='',
                    help='Any reamrk'
                    )

args = parser.parse_args()


torch.cuda.set_device(args.gpu_id)


# Data root directory
data_root = '../data'


# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

else:
    query_dataset = Inshop_Dataset(
        root=data_root,
        mode='query',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

    gallery_dataset = Inshop_Dataset(
        root=data_root,
        mode='gallery',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

""" Original code (single model evaluation) """
# Trained model root directory
checkpoint_DIR = '{}/{}_resnet50_best.pth'.format(args.model_path, args.dataset)


# Backbone Model
if args.model.find('googlenet') + 1:
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.layer_norm, bn_freeze=1)
elif args.model.find('bn_inception') + 1:
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.layer_norm, bn_freeze=1)
elif args.model.find('resnet50') + 1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.layer_norm, bn_freeze=1)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

if os.path.isfile(checkpoint_DIR):
    print('=> loading checkpoint {}'.format(checkpoint_DIR))
    checkpoint = torch.load(checkpoint_DIR, map_location='cuda:{}'.format(args.gpu_id))
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> No checkpoint found at {}'.format(checkpoint_DIR))
    sys.exit(0)

with torch.no_grad():
    print("**Evaluating...**")
    if args.dataset == 'Inshop':
        Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)

    elif args.dataset != 'SOP':
        Recalls, _ = utils.evaluate_cos(model, dl_ev, eval_nmi=False)
        # For evaluating RP and MAP@R, use the following code...
        # RP, MAP = utils.evaluate_Rstat(model, dl_ev)

    else:
        Recalls, _ = utils.evaluate_cos_SOP(model, dl_ev, eval_nmi=False)
        # For evaluating RP and MAP@R, use the following code...
        # RP, MAP = utils.evaluate_Rstat_SOP(model, dl_ev)

