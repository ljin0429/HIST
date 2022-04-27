import torch, math, time, argparse, os, random
import torch.nn as nn
import numpy as np

from torch.utils.data.sampler import BatchSampler
import dataset
from dataset import sampler
from dataset.Inshop import Inshop_Dataset

import utils
from hist import *

import net
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *

from tqdm import *
import wandb

parser = argparse.ArgumentParser()

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

parser.add_argument('--dataset',
                    default='cars',
                    help='Training dataset, e.g. cub, cars, SOP'
                    )

parser.add_argument('--model',
                    default='resnet50',
                    help='Model for training, e.g. bn_inception, resnet50'
                    )

parser.add_argument('--embedding-size',
                    default=512,
                    type=int,
                    dest='sz_embedding',
                    help='Size of embedding that is appended to backbone model.'
                    )

parser.add_argument('--hgnn-hidden',
                    default=512,
                    type=int,
                    help='Size of hidden units in HGNN.'
                    )

parser.add_argument('--add-gmp',
                    default=1,
                    type=int,
                    help='if 1, add GMP feature, else if set to 0, only use GAP.'
                    )

parser.add_argument('--batch-size',
                    default=32,
                    type=int,
                    dest='sz_batch',
                    help='Number of samples per batch.'
                    )

parser.add_argument('--epochs',
                    default=50,
                    type=int,
                    dest='nb_epochs',
                    help='Number of training epochs.'
                    )

parser.add_argument('--optimizer',
                    default='adam',
                    help='Optimizer setting'
                    )

parser.add_argument('--lr',
                    default=1e-4,
                    type=float,
                    help='Learning rate for embedding network parameters'
                    )

parser.add_argument('--lr-ds',
                    default=1e-1,
                    type=float,
                    help='Learning rate for class prototypical distribution parameters'
                    )

parser.add_argument('--lr-hgnn-factor',
                    default=10,
                    type=float,
                    help='Learning rate multiplication factor for HGNN parameters'
                    )

parser.add_argument('--weight-decay',
                    default=1e-4,
                    type=float,
                    help='Weight decay setting'
                    )

parser.add_argument('--lr-decay-step',
                    default=10,
                    type=int,
                    help='Learning decay step setting'
                    )

parser.add_argument('--lr-decay-gamma',
                    default=0.5,
                    type=float,
                    help='Learning decay gamma setting'
                    )

parser.add_argument('--tau',
                    default=32,
                    type=float,
                    help='temperature scale parameter for softmax'
                    )

parser.add_argument('--alpha',
                    default=0.9,
                    type=float,
                    help='hardness scale parameter for construction of H'
                    )

parser.add_argument('--ls',
                    default=1,
                    type=float,
                    help='loss scale balancing parameters (lambda_s)'
                    )

parser.add_argument('--IPC',
                    default=0,
                    type=int,
                    help='Balanced sampling, images per class'
                    )

parser.add_argument('--warm',
                    default=1,
                    type=int,
                    help='Warmup training epochs, if set to 0, do not warm up'
                    )

parser.add_argument('--bn-freeze',
                    default=1,
                    type=int,
                    help='Batch normalization parameter freeze, if set to 0, do not freeze'
                    )

parser.add_argument('--layer-norm',
                    default=1,
                    type=int,
                    help='Layer normalization'
                    )

parser.add_argument('--remark',
                    default='',
                    help='Any remark'
                    )

parser.add_argument('--run-num',
                    default=1,
                    type=int,
                    help='The number of repetitive run'
                    )

args = parser.parse_args()


# Set fixed random seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


torch.cuda.set_device(args.gpu_id)


# Directory for Logging
log_folder_name = 'run_{}'.format(args.run_num)
LOG_DIR = '../res/{}/{}'.format(args.dataset, log_folder_name)

if not os.path.exists('{}'.format(LOG_DIR)):
    os.makedirs('{}'.format(LOG_DIR))


# Wandb Initialization
wb_project_name = 'hist_{}'.format(args.dataset)
wandb.init(project=wb_project_name, notes=LOG_DIR)
wandb.config.update(args)
wandb.run.name = '{}'.format(log_folder_name)


# Data root directory
data_root = '../data'


# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    trn_dataset = dataset.load(name=args.dataset,
                               root=data_root,
                               mode='train',
                               transform=dataset.utils.make_transform(is_train=True,
                                                                      is_inception=(args.model == 'bn_inception')
                                                                      )
                               )
else:
    trn_dataset = Inshop_Dataset(root=data_root,
                                 mode='train',
                                 transform=dataset.utils.make_transform(is_train=True,
                                                                        is_inception=(args.model == 'bn_inception')
                                                                        )
                                 )

if args.IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset,
                                               batch_size=args.sz_batch,
                                               images_per_class=args.IPC
                                               )

    batch_sampler = BatchSampler(balanced_sampler,
                                 batch_size=args.sz_batch,
                                 drop_last=True
                                 )

    dl_tr = torch.utils.data.DataLoader(trn_dataset,
                                        num_workers=args.nb_workers,
                                        pin_memory=True,
                                        batch_sampler=batch_sampler
                                        )
    print('Balanced Sampling')
else:
    dl_tr = torch.utils.data.DataLoader(trn_dataset,
                                        batch_size=args.sz_batch,
                                        shuffle=True,
                                        num_workers=args.nb_workers,
                                        drop_last=True,
                                        pin_memory=True
                                        )
    print('Random Sampling')

if args.dataset != 'Inshop':
    ev_dataset = dataset.load(name=args.dataset,
                              root=data_root,
                              mode='eval',
                              transform=dataset.utils.make_transform(is_train=False,
                                                                     is_inception=(args.model == 'bn_inception')
                                                                     )
                              )

    dl_ev = torch.utils.data.DataLoader(ev_dataset,
                                        batch_size=args.sz_batch,
                                        shuffle=False,
                                        num_workers=args.nb_workers,
                                        pin_memory=True
                                        )
else:
    query_dataset = Inshop_Dataset(root=data_root,
                                   mode='query',
                                   transform=dataset.utils.make_transform(is_train=False,
                                                                          is_inception=(args.model == 'bn_inception')
                                                                          )
                                   )

    dl_query = torch.utils.data.DataLoader(query_dataset,
                                           batch_size=args.sz_batch,
                                           shuffle=False,
                                           num_workers=args.nb_workers,
                                           pin_memory=True
                                           )

    gallery_dataset = Inshop_Dataset(root=data_root,
                                     mode='gallery',
                                     transform=dataset.utils.make_transform(is_train=False,
                                                                            is_inception=(args.model == 'bn_inception')
                                                                            )
                                     )

    dl_gallery = torch.utils.data.DataLoader(gallery_dataset,
                                             batch_size=args.sz_batch,
                                             shuffle=False,
                                             num_workers=args.nb_workers,
                                             pin_memory=True
                                             )

nb_classes = trn_dataset.nb_classes()


# Feature Embedding (Backbone)
if args.model.find('googlenet') + 1:
    model = googlenet(embedding_size=args.sz_embedding,
                      pretrained=True,
                      is_norm=args.layer_norm,
                      bn_freeze=args.bn_freeze,
                      add_gmp=args.add_gmp
                      )

elif args.model.find('bn_inception') + 1:
    model = bn_inception(embedding_size=args.sz_embedding,
                         pretrained=True,
                         is_norm=args.layer_norm,
                         bn_freeze=args.bn_freeze,
                         add_gmp=args.add_gmp
                         )

elif args.model.find('resnet50') + 1:
    model = Resnet50(embedding_size=args.sz_embedding,
                     pretrained=True,
                     is_norm=args.layer_norm,
                     bn_freeze=args.bn_freeze,
                     add_gmp=args.add_gmp
                     )

model = model.cuda()


# Class Prototypical Distributions -> Hypergraph model
d2hg = CDs2Hg(nb_classes=nb_classes,
              sz_embed=args.sz_embedding,
              tau=args.tau,
              alpha=args.alpha)
d2hg.cuda()


# Hypergraph Neural Network
hnmp = HGNN(nb_classes=nb_classes,
            sz_embed=args.sz_embedding,
            hidden=args.hgnn_hidden)
hnmp.cuda()


# Overall train parameters
param_groups = []
param_groups.append({'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))),
                     'lr': args.lr})
param_groups.append({'params': model.model.embedding.parameters(),
                     'lr': args.lr})
param_groups.append({'params': d2hg.parameters(),
                     'lr': args.lr_ds})
param_groups.append({'params': hnmp.parameters(),
                     'lr': args.lr * args.lr_hgnn_factor})


# Optimizer Setting
if args.optimizer == 'sgd':
    opt = torch.optim.SGD(param_groups,
                          lr=float(args.lr),
                          weight_decay=args.weight_decay,
                          momentum=0.9,
                          nesterov=True
                          )

elif args.optimizer == 'adam':
    opt = torch.optim.Adam(param_groups,
                           lr=float(args.lr),
                           weight_decay=args.weight_decay
                           )

elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups,
                              lr=float(args.lr),
                              alpha=0.9,
                              weight_decay=args.weight_decay,
                              momentum=0.9
                              )

elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups,
                            lr=float(args.lr),
                            weight_decay=args.weight_decay
                            )

scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)


print("Training arguments: {}".format(vars(args)))
print("===============================")

losses_list = []
best_recall = [0]
best_epoch = 0
break_out_flag = False

""" Warm up: Train only new params, helps stabilize learning. """
if args.warm > 0:
    print("** Warm up training for {} epochs... **".format(args.warm))
    freeze_params = param_groups[0]['params']

    for epoch in range(0, args.warm):

        model.train()
        losses_per_epoch = []

        # BN freeze
        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = model.model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Freeze backbone params
        for param in freeze_params:
            param.requires_grad = False

        pbar = tqdm(enumerate(dl_tr))

        for batch_idx, (x, y) in pbar:
            m = model(x.squeeze().cuda())
            target = y.squeeze().cuda()

            # Hypergraph construction & distribution loss
            dist_loss, H = d2hg(m, target)
            H.cuda()

            # Hypergraph node classification
            out = hnmp(m, H)
            criterion = nn.CrossEntropyLoss()
            ce_loss = criterion(out, target)

            loss = dist_loss + args.ls * ce_loss

            opt.zero_grad()
            loss.backward()

            opt.step()

            pbar.set_description(
                'Warm-up Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx + 1, len(dl_tr),
                           100. * batch_idx / len(dl_tr),
                    loss.item()))

            losses_per_epoch.append(loss.data.cpu().numpy())

            if np.isnan(losses_per_epoch[-1]):
                break_out_flag = True
                break

        if break_out_flag:
            print("** Failed training (NaN Loss)... **")
            break

        if epoch >= 0:
            with torch.no_grad():
                print("** Evaluating... **")
                if args.dataset == 'Inshop':
                    Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
                elif args.dataset != 'SOP':
                    Recalls, NMIs = utils.evaluate_cos(model, dl_ev, eval_nmi=False)
                else:
                    Recalls, NMIs = utils.evaluate_cos_SOP(model, dl_ev, eval_nmi=False)

    # Unfreeze backbone params
    for param in freeze_params:
        param.requires_grad = True

    print("** Warm up training done... **")


print("===============================")
print("** Training for {} epochs... **".format(args.nb_epochs))
for epoch in range(0, args.nb_epochs):

    model.train()
    losses_per_epoch = []

    # BN freeze
    bn_freeze = args.bn_freeze
    if bn_freeze:
        modules = model.model.modules()
        for m in modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (x, y) in pbar:
        m = model(x.squeeze().cuda())
        target = y.squeeze().cuda()

        # Hypergraph construction & distribution loss
        dist_loss, H = d2hg(m, target)
        H.cuda()

        # Hypergraph node classification
        out = hnmp(m, H)
        criterion = nn.CrossEntropyLoss()
        ce_loss = criterion(out, target)

        loss = dist_loss + args.ls * ce_loss

        opt.zero_grad()
        loss.backward()

        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))

        losses_per_epoch.append(loss.data.cpu().numpy())

        if np.isnan(losses_per_epoch[-1]):
            break_out_flag = True
            break

    if break_out_flag:
        print("** Failed training (NaN Loss)... **")
        break

    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()

    if epoch >= 0:
        with torch.no_grad():
            print("** Evaluating... **")
            if args.dataset == 'Inshop':
                Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
            elif args.dataset != 'SOP':
                Recalls, NMIs = utils.evaluate_cos(model, dl_ev, eval_nmi=False)
            else:
                Recalls, NMIs = utils.evaluate_cos_SOP(model, dl_ev, eval_nmi=False)

        # Logging Evaluation Score
        if args.dataset == 'Inshop':
            for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                wandb.log({"R@{}".format(K): Recalls[i]}, step=epoch)
        elif args.dataset != 'SOP':
            for i in range(6):
                wandb.log({"R@{}".format(2 ** i): Recalls[i]}, step=epoch)
        else:
            for i in range(4):
                wandb.log({"R@{}".format(10 ** i): Recalls[i]}, step=epoch)

        # Best model
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch

            # Save model
            torch.save({'model_state_dict': model.state_dict()},
                       '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model))

            with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                f.write('Parameters: {}\n\n'.format(vars(args)))
                f.write('Best Epoch: {}\n'.format(best_epoch))
                if args.dataset == 'Inshop':
                    for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                        f.write("Best Recall@{}: {:.4f}\n".format(K, best_recall[i] * 100))
                elif args.dataset != 'SOP':
                    for i in range(6):
                        f.write("Best Recall@{}: {:.4f}\n".format(2 ** i, best_recall[i] * 100))
                else:
                    for i in range(4):
                        f.write("Best Recall@{}: {:.4f}\n".format(10 ** i, best_recall[i] * 100))

