import sys
import os
import pdb
import argparse
import time
from collections import OrderedDict, defaultdict

sys.path.append('/cluster/yinan/isc2021')

from PIL import Image
from isc.io import write_hdf5_descriptors, read_ground_truth, read_predictions, write_predictions, read_descriptors

import torch
import torchvision
import torchvision.transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import glob
import faiss
import random
import tempfile
import numpy as np
import h5py
import copy
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT
import timm
from pprint import pprint
import pandas as pd

QUERY = '/cluster/shared_dataset/isc2021/query_images/'
REFERENCE = '/cluster/shared_dataset/isc2021/reference_images/'
TRAIN = '/cluster/shared_dataset/isc2021/training_images/training_images/'
CHECK = 'isc2021/data/multigrain_joint_3B_0.5.pth'


def load_model(name, checkpoint_file):
    if name == "zoo_resnet50":
        print('--------------------------------------------------------------')
        print('used model: zoo_resnet50')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet50(pretrained=True)
        model.eval()
        return model

    if name == "multigrain_resnet50":
        print('--------------------------------------------------------------')
        print('used model: multigrain_resnet50')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet50(pretrained=False)
        st = torch.load(checkpoint_file)
        state_dict = OrderedDict([
            (name[9:], v)
            for name, v in st["model_state"].items() if name.startswith("features.")
        ])
        model.fc
        model.fc = None
        model.load_state_dict(state_dict)
        model.eval()
        return model

    if name == "vgg":
        print('--------------------------------------------------------------')
        print('used model: VGG16')
        print('--------------------------------------------------------------')
        model = torchvision.models.vgg16(pretrained=True)
        model.eval()
        return model

    if name == "resnet152":
        print('--------------------------------------------------------------')
        print('used model: ResNet152')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet152(pretrained=True)
        model.eval()
        return model

    if name == "efficientnetb1":
        print('--------------------------------------------------------------')
        print('used model: EfficientNet-b1')
        print('--------------------------------------------------------------')
        model = EfficientNet.from_pretrained('efficientnet-b1')
        model.eval()
        return model

    if name == "efficientnetb7":
        print('--------------------------------------------------------------')
        print('used model: EfficientNet-b7')
        print('--------------------------------------------------------------')
        model = EfficientNet.from_pretrained('efficientnet-b7')
        model.eval()
        return model

    if name == "transformer":
        print('--------------------------------------------------------------')
        print('used model: ViT')
        print('--------------------------------------------------------------')
        model = ViT('B_16_imagenet1k', pretrained=True)
        model.eval()
        return model

    if name == "visformer":
        print('--------------------------------------------------------------')
        print('used model: vit_large_patch16_384')
        print('--------------------------------------------------------------')
        model = timm.create_model('vit_large_patch16_384', pretrained=True)
        model.eval()
        return model

    assert False


def gem_npy(x, p=3, eps=1e-6):
    x = np.clip(x, a_min=eps, a_max=np.inf)
    x = x ** p
    x = x.mean(axis=0)
    return x ** (1. / p)


def generate_train_dataset(query_list, gt_list, train_list, len_data):
    # TODO: generate training list with length len_data
    random.seed(1)
    t_list = list()
    gt_list = gt_list[0: int(len(gt_list)/2)]
    for i in range(len_data):
        label = random.randint(0, 1)
        if label == 0:
            gt = random.sample(gt_list, 1)[0]
            q = gt.query
            r = gt.db
            q = QUERY + q + ".jpg"
            r = REFERENCE + r + ".jpg"
            t_list.append((q, r, label))
        else:
            q = random.sample(query_list, 1)[0]
            r = random.sample(train_list, 1)[0]
            q = QUERY + q + ".jpg"
            t = TRAIN + r + ".jpg"
            t_list.append((q, t, label))
    return t_list


def generate_validation_dataset(query_list, gt_list, train_list, len_data):
    # TODO: generate training list with length len_data
    random.seed(3)
    v_list = list()
    gt_list = gt_list[int(len(gt_list) / 2): -1]
    for i in range(len_data):
        label = random.randint(0, 1)
        if label == 0:
            gt = random.sample(gt_list, 1)[0]
            q = gt.query
            r = gt.db
            q = QUERY + q + ".jpg"
            r = REFERENCE + r + ".jpg"
            v_list.append((q, r, label))
        else:
            q = random.sample(query_list, 1)[0]
            r = random.sample(train_list, 1)[0]
            q = QUERY + q + ".jpg"
            t = TRAIN + r + ".jpg"
            v_list.append((q, t, label))
    return v_list


def generate_extraction_dataset(query_list, db_list):
    query_images = [QUERY + q + ".jpg" for q in query_list]
    db_images = [REFERENCE + r + ".jpg" for r in db_list]
    return query_images, db_images


class SiameseNetwork(nn.Module):
    def __init__(self, model):
        super(SiameseNetwork, self).__init__()
        self.head = load_model(model, CHECK)
        for p in self.parameters():
            p.requires_grad = False
        if model == "zoo_resnet50" or model == "multigrain_resnet50" or model == "resnet152":
            self.map = True
        else:
            self.map = False
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.score = nn.PairwiseDistance(p=2)

    def forward_once(self, x):
        if self.map:
            x = self.head.conv1(x)
            x = self.head.bn1(x)
            x = self.head.relu(x)
            x = self.head.maxpool(x)

            x = self.head.layer1(x)
            x = self.head.layer2(x)
            x = self.head.layer3(x)
            x = self.head.layer4(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = self.flatten(x)
            output = self.fc1(x)
        else:
            x = self.head(x)
            output = self.fc2(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        score = self.score(output1, output2)
        return score


class ContrastiveLoss(torch.nn.Module):
    def __int__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, score, label):
        loss = torch.mean((1 - label) * 0.5 * torch.pow(score, 2) +
                          label * 0.5 * torch.pow(torch.clamp(2.0 - score, min=0.0), 2))
        return loss


class ImageList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        x = Image.open(self.image_list[i])
        x = x.convert("RGB")
        # if self.imsize is not None:
        #     x.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            x = self.transform(x)
        return x


class TrainList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        # x = Image.open(self.image_list[i])
        # x = x.convert("RGB")
        # if self.imsize is not None:
        #     x.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        # if self.transform is not None:
        #     x = self.transform(x)
        # return x
        q, r, label = self.image_list[i]
        query_image = Image.open(q)
        db_image = Image.open(r)
        query_image = query_image.convert("RGB")
        db_image = db_image.convert("RGB")
        if self.transform is not None:
            query_image = self.transform(query_image)
            db_image = self.transform(db_image)
        return query_image, db_image, label


def main():

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)



    group = parser.add_argument_group('feature extraction options')
    aa('--transpose', default=-1, type=int, help="one of the 7 PIL transpose options ")
    aa('--train', default=False, action="store_true", help="run Siamese training")
    aa('--device', default="cuda:0", help='pytroch device')
    aa('--batch_size', default=32, type=int, help="max batch size to use for extraction")
    aa('--num_workers', default=8, type=int, help="nb of dataloader workers")

    group = parser.add_argument_group('model options')
    aa('--model', default='multigrain_resnet50', help="model to use")
    aa('--checkpoint', default='Siamese_Epoch_4.pth', help='best saved model name')
    aa('--GeM_p', default=7.0, type=float, help="Power used for GeM pooling")
    aa('--scales', default="1.0", help="scale levels")
    aa('--imsize', default=512, type=int, help="max image size at extraction time")

    group = parser.add_argument_group('dataset options')
    aa('--query_list', required=True, help="file with  query image filenames")
    aa('--gt_list', required=True, help="file with reference image filenames")
    aa('--train_list', required=True, help="file with training image filenames")
    aa('--db_list', required=True, help="file with training image filenames")
    aa('--len', default=1000, type=int, help="nb of training vectors for the SiameseNetwork")
    aa('--epoch', default=100, type=int, help="nb of training epochs for the SiameseNetwork")
    aa('--i0', default=0, type=int, help="first image to process")
    aa('--i1', default=-1, type=int, help="last image to process + 1")

    group = parser.add_argument_group('output options')
    aa('--query_f', default="isc2021/data/query_siamese.hdf5", help="write query features to this file")
    aa('--db_f', default="isc2021/data/db_siamese.hdf5", help="write query features to this file")
    aa('--net', default="isc2021/checkpoints/Siamese/", help="save network parameters to this file")

    args = parser.parse_args()
    args.scales = [float(x) for x in args.scales.split(",")]

    print("args=", args)


    if args.device == "cpu":
        if 'Linux' in platform.platform():
            os.system(
                'echo hardware_image_description: '
                '$( cat /proc/cpuinfo | grep ^"model name" | tail -1 ), '
                '$( cat /proc/cpuinfo | grep ^processor | wc -l ) cores'
            )
        else:
            print("hardware_image_description:", platform.machine(), "nb of threads:", args.nproc)
    else:
        print("hardware_image_description:", torch.cuda.get_device_name(0))

    # image_list = [l.strip() for l in open(args.file_list, "r")]
    #
    # if args.i1 == -1:
    #     args.i1 = len(image_list)
    # image_list = image_list[args.i0:args.i1]
    #
    # # full path name for the image
    # image_dir = args.image_dir
    # if not image_dir.endswith('/'):
    #     image_dir += "/"
    #
    # image_list = [image_dir + fname for fname in image_list]
    #
    # print("  found %d images" % (len(image_list)))

    gt_list = read_ground_truth(args.gt_list)
    query_list = [l.strip() for l in open(args.query_list, "r")]
    db_list = [l.strip() for l in open(args.db_list, "r")]
    train_list = [l.strip() for l in open(args.train_list, "r")]

    # transform
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if args.model == "transformer" or args.model == "visformer":
        transforms = [
            torchvision.transforms.Resize((384, 384)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]
    else:
        transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]

    if args.transpose != -1:
        transforms.insert(TransposeTransform(args.transpose), 0)

    transforms = torchvision.transforms.Compose(transforms)

    if args.train:
        print("training network")
        t_list = generate_train_dataset(query_list, gt_list, train_list, args.len)
        v_list = generate_validation_dataset(query_list, gt_list, train_list, 200)
        print(f"subsampled {args.len} vectors")

        # im_pairs = TrainPairs(t_list, transform=transforms, imsize=args.imsize)
        im_pairs = TrainList(t_list, transform=transforms, imsize=args.imsize)
        val_pairs = TrainList(v_list, transform=transforms, imsize=args.imsize)
        train_dataloader = DataLoader(dataset=im_pairs, shuffle=True, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
        val_dataloader = DataLoader(dataset=val_pairs, shuffle=True, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
        print("loading model")
        net = SiameseNetwork(args.model)
        net.to(args.device)
        criterion = ContrastiveLoss()
        criterion.to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, weight_decay=0.0)
        loss_history = list()
        epoch_losses = list()
        for epoch in range(args.epoch):
            for i, data in enumerate(train_dataloader, 0):
                q_img, r_img, label = data
                q_img_cp = copy.deepcopy(q_img)
                r_img_cp = copy.deepcopy(r_img)
                label_cp = copy.deepcopy(label)
                q_img = q_img_cp.to(args.device)
                r_img = r_img_cp.to(args.device)
                label = label_cp.to(args.device)
                output = net(q_img, r_img)
                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                loss_history.append(loss)

                if (i+1) % 100 == 0:
                    mean_loss = torch.mean(torch.Tensor(loss_history))
                    loss_history.clear()
                    val_loss = 0.0
                    print("Epoch:{},  Current training loss {}\n".format(epoch, mean_loss))
                    with torch.no_grad():
                        for j, data in enumerate(val_dataloader, 0):
                            q_img, r_img, label = data
                            q_img_cp = copy.deepcopy(q_img)
                            r_img_cp = copy.deepcopy(r_img)
                            label_cp = copy.deepcopy(label)
                            q_img = q_img_cp.to(args.device)
                            r_img = r_img_cp.to(args.device)
                            label = label_cp.to(args.device)
                            output = net(q_img, r_img)
                            val_loss += criterion(output, label)
                        val_loss /= 200
                    print("Epoch:{},  Current validation loss {}\n".format(epoch, val_loss))
            epoch_losses.append(val_loss.cpu())

            trained_model_name = 'Siamese_Epoch_{}.pth'.format(epoch)
            model_full_path = args.net + trained_model_name
            torch.save(net.state_dict(), model_full_path)
            print('model saved as: {}\n'.format(trained_model_name))

        epoch_losses = np.asarray(epoch_losses)
        best_epoch = np.argmin(epoch_losses)
        best_model_name = 'Siamese_Epoch_{}.pth'.format(best_epoch)
        pth_files = glob.glob(args.net + '*.pth')
        pth_files.remove(args.net + best_model_name)
        for file in pth_files:
            os.remove(file)
        print("best model is: {}\n".format(best_model_name))

    else:
        print("computing features")
        query_images, db_images = generate_extraction_dataset(query_list, db_list)
        query_dataset = ImageList(query_images, transform=transforms)
        db_dataset = ImageList(db_list, transform=transforms)

        net = SiameseNetwork(args.model)
        state_dict = torch.load(args.net + args.checkpoint)
        net.load_state_dict(state_dict)
        net.to(args.device)
        print("checkpoint {} loaded\n".format(args.checkpoint))
        with torch.no_grad():
            query_feat, db_feat = list(), list()
            for i, x in enumerate(query_dataset):
                x_cp = copy.deepcopy(x)
                x = x_cp.to(args.device)
                x = x.unsqueeze(0)
                o = net.forward_once(x)
                print(o.size)
                o = torch.squeeze(o)
                print(o.size)
                break

            for i, x in enumerate(db_dataset):
                x_cp = copy.deepcopy(x)
                x = x_cp.to(args.device)
                x = x.unsqueeze(0)
                o = net.forward_once(x)
                break

    # im_dataset = ImageList(image_list, transform=transforms, imsize=args.imsize)
    #
    # print("loading model")
    # net = load_model(args.model, args.checkpoint)
    # net.to(args.device)
    #
    # print("computing features")
    #
    # t0 = time.time()
    #
    # with torch.no_grad():
    #     if args.batch_size == 1:
    #         all_desc = []
    #         for no, x in enumerate(im_dataset):
    #             x = x.to(args.device)
    #             print(f"im {no}/{len(im_dataset)}    ", end="\r", flush=True)
    #             x = x.unsqueeze(0)
    #             feats = []
    #             for s in args.scales:
    #                 xs = nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
    #                 o = resnet_activation_map(net, xs)
    #                 o = o.cpu().numpy()    # B, C, H, W
    #                 o = o[0].reshape(o.shape[1], -1).T
    #                 feats.append(o)
    #
    #             feats = np.vstack(feats)
    #             gem = gem_npy(feats, p=args.GeM_p)
    #             all_desc.append(gem)
    #
    #     else:
    #         all_desc = [None] * len(im_dataset)
    #         ndesc = [0]
    #         buckets = defaultdict(list)
    #
    #         def handle_bucket(bucket):
    #             ndesc[0] += len(bucket)
    #             x = torch.stack([xi for no, xi in bucket])
    #             x = x.to(args.device)
    #             print(f"ndesc {ndesc[0]} / {len(all_desc)} handle bucket of shape {x.shape}\r", end="", flush=True)
    #             feats = []
    #             for s in args.scales:
    #                 xs = nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
    #                 o = resnet_activation_map(net, xs)
    #                 o = o.cpu().numpy()    # B, C, H, W
    #                 feats.append(o)
    #
    #             for i, (no, _) in enumerate(bucket):
    #                 feats_i = np.vstack([f[i].reshape(f[i].shape[0], -1).T for f in feats])
    #                 gem = gem_npy(feats_i, p=args.GeM_p)
    #                 all_desc[no] = gem
    #
    #         max_batch_size = args.batch_size
    #
    #         dataloader = torch.utils.data.DataLoader(
    #             im_dataset, batch_size=1, shuffle=False,
    #             num_workers=args.num_workers
    #         )
    #
    #         for no, x in enumerate(dataloader):
    #             x = x[0]  # don't batch
    #             buckets[x.shape].append((no, x))
    #
    #             if len(buckets[x.shape]) >= max_batch_size:
    #                 handle_bucket(buckets[x.shape])
    #                 del buckets[x.shape]
    #
    #         for bucket in buckets.values():
    #             handle_bucket(bucket)
    #
    # all_desc = np.vstack(all_desc)
    #
    # t1 = time.time()
    #
    # print()
    # print(f"image_description_time: {(t1 - t0) / len(image_list):.5f} s per image")
    #
    # if args.train_pca:
    #     d = all_desc.shape[1]
    #     pca = faiss.PCAMatrix(d, args.pca_dim, -0.5)
    #     print(f"Train PCA {pca.d_in} -> {pca.d_out}")
    #     pca.train(all_desc)
    #     print(f"Storing PCA to {args.pca_file}")
    #     faiss.write_VectorTransform(pca, args.pca_file)
    # elif args.pca_file:
    #     print("Load PCA matrix", args.pca_file)
    #     pca = faiss.read_VectorTransform(args.pca_file)
    #     print(f"Apply PCA {pca.d_in} -> {pca.d_out}")
    #     all_desc = pca.apply_py(all_desc)
    #
    # print("normalizing descriptors")
    # faiss.normalize_L2(all_desc)
    #
    # if not args.train_pca:
    #     print(f"writing descriptors to {args.o}")
    #     write_hdf5_descriptors(all_desc, image_list, args.o)


if __name__ == "__main__":
    main()