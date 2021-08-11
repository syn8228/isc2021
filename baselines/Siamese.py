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
from torch.utils.data import Dataset, DataLoader

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
        # model.eval()
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
        # model.eval()
        return model

    if name == "vgg":
        print('--------------------------------------------------------------')
        print('used model: VGG16')
        print('--------------------------------------------------------------')
        model = torchvision.models.vgg16(pretrained=True)
        # model.eval()
        return model

    if name == "resnet152":
        print('--------------------------------------------------------------')
        print('used model: ResNet152')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet152(pretrained=True)
        # model.eval()
        return model

    if name == "efficientnetb1":
        print('--------------------------------------------------------------')
        print('used model: EfficientNet-b1')
        print('--------------------------------------------------------------')
        model = EfficientNet.from_pretrained('efficientnet-b1')
        # model.eval()
        return model

    if name == "efficientnetb7":
        print('--------------------------------------------------------------')
        print('used model: EfficientNet-b7')
        print('--------------------------------------------------------------')
        model = EfficientNet.from_pretrained('efficientnet-b7')
        # model.eval()
        return model

    if name == "transformer":
        print('--------------------------------------------------------------')
        print('used model: ViT')
        print('--------------------------------------------------------------')
        model = ViT('B_16_imagenet1k', pretrained=True)
        # model.eval()
        return model

    if name == "visformer":
        print('--------------------------------------------------------------')
        print('vit_large_patch16_384')
        print('--------------------------------------------------------------')
        model = timm.create_model('vit_large_patch16_384', pretrained=True)
        # model.eval()
        return model

    assert False

def resnet_activation_map(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


def gem_npy(x, p=3, eps=1e-6):
    x = np.clip(x, a_min=eps, a_max=np.inf)
    x = x ** p
    x = x.mean(axis=0)
    return x ** (1. / p)


def generate_train_dataset(query_list, gt_list, train_list, len_data):
    # TODO: generate training list with length len_data
    random.seed(1)
    t_list = list()
    for i in range(len_data):
        label = random.randint(0, 1)
        if label == 1:
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


class SiameseNetwork(nn.Module):
    def __int__(self):
        super(SiameseNetwork, self).__init__()
        # hard shared head parameters
        self.head = timm.create_model('vit_large_patch16_384', pretrained=True)
        self.head.eval()

        d_h = 1000

        self.fc1 = nn.Sequential(
            nn.Linear(d_h, 512),
            nn.Linear(512, 256)
        )

        self.score = nn.PairwiseDistance(p=2)


    def forward_head(self, input_img):
        if self.map:
            output = resnet_activation_map(self.head, input_img)
        else:
            output = self.head(input_img)
        output = self.fc1(output)
        return output

    def forward(self, query, reference):
        q_out = self.head(query)
        q_out = self.fc1(q_out)
        r_out = self.head(reference)
        r_out = self.fc1(r_out)
        score = self.score(q_out, r_out)
        return score


class SiameseNetwork2(nn.Module):
    def __init__(self):
        super(SiameseNetwork2, self).__init__()
        # self.cnn1 = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(1, 4, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(4),
        #     nn.Dropout2d(p=.2),
        #
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(4, 8, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(8),
        #     nn.Dropout2d(p=.2),
        #
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(8, 8, kernel_size=3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(8),
        #     nn.Dropout2d(p=.2),
        # )
        self.head = timm.create_model('vit_large_patch16_384', pretrained=True)
        self.head.eval()

        self.fc1 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.Linear(512, 256)
        )

        self.score = nn.PairwiseDistance(p=2)

    def forward_once(self, x):
        output = self.head(x)
        #output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        score  = self.score(output1, output2)
        return score


class ContrastiveLoss(torch.nn.Module):
    def __int__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.m = margin

    def forward(self, score, label):
        loss = torch.mean((1 - label) * 0.5 * torch.pow(score, 2) + label * 0.5 * torch.pow(torch.clamp(self.m - score, min=0.0), 2))
        return loss


class TrainPairs(Dataset):

    def __int__(self, img_pairs, transform=None):
        Dataset.__init__(self)
        self.transform = transform
        self.train_list = img_pairs

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, i):
        q, r, label = self.train_list[i]
        query_image = Image.open(q)
        db_image = Image.open(r)
        query_image = query_image.convert("RGB")
        db_image = db_image.convert("RGB")
        if self.transform is not None:
            query_image = self.transform(query_image)
            db_image = self.transform(db_image)
        return query_image, db_image, label


class ImageList(Dataset):

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
    aa('--checkpoint', default='isc2021/data/multigrain_joint_3B_0.5.pth', help='override default checkpoint')
    aa('--GeM_p', default=7.0, type=float, help="Power used for GeM pooling")
    aa('--scales', default="1.0", help="scale levels")
    aa('--imsize', default=512, type=int, help="max image size at extraction time")

    group = parser.add_argument_group('dataset options')
    aa('--query_list', required=True, help="file with  query image filenames")
    aa('--gt_list', required=True, help="file with reference image filenames")
    aa('--train_list', required=True, help="file with training image filenames")
    aa('--db_list', required=True, help="file with training image filenames")
    aa('--len', default=10000, type=int, help="nb of training vectors for the SiameseNetwork")
    aa('--epoch', default=100, type=int, help="nb of training epochs for the SiameseNetwork")
    aa('--i0', default=0, type=int, help="first image to process")
    aa('--i1', default=-1, type=int, help="last image to process + 1")

    group = parser.add_argument_group('output options')
    aa('--query_f', default="/isc2021/data/query_siamese.hdf5", help="write query features to this file")
    aa('--db_f', default="/isc2021/data/db_siamese.hdf5", help="write query features to this file")
    aa('--net', default="/isc2021/data/siamese.pth", help="save network parameters to this file")

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
        t_list = generate_train_dataset(query_list, gt_list, train_list, args.len)
        print(f"subsampled {args.len} vectors")

        # im_pairs = TrainPairs(t_list, transform=transforms, imsize=args.imsize)
        im_pairs = ImageList(t_list, transform=transforms, imsize=args.imsize)
        train_dataloader = DataLoader(dataset=im_pairs, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)
        print("loading model")
        net = SiameseNetwork2()
        net.to(args.device)
        print(net)
        criterion = ContrastiveLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0)

        for epoch in range(args.epoch):
            for i, data in enumerate(train_dataloader, 0):
                q_img, r_img, label = data
                q_img.to(args.device)
                r_img.to(args.device)
                label.to(args.device)
                output = net(q_img, r_img)
                optimizer.zero_grad()
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                if i % 1000 == 0:
                    print("Epoch:{},  Current loss {}\n".format(epoch, loss))

        torch.save(net.state_dict(), args.net)






    # print("computing features")
    #
    # t0 = time.time()
    #
    # with torch.no_grad():
    #     if args.batch_size == 1:
    #         all_desc = []
    #         for no, x in enumerate(im_dataset):
    #             x_cp = copy.deepcopy(x)
    #             x = x_cp.to(args.device)
    #             print(f"im {no}/{len(im_dataset)}    ", end="\r", flush=True)
    #             x = x.unsqueeze(0)
    #             feats = []
    #             for s in args.scales:
    #                 xs = nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
    #                 if args.model == "multigrain_resnet50" or args.model == "zoo_resnet50":
    #                     o = resnet_activation_map(net, xs)
    #                 else:
    #                     o = net(xs)
    #                 o = o.cpu().numpy()  # B, C, H, W
    #                 o = o[0].reshape(o.shape[1], -1).T
    #                 feats.append(o)
    #
    #             feats = np.vstack(feats)
    #             gem = gem_npy(feats, p=args.GeM_p)
    #             all_desc.append(gem)
    #
    #         max_batch_size = args.batch_size
    #
    #         dataloader = torch.utils.data.DataLoader(
    #             im_dataset, batch_size=1, shuffle=False,
    #             num_workers=args.num_workers
    #         )
    #
    #         for no, x in enumerate(dataloader):
    #             x = copy.deepcopy(x[0])  # don't batch
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