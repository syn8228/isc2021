import sys
import os
import pdb
import argparse
import time
from collections import OrderedDict, defaultdict

sys.path.append('/cluster/yinan/isc2021')

from PIL import Image
from isc.io import write_hdf5_descriptors, read_ground_truth, read_predictions, write_predictions, read_descriptors
from utils.argumentation import *

import torch
import torchvision
import torchvision.transforms
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import os
import glob
import faiss
import random
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import h5py
import copy
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT
import timm
from pprint import pprint
import pandas as pd
from torchvision.transforms import Compose

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

def imshow(img, text=None, should_save=False, pth=None):
    np_img = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    if should_save:
        plt.savefig(pth)
    plt.show()


def gem_npy(x, p=3, eps=1e-6):
    x = np.clip(x, a_min=eps, a_max=np.inf)
    x = x ** p
    x = x.mean(axis=0)
    return x ** (1. / p)


def generate_train_dataset(query_list, gt_list, train_list, len_data):
    # TODO: generate training list with length len_data
    # random.seed(1)
    t_list = list()
    gt_list = gt_list[0: int(len(gt_list)*3/4)]
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
    # random.seed(3)
    v_list = list()
    gt_list = gt_list[int(len(gt_list)*3/4): -1]
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


def generate_extraction_dataset(query_list, db_list, train_list):
    query_images = [QUERY + q + ".jpg" for q in query_list]
    db_images = [REFERENCE + r + ".jpg" for r in db_list]
    train_images = [TRAIN + t + ".jpg" for t in train_list]
    return query_images, db_images, train_images


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
            # nn.Linear(2048 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),

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
                          label * 0.5 * torch.pow(torch.clamp(15.0 - score, min=0.0), 2))
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
        if self.transform is not None:
            x = self.transform(x)
        return x


class ValList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        q, r, label = self.image_list[i]
        query_image = Image.open(q)
        db_image = Image.open(r)
        query_image = query_image.convert("RGB")
        db_image = db_image.convert("RGB")
        if self.transform is not None:
            query_image = self.transform(query_image)
            db_image = self.transform(db_image)
        return query_image, db_image, label


class TrainList(Dataset):

    def __init__(self, image_list, imsize=None, transform=None, argumentation=None):
        Dataset.__init__(self)
        self.image_list = image_list
        self.transform = transform
        self.imsize = imsize
        self.argumentation = argumentation

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        label = random.randint(0, 1)
        background = Image.open(random.sample(self.image_list, 1)[0])
        self.argumentation.append(MergeImage(background, probability=0.2))
        random.shuffle(self.argumentation)
        argument = Compose(self.argumentation)
        if label == 0:
            db_image = Image.open(self.image_list[i])
            db_image = db_image.convert("RGB")
            query_image = argument(db_image)
            if self.transform is not None:
                query_image = self.transform(query_image)
                db_image = self.transform(db_image)
        else:
            db_image = Image.open(self.image_list[i])
            query_image = Image.open(random.sample(self.image_list, 1)[0])
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
    aa('--track1', default=False, action="store_true", help="run feature extraction for track1")
    aa('--device', default="cuda:0", help='pytroch device')
    aa('--batch_size', default=32, type=int, help="max batch size to use for extraction")
    aa('--num_workers', default=8, type=int, help="nb of dataloader workers")

    group = parser.add_argument_group('model options')
    aa('--model', default='multigrain_resnet50', help="model to use")
    aa('--checkpoint', default='Siamese_Epoch_4.pth', help='best saved model name')
    aa('--GeM_p', default=7.0, type=float, help="Power used for GeM pooling")
    aa('--scales', default="1.0", help="scale levels")
    aa('--imsize', default=512, type=int, help="max image size at extraction time")
    aa('--lr', default=0.0001, type=float, help="learning rate")
    aa('--weight_decay', default=0.0005, type=float, help="max image size at extraction time")

    group = parser.add_argument_group('dataset options')
    aa('--query_list', required=True, help="file with  query image filenames")
    aa('--gt_list', required=True, help="file with ground truth image filenames")
    aa('--train_list', required=True, help="file with training image filenames")
    aa('--db_list', required=True, help="file with training image filenames")
    aa('--len', default=1000, type=int, help="nb of training vectors for the SiameseNetwork")
    aa('--epoch', default=100, type=int, help="nb of training epochs for the SiameseNetwork")
    aa('--i0', default=0, type=int, help="first image to process")
    aa('--i1', default=-1, type=int, help="last image to process + 1")

    group = parser.add_argument_group('output options')
    aa('--query_f', default="isc2021/data/query_siamese.hdf5", help="write query features to this file")
    aa('--db_f', default="isc2021/data/db_siamese.hdf5", help="write query features to this file")
    aa('--train_f', default="isc2021/data/train_siamese.hdf5", help="write training features to this file")
    aa('--net', default="isc2021/checkpoints/Siamese/", help="save network parameters to this folder")
    aa('--images', default="isc2021/data/images/siamese/", help="save visualized test result to this folder")

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

    gt_list = read_ground_truth(args.gt_list)
    query_list = [l.strip() for l in open(args.query_list, "r")]
    db_list = [l.strip() for l in open(args.db_list, "r")]
    train_list = [l.strip() for l in open(args.train_list, "r")]

    query_images, db_images, train_images = generate_extraction_dataset(query_list, db_list, train_list)

    if args.i1 != -1 or args.i0 != 0:
        db_list = db_images[args.i0:args.i1]
        train_list = train_images[args.i0:args.i1]
    else:
        db_list = db_images
        train_list = train_images


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
            torchvision.transforms.Resize((args.imsize, args.imsize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
        ]

    if args.transpose != -1:
        transforms.insert(TransposeTransform(args.transpose), 0)

    transforms = torchvision.transforms.Compose(transforms)

    if args.train:
        argu_list = [
            VerticalFlip(probability=0.8),
            HorizontalFlip(probability=0.8),
            Rotate(probability=0.8),
            GaussianBlur(probability=0.5),
            ColRec(probability=0.5),
            GaussianNoise(probability=0.5),
            ZoomIn(probability=0.5),
            ZoomOut(probability=0.5),
            RandomCut(0.2),
            NegativeImage(0.1),
        ]


        print("training network")
        val_list = train_images[0:args.len]
        val_pairs = TrainList(val_list, transform=transforms, imsize=args.imsize, argumentation=argu_list)
        val_dataloader = DataLoader(dataset=val_pairs, shuffle=True, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
        print("loading model")
        net = SiameseNetwork(args.model)
        net.to(args.device)
        criterion = ContrastiveLoss()
        criterion.to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr=args.lr, weight_decay=args.weight_decay)
        loss_history = list()
        epoch_losses = list()
        epoch_size = int(len(train_list)/args.epoch)
        for epoch in range(args.epoch):
            train_subset = train_list[epoch * epoch_size: (epoch+1)*epoch_size - 1]
            im_pairs = TrainList(train_subset, transform=transforms, imsize=args.imsize, argumentation=argu_list)
            train_dataloader = DataLoader(dataset=im_pairs, shuffle=True, num_workers=args.num_workers,
                                          batch_size=args.batch_size)
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

                if (i+1)*args.batch_size % 500 == 0:
                    mean_loss = torch.mean(torch.Tensor(loss_history))
                    loss_history.clear()

                    print("Epoch:{},  Current training loss {}\n".format(epoch, mean_loss))
            val_loss = []
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
                    val_loss.append(criterion(output, label))
                val_loss = torch.mean(torch.Tensor(val_loss))
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
        print("best model is: {} with validation loss {}\n".format(best_model_name, epoch_losses[best_epoch]))

    else:
        print("computing features")
        query_dataset = ImageList(query_images, transform=transforms)
        db_dataset = ImageList(db_images, transform=transforms)
        train_dataset = ImageList(train_images, transform=transforms)

        net = SiameseNetwork(args.model)
        state_dict = torch.load(args.net + args.checkpoint)
        net.load_state_dict(state_dict)
        net.to(args.device)
        print("checkpoint {} loaded\n".format(args.checkpoint))
        print("test model\n")
        test_list = generate_validation_dataset(query_list, gt_list, train_list, 20)
        test_data = ValList(test_list, transform=transforms, imsize=args.imsize)
        test_loader = DataLoader(dataset=test_data, shuffle=True, num_workers=args.num_workers,
                                 batch_size=1)
        with torch.no_grad():
            distance_p = []
            distance_n = []
            for i, data in enumerate(test_loader, 0):
                img_name = 'test_{}.jpg'.format(i)
                img_pth = args.images + img_name
                q_img, r_img, label = data
                concatenated = torch.cat((q_img, r_img), 0)
                q_img_cp = copy.deepcopy(q_img)
                r_img_cp = copy.deepcopy(r_img)
                q_img = q_img_cp.to(args.device)
                r_img = r_img_cp.to(args.device)
                score = net(q_img, r_img).cpu()
                if label == 0:
                    label = 'matched'
                    distance_p.append(score.item())
                    print('matched with distance: {:.4f}\n'.format(score.item()))
                if label == 1:
                    label = 'not matched'
                    distance_n.append(score.item())
                    print('not matched with distance: {:.4f}\n'.format(score.item()))
                imshow(torchvision.utils.make_grid(concatenated),
                       'Dissimilarity: {:.2f} Label: {}'.format(score.item(), label), should_save=True, pth=img_pth)
        mean_distance_p = torch.mean(torch.Tensor(distance_p))
        mean_distance_n = torch.mean(torch.Tensor(distance_n))
        print('not matched mean distance: {:.4f}\n'.format(mean_distance_n))
        print('matched mean distance: {:.4f}\n'.format(mean_distance_p))

        if args.batch_size == 1:
            with torch.no_grad():
                query_feat, db_feat = list(), list()
                t0 = time.time()
                for i, x in enumerate(query_dataset):
                    x_cp = copy.deepcopy(x)
                    x = x_cp.to(args.device)
                    x = x.unsqueeze(0)
                    o = net.forward_once(x)
                    o = torch.squeeze(o)
                    query_feat.append(o.cpu())
                t1 = time.time()
                query_feat = np.vstack(query_feat)
                write_hdf5_descriptors(query_feat, query_images, args.query_f)
                print(f"writing query descriptors to {args.query_f}")
                print(f"query_image_description_time: {(t1 - t0) / len(query_images):.5f} s per image")

                t0 = time.time()
                for i, x in enumerate(db_dataset):
                    x_cp = copy.deepcopy(x)
                    x = x_cp.to(args.device)
                    x = x.unsqueeze(0)
                    o = net.forward_once(x)
                    o = torch.squeeze(o)
                    db_feat.append(o.cpu())
                t1 = time.time()
                db_feat = np.vstack(db_feat)
                write_hdf5_descriptors(db_feat, db_images, args.db_f)
                print(f"writing reference descriptors to {args.db_f}")
                print(f"db_image_description_time: {(t1 - t0) / len(db_images):.5f} s per image")

                if args.track1:
                    train_feat = list()
                    t0 = time.time()
                    for i, x in enumerate(train_dataset):
                        x_cp = copy.deepcopy(x)
                        x = x_cp.to(args.device)
                        x = x.unsqueeze(0)
                        o = net.forward_once(x)
                        o = torch.squeeze(o)
                        train_feat.append(o.cpu())
                    t1 = time.time()
                    train_feat = np.vstack(train_feat)
                    write_hdf5_descriptors(train_feat, train_images, args.train_f)
                    print(f"writing reference descriptors to {args.train_f}")
                    print(f"train_image_description_time: {(t1 - t0) / len(db_images):.5f} s per image")

        else:
            query_loader = DataLoader(dataset=query_dataset, shuffle=False, num_workers=args.num_workers,
                                          batch_size=args.batch_size)
            db_loader = DataLoader(dataset=db_dataset, shuffle=False, num_workers=args.num_workers,
                                          batch_size=args.batch_size)
            query_feats = list()
            db_feats = list()
            t0 = time.time()
            with torch.no_grad():
                for no, data in enumerate(query_loader):
                    images = data
                    images = images.to(args.device)
                    feats = net.forward_once(images)
                    query_feats.append(feats.cpu().numpy())
                t1 = time.time()
                query_feats = np.vstack(query_feats)
                write_hdf5_descriptors(query_feats, query_images, args.query_f)
                print(f"writing query descriptors to {args.db_f}")
                print(f"db_image_description_time: {(t1 - t0) / len(db_images):.5f} s per image")

                t0 = time.time()
                for no, data in enumerate(db_loader):
                    images = data
                    images = images.to(args.device)
                    feats = net.forward_once(images)
                    db_feats.append(feats.cpu().numpy())
                t1 = time.time()
                db_feats = np.vstack(db_feats)
                write_hdf5_descriptors(db_feats, db_images, args.db_f)
                print(f"writing reference descriptors to {args.db_f}")
                print(f"db_image_description_time: {(t1 - t0) / len(db_images):.5f} s per image")

                if args.track1:
                    train_feats = list()
                    train_loader = DataLoader(dataset=train_dataset, shuffle=False,
                                              num_workers=args.num_workers, batch_size=args.batch_size)
                    t0 = time.time()
                    for no, data in enumerate(train_loader):
                        images = data
                        images = images.to(args.device)
                        feats = net.forward_once(images)
                        train_feats.append(feats.cpu().numpy())
                    t1 = time.time()
                    train_feats = np.vstack(train_feats)
                    write_hdf5_descriptors(train_feats, train_images, args.train_f)
                    print(f"writing reference descriptors to {args.train_f}")
                    print(f"train_image_description_time: {(t1 - t0) / len(db_images):.5f} s per image")


if __name__ == "__main__":
    main()