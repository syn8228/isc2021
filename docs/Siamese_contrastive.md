# SiameseNetwork trained by Contrastive Loss

Training Example:
python3	isc2021/baselines/Siamese.py 
--train
--query_list isc2021/list_files/subset_1_queries
--gt_list isc2021/list_files/subset_1_ground_truth.csv 
--train_list isc2021/list_files/train 
--db_list isc2021/list_files/subset_1_references 
--epoch 2 
--model visformer 
--batch_size 5 
--len 2000 
--lr 0.00003 
--weight_decay 0.00005 
--i0 0 
--i1 400

Options:
--transpose: insert one of the 7 PIL transpose, default:-1
--train: train network or use network for feature extraction
--device: pytorch device, default:¡±cuda:0¡±
--batch_size: Dataloader batch size
--num_workers: number of Dataloader workers

Model Options:
--model: select pretrained model
--lr: learning rate
--weight_decay: l2 regularization weight

Model list: 
zoo_resnet50: 	use resnet50 from pytorch
multigrain_resnet50: 	use resnet50 from facebook
vgg
resnet152
efficientnetb1
efficientnetb7
visformer: 	visual transformer vit_large_patch16_384

Dataset Options:
--query_list: path to file with name of query images 
--db_list: path to file with name of reference images
--gt_list: path to file with list of ground truth

--train_list: path to file with name of images in training dataset
--len: lenth of validation set
--epoch: epoch of training, each epoch use different sequence of argumentations
--i0and --i1: choose part of training list i0 is start index, i1 is end index

Output Options:
--net: save checkpoint to this folder  















