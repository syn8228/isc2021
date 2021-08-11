from isc.io import read_ground_truth, read_predictions, write_predictions, read_descriptors
import random
db_dir = '/clusrer/shared_datas/isc2021/references/'
query_dir = '/clusrer/shared_datas/isc2021/query/'
train_dir = '/clusrer/shared_datas/isc2021/train/'
len = 1000

gt_list = read_ground_truth('D:\FAU\MME5\References\isc2021-master\list_files\subset_1_ground_truth.csv')
query_list = [l.strip() for l in open('D:\FAU\MME5\References\isc2021-master\list_files\subset_1_queries', "r")]
# db_list = [l.strip() for l in open('D:\FAU\MME5\References\isc2021-master\list_files\subset_1_references', "r")]
train_list = [l.strip() for l in open('D:\FAU\MME5\References\isc2021-master\list_files\\train', "r")]


def generate_train_list(query_list, gt_list, train_list, len_data):
    train_data = list()
    for i in range(len_data):
        label = random.randint(0, 1)
        if label == 1:
            gt = random.sample(gt_list, 1)[0]
            q = gt.query
            r = gt.db
            q = query_dir + q + ".jpg"
            r = db_dir + r + ".jpg"
            train_data.append((q, r, label))
        else:
            q = random.sample(query_list, 1)[0]
            r = random.sample(train_list, 1)[0]
            q = query_dir + q + ".jpg"
            t = train_dir + r + ".jpg"
            train_data.append((q, t, label))
    return train_data

list = generate_train_list(query_list, gt_list, train_list, 10)
print(list)





