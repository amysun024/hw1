import gzip
import pickle
import numpy as np

# mnist 数据集的四个文件名称
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

# 将标签数据从 mnist 数据集文件中以 np 数组格式读取出来
def _load_label(file_name):
    file_path =  "./" + file_name

    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels

# 将输入数据从 mnist 数据集文件中以 np 数组格式读取出来
def _load_img(file_name):
    file_path =  "./" + file_name

    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, img_size)

    return data

# 将下载的四个 mnist 文件以 np 数组的格式读取出来, 并返回
def _convert_numpy():
    dataset = {}

    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset

# 保存 minst 数据集
save_file = "./mnist.pkl"

def init_mnist():
    dataset = _convert_numpy()

    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

if __name__ == '__main__':
    init_mnist()
