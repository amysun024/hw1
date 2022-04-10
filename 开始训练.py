from 神经网络 import *
from 训练器 import *
import matplotlib.pylab as plt

## 数据预处理（one-hot转换、正规化处理）

save_file = "./mnist.pkl"

# 将标签数据转换成 one-hot 格式
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

# 加载并按照给定格式返回 训练数据 和 测试数据
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    # 从 mnist.pkl 文件中加载数据
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    # 输入数据的正规化处理
    if normalize:
        # 仅遍历处理输入数据
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    # 标签数据的 one-hot 处理
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    # 如果不返回一维数据, 就要将一维序列数据转化为多维数据
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    # 返回训练数据和测试数据
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 设置遍历数据集的次数
max_epochs = 20

# 生成神经网络的对象
network = SimpleConvNet(input_dim = (1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'filter_stride':1, 'filter_pad':0},
                       output_size = 10, weight_init_std = 0.01)

# 生成训练器的对象
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr':0.01,'epochs': max_epochs},
                  evaluate_sample_num_per_epoch=10000)
if __name__ == '__main__':
    # 开始训练
    trainer.train()

    # 保存训练出来的参数结果
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    # 展示每一 epoch 的结果
    x = np.arange(max_epochs)
    plt.plot(x,trainer.train_loss_lists)
    plt.xlabel("epochs")
    plt.ylabel("train loss")
    plt.show()
    plt.plot(x,trainer.test_loss_list)
    plt.xlabel("epochs")
    plt.ylabel("test loss")
    plt.show()
    plt.plot(x,trainer.test_acc_list)
    plt.xlabel("epochs")
    plt.ylabel("test acc")
    plt.show()
