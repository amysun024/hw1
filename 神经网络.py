import pickle
from 各层单独实现 import *
from collections import OrderedDict # 有序字典

### 简单卷积神经网路
# 将各层拼接起来实现
class SimpleConvNet:

    """简单的两层卷积神经网络
    网络结构: conv - relu - pool -
            affine
    """
    def __init__(self,
                 input_dim = (1, 28, 28),
                 conv_param = {'filter_num':30, 'filter_size':2, 'filter_stride':2, 'filter_pad':0},
                output_size = 10, weight_init_std = 0.01, L2_lambda = 0.001):
        # 滤波器的参数, 包括了 FN, FH, FW, stride, pas
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']     # FH, FW 都等于 filter_size
        filter_stride = conv_param['filter_stride']
        filter_pad = conv_param['filter_pad']

        # 各层输入输出的大小
        input_size = input_dim[1]   # 输入的 H 和 W
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1      # 计算卷积层输出的 OH 和 OW
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))    # 计算池化层输出的总大小
        # 本来池化层输出的 size 应该是 OH 和 OW 的一半, 因为池化的目标区域大小为 2 * 2, 步幅为 2
        # 这里计算的是总的大小, 所以应该是滤波器的数量 * 卷积层输出的高的一半 * 卷积层输出的宽的一半

        # 初始化权重
        self.params = {}
        self.L2_lambda = L2_lambda

        # 第一层: 卷积层参数
        # 滤波器 :FN, C, FH, FW
        # 偏置: FN
        self.params['W1'] = np.random.randn(filter_num, input_dim[0], filter_size, filter_size) * weight_init_std
        self.params['b1'] = np.zeros(filter_num)

        # 第二层: affine 层参数, 同时也是输出层, 后接 softmax 的输出层激活函数
        self.params['W2'] = np.random.randn(pool_output_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

        # 创建各层类的对象, 由于层之间是有次序的, 所以需要用有序字典来存储各个层对象
        self.layers = OrderedDict()
        # 卷积层
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['filter_stride'], conv_param['filter_pad'])
        # 卷积层的激活函数
        self.layers['Relu1'] = Relu()
        # 卷积层的池化层
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 仿射层 Affine
        self.layers['Affine'] = Affine(self.params['W2'], self.params['b2'])
        # 输出层 激活函数 softmax + 损失函数层
        self.last_layer = SoftmaxWithLoss()

    # 正向传播, 仅到 Affine 层
    def predict(self, x):
        # 没有遍历最后一层的 softmax + 损失函数
        # 最终的结果是最后一个 Affine 的输出结果, 不是 softmax 层输出的概率
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # 正向传播, 到最后的 loss 层
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t, self.params, self.L2_lambda)

    # 误差反向传播法求各层权重 W 和 b 的梯度,x 是输入数据, t 是标签数据
    def gradient(self, x, t):
        self.loss(x, t)     # 先正向传播一遍, 使各层记录下反向传播需要用到的中间数据

        # 反向传播求各层梯度
        douput = 1
        # 由于最后的 softmax + loss 层是单独存储的, 所以需要单独传播
        douput = self.last_layer.backward(douput)

        # 先把各层对象拿出来逆序, 因为是反向传播
        layers = list(self.layers.values())
        layers.reverse()

        # 然后开始反向传播
        for layer in layers:
            douput = layer.backward(douput)

        # 组装返回的结果, 即梯度的字典变量
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine'].dW, self.layers['Affine'].db

        return grads

    # 计算精确度
    def accuracy(self, x, t, batch_size=100):
        # t 为 one-hot 中标签数据的下标, 即正确结果的下标
        if t.ndim != 1: t = np.argmax(t, axis=1)

        # 精确度初始化为 0
        acc = 0.0

        # 每批数据有 100 , 数据按批计算精确度
        for i in range(int(x.shape[0] / batch_size)):  # 第 i 批数据
            tx = x[i * batch_size:(i + 1) * batch_size]  # 第 i 批的输入数据
            tt = t[i * batch_size:(i + 1) * batch_size]  # 第 i 批的标签数据
            y = self.predict(tx)  # 第 i 批的预测数据
            y = np.argmax(y, axis=1)  # y 表示为预测结果的下标
            acc += np.sum(y == tt)  # 统计正确预测的个数

        return acc / x.shape[0]  # 正确预测的个数 除以 总的数据个数

    # 从文件中加载参数
    def load_params(self, file_name = "params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate('Conv1', 'Affine'):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

    # 将参数保存到文件中
    def save_params(self, file_name = "params.pkl"):
        # 组织保存的参数
        params = {}
        for key, val in self.params.items():
            params[key] = val
        # 向文件写入参数
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)