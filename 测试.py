from 开始训练 import load_mnist
from 神经网络 import *
# ReLU层
class Relu:
    def __init__(self):
        self.is_negative = None

    def forward(self, input):
        self.is_negative = input <= 0
        output = input.copy()
        output[self.is_negative] = 0
        return output

# softmax 层和 cross entropy error 层
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, input, t, params, L2_lambda):
        self.y = softmax(input)
        self.t = t
        self.loss = cross_entropy_error(self.y,self.t,params,L2_lambda)
        return self.loss

# 仿射层 Affine 层, 即计算 Y = X * W + b 的那一层
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        output = np.dot(self.x, self.W) + self.b
        return output

# 卷积层
class Convolution:
    # W 是四维数据, b 是一维数据
    def __init__(self, W, b, stride = 1, pad = 0):
        # 卷积层的参数
        self.W = W              # 权重参数
        self.b = b              # 偏置参数
        self.stride = stride    # 步幅
        self.pad = pad          # 填充

        # 反向传播时计算梯度用到的数据
        self.x = None           # 该层的输入数据
        self.col_x = None       # 输入数据的二维格式
        self.col_W = None       # 权重参数的二维格式

        # 记录梯度
        self.dW = None          # 损失函数 L 对权重 W 的梯度
        self.db = None          # 损失函数 L 对偏置 b 的梯度

    # x 是四维数据
    def forward(self, x):
        # 记录输入数据和权重的 shape
        N, C, H, W = x.shape
        FN, C, FH, FW = self.W.shape

        # 计算输出数据的高和宽
        OH = int(1 + (H + 2 * self.pad - FH) / self.stride)
        OW = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # x 和 W 转化为二维数据, 方便进行矩阵乘法运算
        col_x = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T    # -1 表示自动计算, T 表示竖向展开

        # 计算结果
        output = np.dot(col_x, col_W) + self.b

        # 重塑结果的 shape
        output = output.reshape(N, OH, OW, -1).transpose(0,3,1,2)

        # 记录数据
        self.x = x
        self.col_x = col_x
        self.col_W = col_W

        return output


# 池化层
class Pooling:
    # pool_h 和 pool_w 表示池化层的目标区域高和宽
    def __init__(self, pool_h, pool_w, stride=1, pad = 0):
        # 记录参数
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape

        OH = int(1 + (H + 2 * self.pad - self.pool_h) / self.stride)
        OW = int(1 + (W + 2 * self.pad - self.pool_w) / self.stride)

        col_x = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col_x = col_x.reshape(-1, self.pool_h * self.pool_w)

        # Max 池化
        output = np.max(col_x, axis = 1)

        output = output.reshape(N, OH, OW, C).transpose(0,3,1,2)

        self.x = x
        self.arg_max = np.argmax(col_x, axis=1)

        return output

import pickle
from 各层单独实现 import *
from collections import OrderedDict # 有序字典

class Net:

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

        self.load_params("params.pkl")
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

        self.layers['Conv1'].W = self.params['W' + str(1)]
        self.layers['Conv1'].b = self.params['b' + str(1)]
        self.layers['Affine'].W = self.params['W' + str(2)]
        self.layers['Affine'].b = self.params['b' + str(2)]

if __name__ == '__main__':
    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    # 生成神经网络的对象
    network = Net(input_dim=(1, 28, 28),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'filter_stride': 1, 'filter_pad': 0},
                            output_size=10, weight_init_std=0.01)
    # 生成训练器的对象
    test_acc = network.accuracy(x_test, t_test)
    print("=============== Final Test Accuracy ===============")
    print("test acc:" + str(test_acc))