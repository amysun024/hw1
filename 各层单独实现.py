from 激活函数 import *
from 损失函数 import *
from 卷积的数学实现 import *

# ReLU层
class Relu:
    def __init__(self):
        self.is_negative = None

    def forward(self, input):
        self.is_negative = input <= 0
        output = input.copy()
        output[self.is_negative] = 0
        return output

    def backward(self, doutput):
        doutput[self.is_negative] = 0
        dinput = doutput
        return dinput

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

    def backward(self, doutput = 1):
        batch_num = self.y.shape[0]
        if self.t.size == self.y.size:
            dinput = (self.y - self.t) / batch_num
        else:
            dinput = self.y.copy()
            dinput[np.arange(batch_num), self.t] -= 1
            dinput = dinput / batch_num
        return dinput

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

    def backward(self, doutput):
        dintput = np.dot(doutput, self.W.T)
        self.dW = np.dot(self.x.T, doutput)
        self.db = np.sum(doutput, axis = 0)
        dintput = dintput.reshape(*self.original_x_shape)
        return dintput

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

    def backward(self, doutput):
        # 取出 W 的 shape, 重塑 doutput 的shape
        FN, C, FH, FW = self.W.shape
        doutput = doutput.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 根据公式计算并记录相应的梯度
        self.db = np.sum(doutput, axis = 0)
        self.dW = np.dot(self.col_x.T, doutput)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 计算 dx
        dcol_x = np.dot(doutput, self.col_W.T)
        dx = col2im(dcol_x, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


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

    def backward(self, doutput):
        doutput = doutput.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((doutput.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = doutput.flatten()
        dmax = dmax.reshape(doutput.shape + (pool_size,))

        dcol_x = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

        dx = col2im(dcol_x, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


