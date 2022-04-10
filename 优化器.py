import numpy as np

# 随机梯度下降法SGD

class SGD:
    def __init__(self, lr=0.01,epochs = 0,L2_lambda = 0.001):
        self.lr = lr
        self.initiallr = 0.01
        self.m = None
        self.beta1 = 0.9
        self.L2_lambda = L2_lambda
        self.decay = 0.01 / epochs

    # 更新权重
    def update(self, params, grads):
        if self.m is None:
            self.m = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            params[key] -= self.lr * self.m[key] - self.L2_lambda * params[key]

    # 更新学习率
    def updatelr(self,iterations = 1):
         self.lr = self.initiallr / (1 + self.decay * iterations)