import numpy as np

# 批数据的平均交叉熵误差
# y 为预测数据, t 为正确数据
def cross_entropy_error(y, t, params, L2_lambda):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_num = y.shape[0]

    # chengfa = 0
    # for key in params.keys():
    #     chengfa += 1/2 * L2_lambda * np.sum(params[key].reshape(-1)**2)

    return -np.sum(np.log(y[np.arange(batch_num), t] + 1e-7)) / batch_num