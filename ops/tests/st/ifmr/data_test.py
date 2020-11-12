
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from functools import reduce
import sys
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
# from amct_mindspore.cells import TestQuantIfmrTik

import numpy as np
import mindspore.context as context
from mindspore import Tensor
from functools import reduce

context.set_context(device_target="Ascend")

class TestQuantIfmrTik(Cell):
    def __init__(self,
                 min_percentile=0.999999,
                 max_percentile=0.999999,
                 search_range=[0.7, 1.3],
                 search_step=0.01,
                 offset_flag=True):
        super(TestQuantIfmrTik, self).__init__()
        self.ifmr = P.IFMR(min_percentile, max_percentile, search_range, search_step, offset_flag)

    def construct(self, x, y, z, w):
        s, o = self.ifmr(x, y, z, w)
        return s, o

context.set_context(device_target="Ascend")

EPSILON = np.finfo(np.float64).eps
RESOLUTION = np.finfo(np.float64).resolution


def fake_quant(float_data, scale, offset):
    output_data = np.round(float_data / scale) + offset
    np.clip(output_data, -128, 127, out=output_data)
    output_data = (output_data - offset) * scale
    return output_data


def calculate_cosine_similarity(a, b):
    a = a.astype(np.float64) + RESOLUTION
    b = b.astype(np.float64) + RESOLUTION
    if a.shape != b.shape:
        raise RuntimeError('a shape {} != b shape {}'.format(a.shape, b.shape))
    if a.max() == a.min() and b.max() == b.min():
        return 1
    numerator = np.sum(a * b)
    denominator = ((np.sum(a * a)) ** 0.5) * ((np.sum(b * b)) ** 0.5)
    return numerator / denominator

def cal_scale_offset(data, bins_num, min_percentile, max_percentile, search_range, search_step, with_offset):
    data_shape=data.shape
    data_type=data.dtype
    # 数据预处理
    data_max = np.max(data)
    data_min = np.min(data)
    if data_min>0:
        data_min = 0
    if data_max<0:
        data_max =0
    data_num = reduce(lambda x, y: x * y, data_shape)
    data_max = np.array([data_max], dtype=data_type)
    data_min = np.array([data_min], dtype=data_type)
    # 计算累加和
    bins, threshold = np.histogram(data, bins_num)
    cumsum = np.cumsum(bins).astype(np.int32)
    
    # 生成 scale&offset
    cdf = cumsum / data_num
    max_index = np.where(cdf > max_percentile, 0, 1).sum()
    min_index = np.where(cdf > 1 - min_percentile, 0, 1).sum()
    max_init = max_index / bins_num * (data_max - data_min) + data_min
    min_init = min_index / bins_num * (data_max - data_min) + data_min
    step = np.arange(search_range[0], search_range[1], search_step)
    max_list = max_init * step
    min_list = min_init * np.ones(step.shape)
    scale = (max_list - min_list) / 255
    print(scale)

    offset = np.round(min_list / scale)
    offset = -(offset + 128)
    print(offset)
    data_list = data.flatten()

    loss_list = np.zeros(len(step))

    for i in range(len(step)):
        quant_data_list = np.round(data_list / scale[i]) + offset[i]
        np.clip(quant_data_list, -128, 127, out=quant_data_list)
        quant_data_list = (quant_data_list - offset[i]) * scale[i]
        loss = np.sum(np.square(quant_data_list - data_list))
        loss_list[i] = loss
    print(loss_list)

    index = np.unravel_index(np.argmin(loss_list), loss_list.shape)
    print(scale[index])
    print(offset[index])
    fm = Tensor(data)
    fm_min = Tensor(data_min)
    fm_max = Tensor(data_max)
    cusum_bins = Tensor(cumsum)
    net = TestQuantIfmrTik(min_percentile, max_percentile, search_range, search_step, with_offset)
    tik_scale, tik_offset = net(fm, fm_min, fm_max, cusum_bins)
    print(tik_scale.asnumpy()[0])
    print(tik_offset.asnumpy()[0])
    if abs((tik_scale.asnumpy()[0] - scale[index])/scale[index])>0.0001:
        print("scale diff is larger than 0.0001")
        raise ValueError("!!!!!!!!!!!!!!!")
    #if abs((tik_offset.asnumpy()[0] - offset[index])/offset[index])>0.0001:
    if abs((tik_offset.asnumpy()[0] - loss_list[index])/loss_list[index])>0.0001:
    #if tik_offset.asnumpy()[0]!=30.0:
    #if int(tik_offset.asnumpy()[0]) % 30:
        print("offset diff is larger than 0.0001")
        raise ValueError("!!!!!!!!!!!!!!!")
    return tik_scale.asnumpy()[0], tik_offset.asnumpy()[0]

if __name__ == '__main__':
    for j in range(100000000):	
        for i in range(len(sys.argv)):
            if i == 0:
                continue
            print("data: " + sys.argv[i])
            data = np.loadtxt(sys.argv[i], dtype='float32')
            #data = np.random.random((24,2)).astype(np.float32)
            #data = np.zeros((24,2), np.float32)
            scale, offset = cal_scale_offset(data, 512, 
                     0.999999,
                     0.999999,
                     [0.7, 1.3],
                     0.01,
                     True)
            print(scale, offset)
            fake_data = fake_quant(data, scale, offset)
            sim = calculate_cosine_similarity(fake_data, data)
            print("cosine sim: %f" %sim)





