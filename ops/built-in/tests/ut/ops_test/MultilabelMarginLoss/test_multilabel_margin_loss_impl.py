# # -*- coding:utf-8 -*-
import sys
import numpy as np
import torch
from op_test_frame.ut import BroadcastOpUT
from torch.autograd import Variable

ut_case = BroadcastOpUT("multilabel_margin_loss")

input_x = np.array([[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]]).astype("float32")
target = np.array([[1, 1, 1, 1], [1, 1, 1, 1]]).astype("int32")
is_target = np.zeros((2, 4)).astype("int32")

for i in range(0, target.shape[0]):
    for j in range(0, target.shape[1]):
        if target[i][j] == -1:
            break
        is_target[i][target[i][j]] = is_target[i][target[i][j]] + 1

torch_input = Variable(torch.from_numpy(input_x), requires_grad=True)
torch_target = torch.from_numpy(target).to(torch.int32)

loss = torch.nn.functional.multilabel_margin_loss(torch_input, torch_target.to(torch.int64), reduction="mean")
loss1 = np.array([loss.detach()])


#pylint: disable=unused-argument
def calc_expect_func(x, target1, y, is_target_input):
    return [loss1, is_target]

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 4), "shape": (2, 4),
                "param_type": "input", "value": input_x},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 4), "shape": (2, 4),
                "param_type": "input", "value": target},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 4), "shape": (2, 4),
                "param_type": "output"}],
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 4), "shape": (2, 4),
                "param_type": "input", "value": input_x},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 4), "shape": (2, 4),
                "param_type": "input", "value": target},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, 4), "shape": (2, 4),
                "param_type": "output"}]
})


ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2, ), "shape": (2, ),
                "param_type": "input", "value": input_x},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, ), "shape": (2, ),
                "param_type": "input", "value": target},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (2, ), "shape": (2, ),
                "param_type": "output"}]
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (200, ), "shape": (200, ),
                "param_type": "input", "value": input_x},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (200, ), "shape": (200, ),
                "param_type": "input", "value": target},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (200, ), "shape": (200, ),
                "param_type": "output"}]
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (200, 1 ), "shape": (200, 1),
                "param_type": "input", "value": input_x},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (200, 1), "shape": (200, 1),
                "param_type": "input", "value": target},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (200, 1), "shape": (200, 1),
                "param_type": "output"}]
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (200, 100), "shape": (200, 100),
                "param_type": "input", "value": input_x},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (200, 100), "shape": (200, 100),
                "param_type": "input", "value": target},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "output"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (200, 100), "shape": (200, 100),
                "param_type": "output"}]
})
