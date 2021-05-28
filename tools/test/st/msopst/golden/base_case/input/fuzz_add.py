import random

def fuzz_branch():
    x_shape = [1, 2]
    return {"input_desc": {"x1": {"shape": x_shape},
                           "x2": {"shape": x_shape}},
            "output_desc": {"y": {"shape": x_shape}}}