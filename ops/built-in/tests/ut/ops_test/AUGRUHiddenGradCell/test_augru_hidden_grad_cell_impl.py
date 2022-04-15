# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
ut_case = OpUT("augru_hidden_grad_cell", "impl.augru_hidden_grad_cell", "augru_hidden_grad_cell")

def gen_augru_hidden_grad_case(t, n, output_size, dtype="float32"):
    case_name = "case_%s_%s_%s" % (t, n, output_size)
    return {"params": [
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (output_size * 3, output_size, 16, 16), "dtype": "float16"},
        {"shape": (output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype},
        {"shape": (output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, 3 * output_size, n, 16, 16), "dtype": dtype},
        {"shape": (t, output_size, n, 16, 16), "dtype": dtype}],
        "case_name": case_name
    }

ut_case.add_case(['Ascend910A'], gen_augru_hidden_grad_case(5, 1, 4))

if __name__ == '__main__':
    ut_case.run('Ascend910A')
    exit(0)
