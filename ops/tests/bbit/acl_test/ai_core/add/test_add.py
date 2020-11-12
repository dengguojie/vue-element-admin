import os
import types
from op_test_frame.st.op_st import st_run

st_run(os.path.dirname(__file__))

# st_run("")
# import sys
# np_op_file = os.path.realpath("./np_add.py")
# np_dir = os.path.dirname(np_op_file)
# sys.path.append(np_dir)
# np_module_name = os.path.basename(np_op_file)[:-3]
# __import__(np_module_name)
# np_module = sys.modules[np_module_name]
# print(np_module.__dict__.keys())
# print(np_module.__dict__["np_add"])
# print(isinstance(np_module.__dict__["np_add"], types.FunctionType))
# print(np_module.__dict__["np"])

# print(os.path.sep)
# aa = zip({"a": 1, "b":2}, {"c":3, "d":4})
# for key, val in aa:
#     print(key, val)
# import re
# print(re.sub(r'[^0-9a-z]', "", "dsada-1231das"))

# a = [[1, 2], [3, 4], [5, 6]]
# b = ["float16", "float32"]
# c = [[0, 1], [1, 2]]
#
# import itertools
#
# for x in itertools.product(c, b, a):
#     print(x[2], x[1], x[0])

# import re
# case_reg = re.compile('# case1: (.|\n)*#case1: end')
# all_case_str = case_reg.sub("abc", """dsada # case1: dasda
# sdasda #case1: end asdsads3213asda""")
# print(all_case_str)