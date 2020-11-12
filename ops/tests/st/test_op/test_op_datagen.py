"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

"""

import numpy as np
import sys
from dataFormat import *


def test_op(name, shape, src_type):
    sys.stdout.write("Info: writing input for %s...\n" % name)
    """
    TODO:
    write codes for generating data.
    """
    sys.stdout.write("Info: writing output for %s done!!!\n" % name)


def gen_test_op_data(isBBIT=False):
    pass


if __name__ == "__main__":
    gen_test_op_data()
