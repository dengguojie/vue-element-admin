"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

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
import shutil

def leaky_relu(name, shape, src_type):
    sys.stdout.write("Info: writing input for %s...\n"%name)
    """
    TODO:
    write codes for generating data.
    """
    sys.stdout.write("Info: writing output for %s done!!!\n"%name)

def gen_leaky_data(name, shape, dtype, neg, range):


    shape_str = '_'.join(map(str, shape))
    case_name = shape_str + "_" + dtype
    path = "./../data/" + name + "/" + case_name+"/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    # print(path)
    inputname = 'input_' + '_'.join(map(str, shape)) + '.data'
    outputname = 'output_' + '_'.join(map(str, shape)) + '.data'
    input_path = path + inputname
    print(input_path)
    output_path = path + outputname
    max = range[1]
    min = range[0]
    if dtype == "float16":
        stype=np.float16
    elif dtype == "float32":
        stype = np.float32
    elif dtype == "int8":
        stype = np.int8
    elif dtype == "int32":
        strpe == np.int32
    else:
        raise RuntimeError("unsupported dtype_st:%s "%dtype)
    input_x = np.random.uniform(min, max, shape).astype(stype)
    input_x.tofile(input_path)
    res = np.maximum(neg*input_x, input_x)
    res.tofile(output_path)



def gen_leaky_relu_data(isBBIT=False):
    gen_leaky_data("leaky_relu", (1,16,10,10),"float16",0,(-5,5))
    gen_leaky_data("leaky_relu",(1,11,8,8),"float32",-0.3,(-1,1))
    gen_leaky_data("leaky_relu",(1,128),"int8", -5,(0,5))

if __name__ =="__main__":
    gen_leaky_relu_data()
