# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np



def matrix_to_zN(matrix, shape, dtype):  # m, n
    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0

    j_outer = max(w // 16,1)
    i_outer = max(h // 16,1)
    j_inner = 16 if w//16>0 else 1
    i_inner = 16 if h//16>0 else 1

    if len(shape) > 2:
        for batch in range(np.prod(shape[:-2])):
            for j in range(0, j_outer):
                for i in range(0, i_outer):
                    for ii in range(0, i_inner):
                        for jj in range(0, j_inner):
                            tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
        res_shape = (np.prod(shape[:-2]),j_outer,i_outer,i_inner,j_inner)
    else:
        for j in range(0, j_outer):
            for i in range(0, i_outer):
                for ii in range(0, i_inner):
                    for jj in range(0, j_inner):
                        tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                        idx = idx + 1
        res_shape = (j_outer,i_outer,i_inner,j_inner)
    return tmp,res_shape

def maxtrix_zN_reverse(matrix, shape, dtype):
    idx = 0
    j_outer,i_outer,i_inner,j_inner = shape[-4],shape[-3],shape[-2],shape[-1]
    h = i_outer*i_inner
    w = j_outer*j_inner

    if len(shape) is 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape,h,w), dtype=dtype)
        print((batch_shape,h,w),matrix.shape)
        for batch in range(batch_shape):
            for j in range(0, j_outer):
                for i in range(0, i_outer):
                    for ii in range(0, i_inner):
                        for jj in range(0, j_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    elif len(shape) is 4:
        tmp = np.zeros((h,w), dtype=dtype)
        for j in range(0, j_outer):
            for i in range(0, i_outer):
                for ii in range(0, i_inner):
                    for jj in range(0, j_inner):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix[idx]
                        idx = idx + 1
        print((h,w))

    return tmp


    idx = 0
    if len(shape)==2:
        h = shape[0]*16
        tmp = np.zeros((h,1), dtype=dtype)
        for i in range(0, h // 16):
            tmp[idx][0]= matrix[idx]
            idx = idx + 1
    if len(shape)==3:
        batch = shape[0]
        h = shape[1]*16
        tmp = np.zeros((batch,h,1), dtype=dtype)
        for batch in range(np.prod(shape[:-2])):
            for i in range(0, h):
                tmp[batch][i][0] = matrix[idx]
                idx = idx + 1
    elif len(shape)==4:
        h,w = shape[0]*16,shape[1]*16
        tmp = np.zeros((h,w), dtype=dtype)
        for i in range(0, h // 16):
            for j in range(0, w // 16):
                for jj in range(0, 16):
                    for ii in range(0, 16):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix[idx]
                        idx = idx + 1
    elif len(shape)==5:
        batch = shape[0]
        h,w = shape[1]*16,shape[2]*16
        tmp = np.zeros((batch,h,w), dtype=dtype)
        for batch in range(0, np.prod(shape[:-4])):
            for i in range(0, h // 16):
                for j in range(0, w // 16):
                    for jj in range(0, 16):
                        for ii in range(0, 16):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    return tmp


def matrix_to_nZ(matrix, shape, dtype):  # k, n
    h = shape[-2]
    w = shape[-1]
    tmp = np.zeros(np.prod(shape), dtype=dtype)
    idx = 0

    j_outer = max(w // 16,1)
    i_outer = max(h // 16,1)
    j_inner = 16 if w//16>0 else 1
    i_inner = 16 if h//16>0 else 1

    if len(shape) > 2:
        for batch in range(np.prod(shape[:-2])):
            for i in range(0, i_outer):
                for j in range(0, j_outer):
                    for jj in range(0, j_inner):
                        for ii in range(0, i_inner):
                            tmp[idx] = matrix[batch][i * 16 + ii][j * 16 + jj]
                            idx = idx + 1
        res_shape = (np.prod(shape[:-2]),i_outer,j_outer,j_inner,i_inner)
    else:
        for i in range(0, i_outer):
            for j in range(0, j_outer):
                for jj in range(0, j_inner):
                    for ii in range(0, i_inner):
                        tmp[idx] = matrix[i * 16 + ii][j * 16 + jj]
                        idx = idx + 1
        res_shape = (i_outer,j_outer,j_inner,i_inner)
    return tmp,res_shape



def maxtrix_nZ_reverse(matrix, shape, dtype):

    idx = 0
    i_outer,j_outer,j_inner,i_inner = shape[-4],shape[-3],shape[-2],shape[-1]
    h = i_outer*i_inner
    w = j_outer*j_inner

    if len(shape) is 5:
        batch_shape = shape[0]
        tmp = np.zeros((batch_shape,h,w), dtype=dtype)
        print((batch_shape,h,w),matrix.shape)
        for batch in range(batch_shape):
            for i in range(0, i_outer):
                for j in range(0, j_outer):
                    for jj in range(0, j_inner):
                        for ii in range(0, i_inner):
                            tmp[batch][i * 16 + ii][j * 16 + jj] = matrix[idx]
                            idx = idx + 1
    elif len(shape) is 4:
        tmp = np.zeros((h,w), dtype=dtype)
        for i in range(0, i_outer):
            for j in range(0, j_outer):
                for jj in range(0, j_inner):
                    for ii in range(0, i_inner):
                        tmp[i * 16 + ii][j * 16 + jj]= matrix[idx]
                        idx = idx + 1
        print((h,w))

    return tmp
'''
------------------------------------------------------------------------nZ Test
'''

'''shape_x=(32,32)
src_type = "float32"
x = np.random.uniform(-2, 2, shape_x).astype(src_type)
x_nZ = matrix_to_nZ(x,x.shape,x.dtype)
x_reverse = maxtrix_nZ_reverse(x_nZ,(h//16,w//16,16,16),x_nZ.dtype)
np.allclose(x,x_reverse)'''
'''
#测试用例nZ分形和逆变换
for shape_x in [(32,32),(3,32,32),(32,1),(3,32,1)]:
    src_type = "float32"
    print("&&&"*20)
    print(shape_x)
    x = np.random.uniform(-2, 2, shape_x).astype(src_type)
    x_nZ,nZ_shape = matrix_to_nZ(x,x.shape,x.dtype)
    print(nZ_shape)
    x_reverse = maxtrix_nZ_reverse(x_nZ,nZ_shape,x_nZ.dtype)
    #print(type(x),type(x_reverse))
    #equal = np.abs((x-x_reverse)/x)<0.0001
    #if np.all(equal):
    #    print("EQUAL")
    #else:
    #    print("ERROR")
    np.allclose(x,x_reverse)


#测试用例zN分形和逆变换
for shape_x in [(32,32),(3,32,32),(32,1),(3,32,1)]:
    src_type = "float32"
    print("&&&"*20)
    print(shape_x)
    x = np.random.uniform(-2, 2, shape_x).astype(src_type)
    x_nZ,nZ_shape = matrix_to_zN(x,x.shape,x.dtype)
    print(nZ_shape)
    x_reverse = maxtrix_zN_reverse(x_nZ,nZ_shape,x_nZ.dtype)
    #print(type(x),type(x_reverse))
    #equal = np.abs((x-x_reverse)/x)<0.0001
    #if np.all(equal):
    #    print("EQUAL")
    #else:
    #    print("ERROR")
    np.allclose(x,x_reverse)'''



