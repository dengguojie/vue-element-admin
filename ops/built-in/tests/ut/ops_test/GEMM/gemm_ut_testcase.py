# -*- coding: UTF-8 -*-
alpha_beta = (2, 3)
data_range = (0, 5)

gemm_op_testcase = [
  #"FRACTAL_NZ testcase"   
  # fp16->fp16
 ((32, 64), (64, 96), 'float16', 'float16', True, (32, 96),'FRACTAL_NZ', False, False),
 ((1, 256), (256, 96), 'float16', 'float16', True, (1, 96),'FRACTAL_NZ', False, False),
 ((16, 256), (256, 1), 'float16', 'float16', True, (16, 1),'FRACTAL_NZ', False, False),

 # nobias
 ((16, 16), (16, 16), 'float16', 'float32', True, (16, 16),'FRACTAL_NZ', False, False),
 ((1, 256), (256, 96), 'float16', 'float32', True, (1, 96),'FRACTAL_NZ', False, False),
 ((16, 256), (256, 1), 'float16', 'float32', True, (16, 1),'FRACTAL_NZ', False, False),

 # int8int8->int32
 ((32, 64), (64, 96), 'int8', 'int32', False, (32, 96),'FRACTAL_NZ', False, False),

 # int8int8->float32 ND transpose
 ((256, 32), (96, 256), 'int8', 'float32', True, (32, 96),'ND', True, True),

 # int8int8->float32 ND
 ((32, 256), (256, 96), 'int8', 'float32', True, (32, 96),'ND', False, False),
 ((15, 31), (31, 16), 'int8', 'float32', True, (15, 16),'ND', False, False),

 # nt8int8->int32
 ((32, 256), (256, 96), 'int8', 'float32', False, (32, 96),'FRACTAL_NZ', False, False),

 #ND testcase fp16fp16
 ((16, 16), (16, 16), 'float16', 'float16', True, (16, 16),'ND', False, False),
 ((16, 16), (16, 16), 'float16', 'float32', True, (16, 16),'ND', False, False),

 # ND testcase fp16fp16 transpose
 ((64, 32), (64, 96), 'float16', 'float16', True, (32, 96), 'ND', True, False),
 ((32, 64), (96, 64), 'float16', 'float16', True, (32, 96), 'ND', False, True),
 ((16, 16), (16, 16), 'float16', 'float32', True, (16, 16), 'ND', True, True),

  # ND testcase int8int32 transpose
  ((64, 32), (96, 64), 'int8', 'int32', False, (32, 96),'ND', True, True),
  ((17, 63), (64, 17), 'int8', 'int32', False, (63, 64),'ND', True, True),

  # ND int8->int32
  ((32, 64), (64, 96), 'int8', 'int32', False, (32, 96),'ND', False, False),
  ((63, 17), (17, 64), 'int8', 'int32', False, (63, 64),'ND', False, False),

  # ND testcase int8int32 transpose
  ((16, 2), (1024, 16), 'int8', 'float32', False, (2, 1024),'ND', True, True),

  ((1, 6259), (6259, 5552), 'float16', 'float16', False, (1, 5552),'ND', False, False),
  # Add gemm Bias Case
  ((4429, 1844), (1844, 1), 'float16', 'float16', False, (4429, 1),'FRACTAL_NZ', False, False),
]
