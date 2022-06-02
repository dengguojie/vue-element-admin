import numpy as np
import pytest
import time
import logging
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor

# Import the definition of the Square primtive.
from square import Square
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
logger = logging.getLogger(__name__)


class Net(nn.Cell):
    """Net definition"""

    def __init__(self):
        super(Net, self).__init__()
        self.square = Square()

    def construct(self,input1):
        return self.square(input1)

def test_Square_001_fuzz_case_001_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_001_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_002_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_002_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_003_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_003_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_004_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_004_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_005_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_005_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_006_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_006_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_007_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_007_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_008_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_008_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_009_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_009_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_010_float32():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_010_float32_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))
