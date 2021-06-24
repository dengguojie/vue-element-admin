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

def test_Square_001_fuzz_case_001():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_001_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_002():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_002_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_003():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_003_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_004():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_004_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_005():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_005_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_006():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_006_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_007():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_007_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_008():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_008_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_009():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_009_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))

def test_Square_001_fuzz_case_010():
    
    input1 = np.fromfile('Square/run/out/test_data/data/Test_Square_001_fuzz_case_010_input_0.bin', np.float32)
    input1.shape = [1, 2]

    square_test = Net()
    
    start = time.time()
    
    output1 = square_test(Tensor(input1))

    end = time.time()
    
    print("running time: %.2f s" %(end-start))
