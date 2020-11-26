#include "gtest/gtest.h"
#include "./../comm/one_in_one_out_layer.hpp"

class PAD_ST : public testing::Test {
protected:
 static void SetUpTestCase() {
 std::cout << "PAD_ST ST SetUp" << std::endl;
 }
 static void TearDownTestCase() {
 std::cout << "PAD_ST ST TearDown" << std::endl;
 }
 // Some expensive resource shared by all tests.
 virtual void SetUp()
 {
 }
 virtual void TearDown()
 {
 }
};

/*
* op: pad
* input_shape: [1]
* output_shape: [3]
* stype: float32
* dtype: float32
*/
TEST_F(PAD_ST, test_pad_1_int32_1_1_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "1_int32_1_1_constant_0";
	uint32_t inputSize = 1;
    uint32_t outputSize = 3;

	std::string stubFunc =  "cce_pad_1_int32_1_1_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_1_int32_1_1_constant_0.o";
	
	std::string tilingName = "cce_pad_1_int32_1_1_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/1_int32_1_1_constant_0/pad_input_1_int32_1_1_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/1_int32_1_1_constant_0/pad_output_3_int32_1_1_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<int32_t,int32_t> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/1_int32_1_1_constant_0/actual_pad_output_3_int32_1_1_constant_0.data",
        outputSize * sizeof(int32_t));
    }

	assert(true == ret);
}

/*
* op: pad
* input_shape: [17]
* output_shape: [35]
* stype: float16
* dtype: float16
*/
TEST_F(PAD_ST, test_pad_17_float16_1_17_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "17_float16_1_17_constant_0";
	uint32_t inputSize = 17;
    uint32_t outputSize = 35;

	std::string stubFunc =  "cce_pad_17_float16_1_17_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_17_float16_1_17_constant_0.o";	
	
	std::string tilingName = "cce_pad_17_float16_1_17_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/17_float16_1_17_constant_0/pad_input_17_float16_1_17_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/17_float16_1_17_constant_0/pad_output_35_float16_1_17_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<fp16_t,fp16_t> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/17_float16_1_17_constant_0/actual_pad_output_35_float16_1_17_constant_0.data",
        outputSize * sizeof(fp16_t));
    }

	assert(true == ret);
}

/*
* op: pad
* input_shape: [3, 3, 131075]
* output_shape: [3, 6, 131075]
* stype: float32
* dtype: float32
*/
/*TEST_F(PAD_ST, test_pad_3_3_131075_float32_0_0_0_3_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "3_3_131075_float32_0_0_0_3_0_0_constant_0";
	uint32_t inputSize = 3*3*131075;
    uint32_t outputSize = 3*6*131075;

	std::string stubFunc =  "cce_pad_3_3_131075_float32_0_0_0_3_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_3_3_131075_float32_0_0_0_3_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_3_3_131075_float32_0_0_0_3_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/3_3_131075_float32_0_0_0_3_0_0_constant_0/pad_input_3_3_131075_float32_0_0_0_3_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/3_3_131075_float32_0_0_0_3_0_0_constant_0/pad_output_3_6_131075_float32_0_0_0_3_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<float,float> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/3_3_131075_float32_0_0_0_3_0_0_constant_0/actual_pad_output_3_6_131075_float32_0_0_0_3_0_0_constant_0.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);	
}*/

/*
* op: pad
* input_shape: [3, 131075]
* output_shape: [6, 131075]
* stype: float16
* dtype: float16
*/
/*TEST_F(PAD_ST, test_pad_3_131075_float16_3_0_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "3_131075_float16_3_0_0_0_constant_0";
	uint32_t inputSize = 3*131075;
    uint32_t outputSize = 6*131075;

	std::string stubFunc =  "cce_pad_3_131075_float16_3_0_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_3_131075_float16_3_0_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_3_131075_float16_3_0_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/3_131075_float16_3_0_0_0_constant_0/pad_input_3_131075_float16_3_0_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/3_131075_float16_3_0_0_0_constant_0/pad_output_6_131075_float16_3_0_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<fp16_t,fp16_t> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/3_131075_float16_3_0_0_0_constant_0/actual_pad_output_6_131075_float16_3_0_0_0_constant_0.data",
        outputSize * sizeof(fp16_t));
    }

	assert(true == ret);	
}*/

/*
* op: pad
* input_shape: [3, 17]
* output_shape: [9, 34]
* stype: float16
* dtype: float16
*/
TEST_F(PAD_ST, test_pad_3_17_int8_3_3_0_17_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "3_17_int8_3_3_0_17_constant_0";
	uint32_t inputSize = 3*17;
    uint32_t outputSize = 9*34;

	std::string stubFunc =  "cce_pad_3_17_int8_3_3_0_17_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_3_17_int8_3_3_0_17_constant_0.o";
	
	std::string tilingName = "cce_pad_3_17_int8_3_3_0_17_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/3_17_int8_3_3_0_17_constant_0/pad_input_3_17_int8_3_3_0_17_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/3_17_int8_3_3_0_17_constant_0/pad_output_9_34_int8_3_3_0_17_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<int8_t,int8_t> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/3_17_int8_3_3_0_17_constant_0/actual_pad_output_9_34_int8_3_3_0_17_constant_0.data",
        outputSize * sizeof(int8_t));
    }

	assert(true == ret);
}

/*
* op: pad
* input_shape: [3, 1027]
* output_shape: [6, 1027]
* stype: float32
* dtype: float32
*/
TEST_F(PAD_ST, test_pad_3_1027_uint8_3_0_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "3_1027_uint8_3_0_0_0_constant_0";
	uint32_t inputSize = 3*1027;
    uint32_t outputSize = 6*1027;

	std::string stubFunc =  "cce_pad_3_1027_uint8_3_0_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_3_1027_uint8_3_0_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_3_1027_uint8_3_0_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/3_1027_uint8_3_0_0_0_constant_0/pad_input_3_1027_uint8_3_0_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/3_1027_uint8_3_0_0_0_constant_0/pad_output_6_1027_uint8_3_0_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<uint8_t,uint8_t> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/3_1027_uint8_t_3_0_0_0_constant_0/actual_pad_output_6_1027_uint8_3_0_0_0_constant_0.data",
        outputSize * sizeof(uint8_t));
    }

	assert(true == ret);
}

/*
* op: pad
* input_shape: [32, 128, 1024]
* output_shape: [32, 512, 1024]
* stype: float16
* dtype: float16
*/
/*TEST_F(PAD_ST, test_pad_32_128_1024_float16_0_0_0_384_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "32_128_1024_float16_0_0_0_384_0_0_constant_0";
	uint32_t inputSize = 32*128*1024;
    uint32_t outputSize = 32*512*1024;

	std::string stubFunc =  "cce_pad_32_128_1024_float16_0_0_0_384_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_32_128_1024_float16_0_0_0_384_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_32_128_1024_float16_0_0_0_384_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/32_128_1024_float16_0_0_0_384_0_0_constant_0/pad_input_32_128_1024_float16_0_0_0_384_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/32_128_1024_float16_0_0_0_384_0_0_constant_0/pad_output_32_512_1024_float16_0_0_0_384_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<fp16_t,fp16_t> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/32_128_1024_float16_0_0_0_384_0_0_constant_0/actual_pad_output_32_512_1024_float16_0_0_0_384_0_0_constant_0.data",
        outputSize * sizeof(fp16_t));
    }

	assert(true == ret);	
}*/

/*
* op: pad
* input_shape: [2, 2, 16]
* output_shape: [2, 6, 16]
* stype: float16
* dtype: float16
*/
TEST_F(PAD_ST, test_pad_2_2_16_float16_0_0_1_3_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "2_2_16_float16_0_0_1_3_0_0_constant_0";
	uint32_t inputSize = 2*2*16;
    uint32_t outputSize = 2*6*16;

	std::string stubFunc =  "cce_pad_2_2_16_float16_0_0_1_3_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_2_2_16_float16_0_0_1_3_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_2_2_16_float16_0_0_1_3_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/2_2_16_float16_0_0_1_3_0_0_constant_0/pad_input_2_2_16_float16_0_0_1_3_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/2_2_16_float16_0_0_1_3_0_0_constant_0/pad_output_2_6_16_float16_0_0_1_3_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<fp16_t,fp16_t> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/2_2_16_float16_0_0_1_3_0_0_constant_0/actual_pad_output_2_6_16_float16_0_0_1_3_0_0_constant_0.data",
        outputSize * sizeof(fp16_t));
    }

	assert(true == ret);	
}

/*
* op: pad
* input_shape: [2, 2, 9]
* output_shape: [16, 16, 23]
* stype: float32
* dtype: float32
*/
TEST_F(PAD_ST, test_pad_2_2_9_float32_7_7_7_7_7_7_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "2_2_9_float32_7_7_7_7_7_7_constant_0";
	uint32_t inputSize = 2*2*9;
    uint32_t outputSize = 16*16*23;

	std::string stubFunc =  "cce_pad_2_2_9_float32_7_7_7_7_7_7_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_2_2_9_float32_7_7_7_7_7_7_constant_0.o";	
	
	std::string tilingName = "cce_pad_2_2_9_float32_7_7_7_7_7_7_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/2_2_9_float32_7_7_7_7_7_7_constant_0/pad_input_2_2_9_float32_7_7_7_7_7_7_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/2_2_9_float32_7_7_7_7_7_7_constant_0/pad_output_16_16_23_float32_7_7_7_7_7_7_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<float,float> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/2_2_9_float32_7_7_7_7_7_7_constant_0/actual_pad_output_16_16_23_float32_7_7_7_7_7_7_constant_0.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);	
}

/*
* op: pad
* input_shape: [2, 2, 1027]
* output_shape: [2, 18, 1027]
* stype: float32
* dtype: float32
*/
TEST_F(PAD_ST, test_pad_2_2_1027_float32_0_0_0_16_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "2_2_1027_float32_0_0_0_16_0_0_constant_0";
	uint32_t inputSize = 2*2*1027;
    uint32_t outputSize = 2*18*1027;

	std::string stubFunc =  "cce_pad_2_2_1027_float32_0_0_0_16_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_2_2_1027_float32_0_0_0_16_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_2_2_1027_float32_0_0_0_16_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/2_2_1027_float32_0_0_0_16_0_0_constant_0/pad_input_2_2_1027_float32_0_0_0_16_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/2_2_1027_float32_0_0_0_16_0_0_constant_0/pad_output_2_18_1027_float32_0_0_0_16_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<float,float> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/2_2_1027_float32_0_0_0_16_0_0_constant_0/actual_pad_output_2_18_1027_float32_0_0_0_16_0_0_constant_0.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);	
}

/*
* op: pad
* input_shape: [2, 2, 1027]
* output_shape: [2, 9, 1034]
* stype: float16
* dtype: float16
*/
TEST_F(PAD_ST, test_pad_2_2_1027_float16_0_0_0_7_0_7_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "2_2_1027_float16_0_0_0_7_0_7_constant_0";
	uint32_t inputSize = 2*2*1027;
    uint32_t outputSize = 2*9*1034;

	std::string stubFunc =  "cce_pad_2_2_1027_float16_0_0_0_7_0_7_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_2_2_1027_float16_0_0_0_7_0_7_constant_0.o";	
	
	std::string tilingName = "cce_pad_2_2_1027_float16_0_0_0_7_0_7_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/2_2_1027_float16_0_0_0_7_0_7_constant_0/pad_input_2_2_1027_float16_0_0_0_7_0_7_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/2_2_1027_float16_0_0_0_7_0_7_constant_0/pad_output_2_9_1034_float16_0_0_0_7_0_7_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<fp16_t,fp16_t> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/2_2_1027_float16_0_0_0_7_0_7_constant_0/actual_pad_output_2_9_1034_float16_0_0_0_7_0_7_constant_0.data",
        outputSize * sizeof(fp16_t));
    }

	assert(true == ret);	
}

/*
* op: pad
* input_shape: [2, 2, 9]
* output_shape: [2, 18, 9]
* stype: float32
* dtype: float32
*/
TEST_F(PAD_ST, test_pad_2_2_9_float32_0_0_0_16_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "2_2_9_float32_0_0_0_16_0_0_constant_0";
	uint32_t inputSize = 2*2*9;
    uint32_t outputSize = 2*18*9;

	std::string stubFunc =  "cce_pad_2_2_9_float32_0_0_0_16_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_2_2_9_float32_0_0_0_16_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_2_2_9_float32_0_0_0_16_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/2_2_9_float32_0_0_0_16_0_0_constant_0/pad_input_2_2_9_float32_0_0_0_16_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/2_2_9_float32_0_0_0_16_0_0_constant_0/pad_output_2_18_9_float32_0_0_0_16_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<float,float> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/2_2_9_float32_0_0_0_16_0_0_constant_0/actual_pad_output_2_18_9_float32_0_0_0_16_0_0_constant_0.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);	
}

/*
* op: pad
* input_shape: [2, 2, 63]
* output_shape: [2, 9, 70]
* stype: float32
* dtype: float32
*/
TEST_F(PAD_ST, test_pad_2_2_63_float32_0_0_0_7_0_7_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "2_2_63_float32_0_0_0_7_0_7_constant_0";
	uint32_t inputSize = 2*2*63;
    uint32_t outputSize = 2*9*70;

	std::string stubFunc =  "cce_pad_2_2_63_float32_0_0_0_7_0_7_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_2_2_63_float32_0_0_0_7_0_7_constant_0.o";	
	
	std::string tilingName = "cce_pad_2_2_63_float32_0_0_0_7_0_7_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/2_2_63_float32_0_0_0_7_0_7_constant_0/pad_input_2_2_63_float32_0_0_0_7_0_7_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/2_2_63_float32_0_0_0_7_0_7_constant_0/pad_output_2_9_70_float32_0_0_0_7_0_7_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<float,float> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/2_2_63_float32_0_0_0_7_0_7_constant_0/actual_pad_output_2_9_70_float32_0_0_0_7_0_7_constant_0.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);	
}

/*
* op: pad
* input_shape: [2, 2, 63]
* output_shape: [2, 18, 63]
* stype: float32
* dtype: float32
*/
TEST_F(PAD_ST, test_pad_2_2_63_float32_0_0_0_16_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "2_2_63_float32_0_0_0_16_0_0_constant_0";
	uint32_t inputSize = 2*2*63;
    uint32_t outputSize = 2*18*63;

	std::string stubFunc =  "cce_pad_2_2_63_float32_0_0_0_16_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_2_2_63_float32_0_0_0_16_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_2_2_63_float32_0_0_0_16_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/2_2_63_float32_0_0_0_16_0_0_constant_0/pad_input_2_2_63_float32_0_0_0_16_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/2_2_63_float32_0_0_0_16_0_0_constant_0/pad_output_2_18_63_float32_0_0_0_16_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<float,float> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/2_2_63_float32_0_0_0_16_0_0_constant_0/actual_pad_output_2_18_63_float32_0_0_0_16_0_0_constant_0.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);	
}

/*
* op: pad
* input_shape: [2, 2, 32640]
* output_shape: [2, 3, 32640]
* stype: float32
* dtype: float32
*/
/*TEST_F(PAD_ST, test_pad_2_2_32640_float32_0_0_0_1_0_0_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "2_2_32640_float32_0_0_0_1_0_0_constant_0";
	uint32_t inputSize = 2*2*32640;
    uint32_t outputSize = 2*3*32640;

	std::string stubFunc =  "cce_pad_2_2_32640_float32_0_0_0_1_0_0_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_2_2_32640_float32_0_0_0_1_0_0_constant_0.o";	
	
	std::string tilingName = "cce_pad_2_2_32640_float32_0_0_0_1_0_0_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/2_2_32640_float32_0_0_0_1_0_0_constant_0/pad_input_2_2_32640_float32_0_0_0_1_0_0_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/2_2_32640_float32_0_0_0_1_0_0_constant_0/pad_output_2_3_32640_float32_0_0_0_1_0_0_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<float,float> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/2_2_32640_float32_0_0_0_1_0_0_constant_0/actual_pad_output_2_3_32640_float32_0_0_0_1_0_0_constant_0.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);	
}*/

/*
* op: pad
* input_shape: [3, 17]
* output_shape: [9, 20]
* stype: float32
* dtype: float32
*/
TEST_F(PAD_ST, test_pad_3_17_float32_3_3_0_3_constant_0) {
    std::string op_name = "pad";
	std::string inputSizeStr = "3_17_float32_3_3_0_3_constant_0";
	uint32_t inputSize = 3*17;
    uint32_t outputSize = 9*20;

	std::string stubFunc =  "cce_pad_3_17_float32_3_3_0_3_constant_0__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/pad/cce_pad_3_17_float32_3_3_0_3_constant_0.o";
	
	std::string tilingName = "cce_pad_3_17_float32_3_3_0_3_constant_0__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/pad/3_17_float32_3_3_0_3_constant_0/pad_input_3_17_float32_3_3_0_3_constant_0.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/pad/3_17_float32_3_3_0_3_constant_0/pad_output_9_20_float32_3_3_0_3_constant_0.data";
    float ratios[2] = {0.001 ,0.001};

	OneInOneOutLayer<float,float> layer{
		op_name,
		inputSizeStr,
		inputSize,
		outputSize,
		bin_path,
		tilingName,
		inputArrAPath,
		expectOutputDataPath,
		ratios,
		(void*)stubFunc.c_str()
	};

	bool ret = layer.test();

    if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/pad/3_17_float32_3_3_0_3_constant_0/actual_pad_output_9_20_float32_3_3_0_3_constant_0.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);
}

