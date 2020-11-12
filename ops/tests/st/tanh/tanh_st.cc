#include "gtest/gtest.h"
#include "./../comm/one_in_one_out_layer.hpp"

class TANH_ST : public testing::Test {
protected:
 static void SetUpTestCase() {
 std::cout << "TANH_ST ST SetUp" << std::endl;
 }
 static void TearDownTestCase() {
 std::cout << "TANH_ST ST TearDown" << std::endl;
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
* op: tanh
* input_shape: (3, 3, 5, 6, 7)
* output_shape: (3, 3, 5, 6, 7)
* stype: float16
* dtype: float16
*/
TEST_F(TANH_ST, test_tanh_3_3_5_6_7_float16) {
    std::string op_name = "tanh";
	std::string inputSizeStr = "3_3_5_6_7_float16";
	uint32_t inputSize = 3*3*5*6*7;
    uint32_t outputSize = 3*3*5*6*7;

	std::string stubFunc =  "cce_tanh_3_3_5_6_7_float16__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/tanh/cce_tanh_3_3_5_6_7_float16.o";	
	
	std::string tilingName = "cce_tanh_3_3_5_6_7_float16__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/tanh/3_3_5_6_7_float16/tanh_input_3_3_5_6_7_float16.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/tanh/3_3_5_6_7_float16/tanh_output_3_3_5_6_7_float16.data";
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
        "./llt/ops/common/data/tanh/3_3_5_6_7_float16/actual_tanh_output_3_3_5_6_7_float16.data",
        outputSize * sizeof(fp16_t));
    }

	//assert(true == ret);
}

/*
* op: tanh
* input_shape: (3, 3, 5, 6, 7)
* output_shape: (3, 3, 5, 6, 7)
* stype: float32
* dtype: float32
*/
TEST_F(TANH_ST, test_tanh_3_3_5_6_7_float32) {
    std::string op_name = "tanh";
	std::string inputSizeStr = "3_3_5_6_7_float32";
	uint32_t inputSize = 3*3*5*6*7;
    uint32_t outputSize = 3*3*5*6*7;

	std::string stubFunc =  "cce_tanh_3_3_5_6_7_float32__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/tanh/cce_tanh_3_3_5_6_7_float32.o";	
	
	std::string tilingName = "cce_tanh_3_3_5_6_7_float32__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/tanh/3_3_5_6_7_float32/tanh_input_3_3_5_6_7_float32.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/tanh/3_3_5_6_7_float32/tanh_output_3_3_5_6_7_float32.data";
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

    //if(!ret)
    {
        layer.writeBinaryFile((void*)layer.outputData,
        "./llt/ops/common/data/tanh/3_3_5_6_7_float32/actual_tanh_output_3_3_5_6_7_float32.data",
        outputSize * sizeof(float));
    }

	// assert(true == ret);
}

/*
* op: tanh
* input_shape: (1, 1)
* output_shape: (1, 1)
* stype: float16
* dtype: float16
*/
TEST_F(TANH_ST, test_tanh_1_1_float16) {
    std::string op_name = "tanh";
	std::string inputSizeStr = "1_1_float16";
	uint32_t inputSize = 1*1;
    uint32_t outputSize = 1*1;

	std::string stubFunc =  "cce_tanh_1_1_float16__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/tanh/cce_tanh_1_1_float16.o";	
	
	std::string tilingName = "cce_tanh_1_1_float16__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/tanh/1_1_float16/tanh_input_1_1_float16.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/tanh/1_1_float16/tanh_output_1_1_float16.data";
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
        "./llt/ops/common/data/tanh/1_1_float16/actual_tanh_output_1_1_float16.data",
        outputSize * sizeof(fp16_t));
    }

	assert(true == ret);	
}

/*
* op: tanh
* input_shape: (1, 1)
* output_shape: (1, 1)
* stype: float32
* dtype: float32
*/
TEST_F(TANH_ST, test_tanh_1_1_float32) {
    std::string op_name = "tanh";
	std::string inputSizeStr = "1_1_float32";
	uint32_t inputSize = 1*1;
    uint32_t outputSize = 1*1;

	std::string stubFunc =  "cce_tanh_1_1_float32__kernel0";

	std::string bin_path = "./llt/ops/common/kernel_bin/tanh/cce_tanh_1_1_float32.o";	
	
	std::string tilingName = "cce_tanh_1_1_float32__kernel0";
	
	std::string inputArrAPath = "./llt/ops/common/data/tanh/1_1_float32/tanh_input_1_1_float32.data";

	std::string expectOutputDataPath = "./llt/ops/common/data/tanh/1_1_float32/tanh_output_1_1_float32.data";
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
        "./llt/ops/common/data/tanh/1_1_float32/actual_tanh_output_1_1_float32.data",
        outputSize * sizeof(float));
    }

	assert(true == ret);	
}

