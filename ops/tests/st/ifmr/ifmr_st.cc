/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Description: Huawei Code
 *
 * Author: Huawei
 *
 * Create: 2020-01-01
 *
 */

#include "gtest/gtest.h"
#include "four_in_two_out_layer.hpp"

class IFMR_ST : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IFMR_ST ST SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "IFMR_ST ST TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() {}
    virtual void TearDown() {}
};

/*
* op: ifmr
* input_shape: (10, 10)
* input_shape1: (1)
* input_shape2: (1)
* input_shape3: (10)
* output_shape: (1)
* output1_shape: (1)
* stype: float32
* dtype: float32
*/
TEST_F(IFMR_ST, test_ifmr_10_10_float32)
{
    std::string op_name = "ifmr";
    std::string inputSizeStr = "10_10_float32";
    uint32_t inputSize = 10 * 10;
    uint32_t inputBSize = 1;
    uint32_t inputCSize = 1;
    uint32_t inputDSize = 10;
    uint32_t outputSize = 1;
    uint32_t outputBSize = 1;

    const char *stubFunc = "cce_ifmr_10_10_float32__kernel0";

    std::string bin_path = "./llt/ops/common/kernel_bin/ifmr/cce_ifmr_10_10_float32.o";

    std::string tilingName = "cce_ifmr_10_10_float32__kernel0";

    std::string inputArrAPath = "./llt/ops/common/data/ifmr/10_10_float32/ifmr_input1_10_10_float32.data";
    std::string inputArrBPath = "./llt/ops/common/data/ifmr/10_10_float32/ifmr_input2_1_float32.data";
    std::string inputArrCPath = "./llt/ops/common/data/ifmr/10_10_float32/ifmr_input3_1_float32.data";
    std::string inputArrDPath = "./llt/ops/common/data/ifmr/10_10_float32/ifmr_input4_10_float32.data";

    std::string expectOutputDataPath = "./llt/ops/common/data/ifmr/10_10_float32/ifmr_output1_1_float32.data";
    std::string expectOutputBDataPath = "./llt/ops/common/data/ifmr/10_10_float32/ifmr_output2_1_float32.data";
//    float ratios[2] = {0.0001, 0.0001};
    float ratios[2] = {1, 0.1};

    FourInTwoOutLayer<float, float, float, float, float> layer{op_name, inputSizeStr, inputSize, inputBSize, inputCSize, inputDSize, outputSize,
        outputBSize, bin_path, tilingName, inputArrAPath, inputArrBPath, inputArrCPath, inputArrDPath, expectOutputDataPath, expectOutputBDataPath,
        ratios, (void *)stubFunc};

    bool ret = layer.test();

    if (!ret) {
        layer.writeBinaryFile((void *)layer.outputData,
            "./llt/ops/common/data/ifmr/10_10_float32/actual_add_output_10_10_float32.data", outputSize * sizeof(float));
    }

    assert(true == ret);
}

/*
* op: ifmr
* input_shape: (64, 64, 3)
* input_shape1: (1)
* input_shape2: (1)
* input_shape3: (10)
* output_shape: (1)
* output1_shape: (1)
* stype: float32
* dtype: float32
*/
TEST_F(IFMR_ST, test_add_64_64_3_float32)
{
    std::string op_name = "ifmr";
    std::string inputSizeStr = "64_64_3_float32";
    uint32_t inputSize = 64 * 64 * 3;
    uint32_t inputBSize = 1;
    uint32_t inputCSize = 1;
    uint32_t inputDSize = 10;
    uint32_t outputSize = 1;
    uint32_t outputBSize = 1;

    const char *stubFunc = "cce_ifmr_64_64_3_float32__kernel0";

    std::string bin_path = "./llt/ops/common/kernel_bin/ifmr/cce_ifmr_64_64_3_float32.o";

    std::string tilingName = "cce_ifmr_64_64_3_float32__kernel0";

    std::string inputArrAPath = "./llt/ops/common/data/ifmr/64_64_3_float32/ifmr_input1_64_64_3_float32.data";
    std::string inputArrBPath = "./llt/ops/common/data/ifmr/64_64_3_float32/ifmr_input2_1_float32.data";
    std::string inputArrCPath = "./llt/ops/common/data/ifmr/64_64_3_float32/ifmr_input3_1_float32.data";
    std::string inputArrDPath = "./llt/ops/common/data/ifmr/64_64_3_float32/ifmr_input4_10_float32.data";

    std::string expectOutputDataPath = "./llt/ops/common/data/ifmr/64_64_3_float32/ifmr_output1_1_float32.data";
    std::string expectOutputBDataPath = "./llt/ops/common/data/ifmr/64_64_3_float32/ifmr_output2_1_float32.data";
    //    float ratios[2] = {0.0001, 0.0001};
    float ratios[2] = {1, 0.1};

    FourInTwoOutLayer<float, float, float, float, float> layer{op_name, inputSizeStr, inputSize, inputBSize, inputCSize, inputDSize, outputSize,
        outputBSize, bin_path, tilingName, inputArrAPath, inputArrBPath, inputArrCPath, inputArrDPath, expectOutputDataPath, expectOutputBDataPath,
        ratios, (void *)stubFunc};

    bool ret = layer.test();

    if (!ret) {
        layer.writeBinaryFile((void *)layer.outputData,
        "./llt/ops/common/data/ifmr/64_64_3_float32/actual_add_output_64_64_3_float32.data", outputSize * sizeof(float));
    }

    assert(true == ret);
}
