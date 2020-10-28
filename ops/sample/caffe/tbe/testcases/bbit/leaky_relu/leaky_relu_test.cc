/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include "log.hpp"
#include "./leaky_relu_test.hpp"
#include "one_in_one_out_layer.hpp"
typedef unsigned char   uint8;         /* Unsigned  8 bit quantity     */
typedef signed   char   int8;          /* Signed    8 bit quantity     */
using namespace std;


bool leaky_reluTest::leaky_relu_1_16_10_10_float16()
{
    std::string op_name = "leaky_relu";
    std::string inputSizeStr = "1_16_10_10_float16";
    uint32_t inputSize = 1*16*10*10;

    uint32_t outputSize = 1*16*10*10;

    std::string stubFunc =  "leaky_relu_1_16_10_10_float16__kernel0";

    std::string bin_path = "./llt/ops/common/kernel_bin/leaky_relu/leaky_relu_1_16_10_10_float16";

    std::string tilingName = "leaky_relu_1_16_10_10_float16__kernel0";

    std::string inputArrAPath = "./llt/ops/common/data/leaky_relu/1_16_10_10_float16/input_1_16_10_10.data";

    std::string expectOutputDataPath = "./llt/ops/common/data/leaky_relu/1_16_10_10_float16/output_1_16_10_10.data";
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
        (void*)stubFunc.c_str(),
        false
    };

    return layer.test();


}
bool leaky_reluTest::leaky_relu_1_11_8_8_float32() {
    std::string op_name = "leaky_relu";
    std::string inputSizeStr = "1_11_8_8_float32";
    uint32_t inputSize = 1*11*8*8;

    uint32_t outputSize = 1*11*8*8;

    std::string stubFunc =  "leaky_relu_1_11_8_8_float32__kernel0";

    std::string bin_path = "./llt/ops/common/kernel_bin/leaky_relu/leaky_relu_1_11_8_8_float32";

    std::string tilingName = "leaky_relu_1_11_8_8_float32__kernel0";

    std::string inputArrAPath = "./llt/ops/common/data/leaky_relu/1_11_8_8_float32/input_1_11_8_8.data";

    std::string expectOutputDataPath = "./llt/ops/common/data/leaky_relu/1_11_8_8_float32/output_1_11_8_8.data";
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
        (void*)stubFunc.c_str(),
        false
    };

    return layer.test();


}
bool leaky_reluTest::leaky_relu_1_128_int8() {
    std::string op_name = "leaky_relu";
    std::string inputSizeStr = "1_128_int8";
    uint32_t inputSize = 1*128;

    uint32_t outputSize = 1*128;

    std::string stubFunc =  "leaky_relu_1_128_int8__kernel0";

    std::string bin_path = "./llt/ops/common/kernel_bin/leaky_relu/leaky_relu_1_128_int8";

    std::string tilingName = "leaky_relu_1_128_int8__kernel0";

    std::string inputArrAPath = "./llt/ops/common/data/leaky_relu/1_128_int8/input_1_128.data";

    std::string expectOutputDataPath = "./llt/ops/common/data/leaky_relu/1_128_int8/output_1_128.data";
    float ratios[2] = {0.001 ,0.001};

    OneInOneOutLayer<int8,int8> layer{
        op_name,
        inputSizeStr,
        inputSize,
        outputSize,

        bin_path,
        tilingName,
        inputArrAPath,
        expectOutputDataPath,

        ratios,
        (void*)stubFunc.c_str(),
         false
    };

    return layer.test();

}
bool leaky_reluTest::test(string name)
{
    TVM_LOG(CC_LOG_INFO, "TVM batch_norm BBIT begin.");
    bool ret = false;
    if ("leaky_relu_1_11_8_8_float32" == name)
    {
    ret = leaky_relu_1_11_8_8_float32();
    if (!ret) {
        TVM_LOG(CC_LOG_ERROR, "leaky_relu_1_11_8_8_float32 falied");
        return false;
    }
    }
    else if ("leaky_relu_1_16_10_10_float16" == name)
    {
        ret = leaky_relu_1_16_10_10_float16();
        if (!ret) {
            TVM_LOG(CC_LOG_ERROR, "leaky_relu_1_16_10_10_float16 falied");
            return false;
        }
    }else if ("leaky_relu_1_128_int8" == name)
        {
            ret = leaky_relu_1_128_int8();
            if (!ret) {
                TVM_LOG(CC_LOG_ERROR, "leaky_relu_1_128_int8 falied");
                return false;
            }
        }else{
            TVM_LOG(CC_LOG_INFO, "The case[%s] not exist", name.c_str());
            return false;
        }
        TVM_LOG(CC_LOG_INFO, "TVM leaky_reluTest BBIT end.");
        return true;
        }
