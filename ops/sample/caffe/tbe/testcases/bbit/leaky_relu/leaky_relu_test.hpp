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

#include "tvm_bbit.hpp"
#include "register.hpp"
using namespace std;

class leaky_reluTest : public BaseBbitTest{
public:
    leaky_reluTest(){
        testcases.push_back("leaky_relu_1_11_8_8_float32");
        testcases.push_back("leaky_relu_1_16_10_10_float16");
        testcases.push_back("leaky_relu_1_128_int8");

    };

    virtual ~leaky_reluTest() {};

    bool test(string name);
    bool leaky_relu_1_16_10_10_float16();
    bool leaky_relu_1_11_8_8_float32();
    bool leaky_relu_1_128_int8();

};

REGISTER_CLASS(leaky_reluTest)
