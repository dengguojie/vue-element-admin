/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "gtest/gtest.h"
#include "one_in_one_out_layer.hpp"

class TEST_OP_ST : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "TEST_OP_ST ST SetUp" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "TEST_OP_ST ST TearDown" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() {}
};
