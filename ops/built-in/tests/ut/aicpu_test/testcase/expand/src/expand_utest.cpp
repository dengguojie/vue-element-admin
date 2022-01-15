/***
 *  Copyright Huawei Technologies Co., Ltd. 2021-2021.All rights reserved.
 *  Description:This file provides the function of expandding.
 *  Author: Huawei.
 *  Create:2021-10-08.
 ***/
#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include <cmath>

#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_EXPAND_UT : public testing::Test {};

#define CREATE_EXPAND_NODEDEF(shapes, data_types, datas)          \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();  \
  NodeDefBuilder(node_def.get(), "Expand", "Expand")            \
      .Input({"tensor", data_types[0], shapes[0], datas[0]})             \
      .Input({"shape", data_types[1], shapes[1], datas[1]})       \
      .Output({"out", data_types[2], shapes[2], datas[2]});

TEST_F(TEST_EXPAND_UT, TestExpandHigh) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 1, 6}, {3}, {2, 3, 6}}; 
    int32_t input0[12] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
    int32_t input1[3] = {2, 3, 6};
    int32_t output[36] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[36] = {1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6};
    EXPECT_EQ(CompareResult<int32_t>(output, expect_out, 36), true);
}

TEST_F(TEST_EXPAND_UT, TestExpandLow) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 4}, {3}, {2, 2, 4}}; 
    int32_t input0[8] = {1, 1, 2, 2, 3, 3, 4, 4};
    int32_t input1[3] = {2, 2, 4};
    int32_t output[16] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[16] = {1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4};
    EXPECT_EQ(CompareResult<int32_t>(output, expect_out, 16), true);
}

TEST_F(TEST_EXPAND_UT, TestExpandMid) {
    vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 4}, {3}, {2, 2, 4}}; 
    int64_t input0[8] = {1, 1, 2, 2, 3, 3, 4, 4};
    int64_t input1[3] = {2, 2, 4};
    int64_t output[16] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int64_t expect_out[16] = {1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4};
    EXPECT_EQ(CompareResult<int64_t>(output, expect_out, 16), true);
}

TEST_F(TEST_EXPAND_UT, TestExpandCheck) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3, 1}, {3}, {2, 3, 6}}; 
    int32_t input0[12] = {1, 2, 3};
    int32_t input1[3] = {2, 3, 6};
    int32_t output[36] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
    CREATE_EXPAND_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    int32_t expect_out[36] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3};
    EXPECT_EQ(CompareResult<int32_t>(output, expect_out, 36), true);
}