/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 */

#include <gtest/gtest.h>
#ifndef private
#define private public
#define protected public
#endif
#include <Eigen/Core>
#include <iostream>

#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_LINSPACE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)     \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "LinSpace", "LinSpace")   \
      .Input({"start", data_types[0], shapes[0], datas[0]})        \
      .Input({"stop", data_types[1], shapes[1], datas[1]})        \
      .Input({"num", data_types[2], shapes[2], datas[2]})        \
      .Output({"output", data_types[3], shapes[3], datas[3]})           \

template <typename T>
bool CheckResult(T *output_data, T *output_exp, int64_t output_num) {
    for(int64_t i = 0; i < output_num; i++) {
       if(output_data[i] != output_exp[i]) {
        return false;
        }
    }
    return true;
}

template <typename T, typename NUMT>
void RunTestLinSpace(string test_case_no, vector<DataType> data_types) {
  // read data from file for start
  string start_path = ktestcaseFilePath + "linspace/data/linspace_data_start" + test_case_no + ".txt";
  T start[1] = {0};
  EXPECT_EQ(ReadFile(start_path, start, 1), true);

  // read data from file for stop
  string stop_path = ktestcaseFilePath + "linspace/data/linspace_data_stop" + test_case_no + ".txt";
  T stop[1] = {0};
  EXPECT_EQ(ReadFile(stop_path, stop, 1), true);

  // read data from file for num
  string num_path = ktestcaseFilePath + "linspace/data/linspace_data_num" + test_case_no + ".txt";
  NUMT num[1] = {0};
  EXPECT_EQ(ReadFile(num_path, num, 1), true);

  // output
  T output_data[num[0]] = {0};

  // 创建node，运行算子
  vector<vector<int64_t>> shapes = {{}, {}, {}, {num[0]}};
  vector<void*> datas = {(void*)start, (void*)stop, (void*)num, (void*)output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK)

  // 对比结果
  string output_data_path = ktestcaseFilePath +  "linspace/data/linspace_data_output" + test_case_no + ".txt";
  T output_exp[num[0]] = {0};
  EXPECT_EQ(ReadFile(output_data_path, output_exp, num[0]), true);

  bool compare = CheckResult<T>(output_data, output_exp, num[0]);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LINSPACE_UT, SUCESS_11) {
  RunTestLinSpace<float, int32_t>("1", {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT});
}

TEST_F(TEST_LINSPACE_UT, SUCESS_21) {
  RunTestLinSpace<double, int64_t>("2", {DT_DOUBLE, DT_DOUBLE, DT_INT64, DT_DOUBLE});
}

// num = 1
TEST_F(TEST_LINSPACE_UT, SUCESS_3) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {1}};
  float start_data = 2;
  float stop_data = 10;
  int32_t num_data = 1;
  float output_data[1] = {0};
  float output_exp[1] = {2};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)&output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK)

  bool compare = CheckResult<float>(output_data, output_exp, 1);
  EXPECT_EQ(compare, true);
}

// num < 0
TEST_F(TEST_LINSPACE_UT, FAILED_1) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  float start_data = 2;
  float stop_data = 10;
  int64_t num_data = -1;
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)&output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// start 非 scalar
TEST_F(TEST_LINSPACE_UT, FAILED_2) {
  vector<vector<int64_t>> shapes = {{2}, {}, {}, {}};
  float start_data[2] = {2, 10};
  float stop_data = 10;
  int64_t num_data = 5;
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)&output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// stop 非 scalar
TEST_F(TEST_LINSPACE_UT, FAILED_3) {
  vector<vector<int64_t>> shapes = {{}, {2}, {}, {}};
  float start_data = 2;
  float stop_data[2] = {10, 2};
  int64_t num_data = 5;
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)&output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// num 非 scalar
TEST_F(TEST_LINSPACE_UT, FAILED_4) {
  vector<vector<int64_t>> shapes = {{}, {}, {2}, {}};
  float start_data = 2;
  float stop_data = 10;
  int64_t num_data[2] = {2, 5};
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)&output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// start 和 stop 数据类型不同
TEST_F(TEST_LINSPACE_UT, FAILED_5) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  double start_data = 2;
  float stop_data = 10;
  int64_t num_data = 5;
  float output_data[1] = {0};

  vector<DataType> data_types = {DT_DOUBLE, DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)&output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// start 和 stop 输入类型非float double
TEST_F(TEST_LINSPACE_UT, FAILED_6) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  int64_t start_data = 2;
  int64_t stop_data = 10;
  int64_t num_data = 5;
  int64_t output_data[1] = {0};

  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64, DT_INT64};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)&output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}

// num 非int32 int64
TEST_F(TEST_LINSPACE_UT, FAILED_7) {
  vector<vector<int64_t>> shapes = {{}, {}, {}, {5}};
  double start_data = 10;
  double stop_data = 2;
  int8_t num_data = 5;
  double output_data[5] = {0};
  double output_exp[5] = {10, 8, 6, 4, 2};

  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT8, DT_DOUBLE};
  vector<void*> datas = {(void*)&start_data, (void*)&stop_data, (void*)&num_data, (void*)&output_data};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID)
}