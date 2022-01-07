/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#undef private
#undef protected
#include "Eigen/SVD"

using namespace std;
using namespace aicpu;

class TEST_SVD_UT : public testing::Test {};
#define CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv) \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();           \
  NodeDefBuilder(node_def.get(), "Svd", "Svd")                               \
      .Input({"x", data_types[0], shapes[0], datas[0]})                      \
      .Output({"sigma", data_types[1], shapes[1], datas[1]})                 \
      .Output({"u", data_types[2], shapes[2], datas[2]})                     \
      .Output({"v", data_types[3], shapes[3], datas[3]})                     \
      .Attr("full_matrices", full_matrices)                                  \
      .Attr("compute_uv", compute_uv)

TEST_F(TEST_SVD_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{6}, {6}, {6}, {6}};
  float x[6] = {(float)1};
  float s[6] = {(float)1};
  float u[6] = {(float)1};
  float v[6] = {(float)1};
  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SVD_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2}, {2, 2}, {11, 2}};
  float s[2] = {(float)0};
  float u[2 * 11] = {(float)0};
  float v[2 * 11] = {(float)0};
  vector<void *> datas = {(void *)nullptr, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SVD_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT8, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2}, {2, 2}, {11, 2}};
  int8_t x[2 * 11] = {(int8_t)1};
  float s[2] = {(float)1};
  float u[2 * 2] = {(float)1};
  float v[11 * 2] = {(float)1};
  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SVD_UT, DATA_TYPE_FLOAT_SUCC_2D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4}, {3}, {3, 3}, {4, 3}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "svd/data/svd_data_input1_1.txt";
  constexpr uint64_t x_size = 3 * 4;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t s_size = 3;
  float s[s_size] = {0};
  constexpr uint64_t u_size = 3 * 3;
  float u[u_size] = {0};
  constexpr uint64_t v_size = 4 * 3;
  float v[v_size] = {0};
  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "svd/data/svd_data_output1_1.txt";
  float s_exp[s_size] = {0};
  status = ReadFile(data_path, s_exp, s_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output2_1.txt";
  float u_exp[u_size] = {0};
  status = ReadFile(data_path, u_exp, u_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output3_1.txt";
  float v_exp[v_size] = {0};
  status = ReadFile(data_path, v_exp, v_size);
  EXPECT_EQ(status, true);

  bool compare_s = CompareResult(s, s_exp, s_size);
  bool compare_u = CompareResult(u, u_exp, u_size);
  bool compare_v = CompareResult(v, v_exp, v_size);
  EXPECT_EQ(compare_s && compare_u && compare_v, true);
}

TEST_F(TEST_SVD_UT, DATA_TYPE_FLOAT_SUCC_3D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3}, {2, 3, 3}, {2, 4, 3}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "svd/data/svd_data_input1_2.txt";
  constexpr uint64_t x_size = 2 * 3 * 4;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t s_size = 2 * 3;
  float s[s_size] = {0};
  constexpr uint64_t u_size = 2 * 3 * 3;
  float u[u_size] = {0};
  constexpr uint64_t v_size = 2 * 4 * 3;
  float v[v_size] = {0};

  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "svd/data/svd_data_output1_2.txt";
  float s_exp[s_size] = {0};
  status = ReadFile(data_path, s_exp, s_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output2_2.txt";
  float u_exp[u_size] = {0};
  status = ReadFile(data_path, u_exp, u_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output3_2.txt";
  float v_exp[v_size] = {0};
  status = ReadFile(data_path, v_exp, v_size);
  EXPECT_EQ(status, true);

  bool compare_s = CompareResult(s, s_exp, s_size);
  bool compare_u = CompareResult(u, u_exp, u_size);
  bool compare_v = CompareResult(v, v_exp, v_size);
  EXPECT_EQ(compare_s && compare_u && compare_v, true);
}

TEST_F(TEST_SVD_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3}, {2, 3, 3}, {2, 4, 3}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "svd/data/svd_data_input1_3.txt";
  constexpr uint64_t x_size = 2 * 3 * 4;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t s_size = 2 * 3;
  float s[s_size] = {0};
  constexpr uint64_t u_size = 2 * 3 * 3;
  float u[u_size] = {0};
  constexpr uint64_t v_size = 2 * 4 * 3;
  float v[v_size] = {0};

  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "svd/data/svd_data_output1_3.txt";
  float s_exp[s_size] = {0};
  status = ReadFile(data_path, s_exp, s_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output2_3.txt";
  float u_exp[u_size] = {0};
  status = ReadFile(data_path, u_exp, u_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output3_3.txt";
  float v_exp[v_size] = {0};
  status = ReadFile(data_path, v_exp, v_size);
  EXPECT_EQ(status, true);

  bool compare_s = CompareResult(s, s_exp, s_size);
  bool compare_u = CompareResult(u, u_exp, u_size);
  bool compare_v = CompareResult(v, v_exp, v_size);
  EXPECT_EQ(compare_s && compare_u && compare_v, true);
}

TEST_F(TEST_SVD_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4, 4, 16}, {4, 4}, {4, 4, 4}, {4, 16, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "svd/data/svd_data_input1_4.txt";
  constexpr uint64_t x_size = 4 * 4 * 16;
  double x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t s_size = 4 * 4;
  double s[s_size] = {0};
  constexpr uint64_t u_size = 4 * 4 * 4;
  double u[u_size] = {0};
  constexpr uint64_t v_size = 4 * 16 * 4;
  double v[v_size] = {0};

  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "svd/data/svd_data_output1_4.txt";
  double s_exp[s_size] = {0};
  status = ReadFile(data_path, s_exp, s_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output2_4.txt";
  double u_exp[u_size] = {0};
  status = ReadFile(data_path, u_exp, u_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output3_4.txt";
  double v_exp[v_size] = {0};
  status = ReadFile(data_path, v_exp, v_size);
  EXPECT_EQ(status, true);

  bool compare_s = CompareResult(s, s_exp, s_size);
  bool compare_u = CompareResult(u, u_exp, u_size);
  bool compare_v = CompareResult(v, v_exp, v_size);
  EXPECT_EQ(compare_s && compare_u && compare_v, true);
}

TEST_F(TEST_SVD_UT, DATA_TYPE_FLOAT_FULL_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 4, 3}, {2, 3}, {2, 4, 4}, {2, 3, 3}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "svd/data/svd_data_input1_5.txt";
  constexpr uint64_t x_size = 2 * 4 * 3;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t s_size = 2 * 3;
  float s[s_size] = {0};
  constexpr uint64_t u_size = 2 * 4 * 4;
  float u[u_size] = {0};
  constexpr uint64_t v_size = 2 * 3 * 3;
  float v[v_size] = {0};

  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = true;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "svd/data/svd_data_output1_5.txt";
  float s_exp[s_size] = {0};
  status = ReadFile(data_path, s_exp, s_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output2_5.txt";
  float u_exp[u_size] = {0};
  status = ReadFile(data_path, u_exp, u_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output3_5.txt";
  float v_exp[v_size] = {0};
  status = ReadFile(data_path, v_exp, v_size);
  EXPECT_EQ(status, true);

  bool compare_s = CompareResult(s, s_exp, s_size);
  bool compare_u = CompareResult(u, u_exp, u_size);
  bool compare_v = CompareResult(v, v_exp, v_size);
  EXPECT_EQ(compare_s && compare_u && compare_v, true);
}

TEST_F(TEST_SVD_UT, DATA_TYPE_FLOAT_SUCC_2D_ONLY_S_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4}, {3}, {3, 3}, {4, 3}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "svd/data/svd_data_input1_6.txt";
  constexpr uint64_t x_size = 3 * 4;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);
  constexpr uint64_t s_size = 3;
  float s[s_size] = {0};
  constexpr uint64_t u_size = 3 * 3;
  float u[u_size] = {0};
  constexpr uint64_t v_size = 4 * 3;
  float v[v_size] = {0};
  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "svd/data/svd_data_output1_6.txt";
  float s_exp[s_size] = {0};
  status = ReadFile(data_path, s_exp, s_size);
  EXPECT_EQ(status, true);
  bool compare_s = CompareResult(s, s_exp, s_size);
  EXPECT_EQ(compare_s, true);
}

TEST_F(TEST_SVD_UT, DATA_TYPE_DOUBLE_16K_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 32, 32}, {2, 32}, {2, 32, 32}, {2, 32, 32}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "svd/data/svd_data_input1_7.txt";
  constexpr uint64_t x_size = 2 * 32 * 32;
  double x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t s_size = 2 * 32;
  double s[s_size] = {0};
  constexpr uint64_t u_size = 2 * 32 * 32;
  double u[u_size] = {0};
  constexpr uint64_t v_size = 2 * 32 * 32;
  double v[v_size] = {0};

  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = false;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "svd/data/svd_data_output1_7.txt";
  double s_exp[s_size] = {0};
  status = ReadFile(data_path, s_exp, s_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output2_7.txt";
  double u_exp[u_size] = {0};
  status = ReadFile(data_path, u_exp, u_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output3_7.txt";
  double v_exp[v_size] = {0};
  status = ReadFile(data_path, v_exp, v_size);
  EXPECT_EQ(status, true);

  bool compare_s = CompareResult(s, s_exp, s_size);
  bool compare_u = CompareResult(u, u_exp, u_size);
  bool compare_v = CompareResult(v, v_exp, v_size);
  EXPECT_EQ(compare_s && compare_u && compare_v, true);
}

TEST_F(TEST_SVD_UT, DATA_TYPE_DOUBLE_FULL_32K_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 32, 64}, {2, 32}, {2, 32, 32}, {2, 64, 64}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "svd/data/svd_data_input1_8.txt";
  constexpr uint64_t x_size = 2 * 32 * 64;
  double x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t s_size = 2 * 32;
  double s[s_size] = {0};
  constexpr uint64_t u_size = 2 * 32 * 32;
  double u[u_size] = {0};
  constexpr uint64_t v_size = 2 * 64 * 64;
  double v[v_size] = {0};

  vector<void *> datas = {(void *)x, (void *)s, (void *)u, (void *)v};
  bool full_matrices = true;
  bool compute_uv = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices, compute_uv);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "svd/data/svd_data_output1_8.txt";
  double s_exp[s_size] = {0};
  status = ReadFile(data_path, s_exp, s_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output2_8.txt";
  double u_exp[u_size] = {0};
  status = ReadFile(data_path, u_exp, u_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "svd/data/svd_data_output3_8.txt";
  double v_exp[v_size] = {0};
  status = ReadFile(data_path, v_exp, v_size);
  EXPECT_EQ(status, true);

  bool compare_s = CompareResult(s, s_exp, s_size);
  bool compare_u = CompareResult(u, u_exp, u_size);
  bool compare_v = CompareResult(v, v_exp, v_size);
  EXPECT_EQ(compare_s && compare_u && compare_v, true);
}