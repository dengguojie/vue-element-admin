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
#include "Eigen/Core"

#include <string>
using namespace std;
using namespace aicpu;


class TEST_MULTINOMIAL_ALIAS_DRAW_UT : public testing::Test {};


#define CREATE_NODEDEF(shapes, data_types, datas, num_samples, seed)                                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                                    \
  if(seed == -1){                                                                                     \
    NodeDefBuilder(node_def.get(), "MultinomialAliasDraw", "MultinomialAliasDraw")                    \
        .Input({"q", data_types[0], shapes[0], datas[0]})                                             \
        .Input({"j", data_types[1], shapes[1], datas[1]})                                             \
        .Output({"y", data_types[2], shapes[0], datas[2]})                                            \
        .Attr("num_samples", num_samples);                                                            \
    } else {                                                                                          \
    NodeDefBuilder(node_def.get(), "MultinomialAliasDraw", "MultinomialAliasDraw")                    \
        .Input({"q", data_types[0], shapes[0], datas[0]})                                             \
        .Input({"j", data_types[1], shapes[1], datas[1]})                                             \
        .Output({"y", data_types[2], shapes[0], datas[2]})                                            \
        .Attr("num_samples", num_samples)                                                             \
        .Attr("seed", seed);                                                                          \
  }


TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {4}, {10}};
  float input_q[4] = {1, 1, 1, 0};
  int64_t input_j[4] = {(int64_t)1};
  int64_t output[10] = {(int64_t)2};
  int64_t num_samples = 10;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, -1);
  RUN_KERNEL(node_def, HOST, 0);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, ATTR_SEED_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{7}, {7}, {5}};
  float input_q[7] = {1, 1, 1, 0, 0.5, 0.3, 0.2};
  int64_t input_j[7] = {(int64_t)1};
  int64_t output[5] = {(int64_t)2};
  int64_t num_samples = 5;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, 5);
  RUN_KERNEL(node_def, HOST, 0);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {4}, {10}};
  double input_q[4] = {1, 1, -0.5, 0.5};
  int64_t input_j[4] = {(int64_t)1};
  int64_t output[10] = {(int64_t)2};
  int64_t num_samples = 10;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, -1);
  RUN_KERNEL(node_def, HOST, 0);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, ATTR_SEED_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{7}, {7}, {5}};
  double input_q[7] = {1, 1, 1, 0, 0.5, 0.3, 0.2};
  int64_t input_j[7] = {(int64_t)1};
  int64_t output[5] = {(int64_t)2};
  int64_t num_samples = 5;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, 3);
  RUN_KERNEL(node_def, HOST, 0);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, INPUT_Q_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_BOOL, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{7}, {7}, {5}};
  bool input_q[7] = {(bool)1};
  int64_t input_j[7] = {(int64_t)1};
  int64_t output[5] = {(int64_t)2};
  int64_t num_samples = 5;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, -1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, INPUT_J_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{7}, {7}, {5}};
  double input_q[7] = {1, 1, 1, 0, 0.5, 0.3, 0.2};
  double input_j[7] = {(double)1};
  int64_t output[5] = {(int64_t)2};
  int64_t num_samples = 5;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, 5);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, ATTR_NUM_SAMPLE_VALUE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{7}, {7}, {5}};
  double input_q[7] = {1, 1, 1, 0, 0.5, 0.3, 0.2};
  int64_t input_j[7] = {(int64_t)1};
  int64_t output[5] = {(int64_t)2};
  int64_t num_samples = -1;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, -1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, INPUT_SHAPE_SAME_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{7}, {6}, {5}};
  double input_q[7] = {1, 1, 1, 0, 0.5, 0.3,0.5};
  int64_t input_j[6] = {(int64_t)1};
  int64_t output[5] = {(int64_t)2};
  int64_t num_samples = 5;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, 5);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, INPUT_Q_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT,  DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2}, {4}, {5}};
  double input_q[4] = {1, 1, 1, 0};
  int64_t input_j[4] = {(int64_t)1};
  int64_t output[4] = {(int64_t)2};
  int64_t num_samples = 5;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, 5);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_DRAW_UT, INPUT_J_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT,  DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {2,2}, {5}};
  double input_q[4] = {1, 1, 1, 0};
  int64_t input_j[4] = {(int64_t)1};
  int64_t output[4] = {(int64_t)2};
  int64_t num_samples = 5;
  vector<void *> datas = {(void *)input_q, (void *)input_j, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, num_samples, 5);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
