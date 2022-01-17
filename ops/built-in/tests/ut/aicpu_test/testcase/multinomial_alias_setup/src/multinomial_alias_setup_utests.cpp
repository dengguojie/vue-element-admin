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

using namespace std;
using namespace aicpu;


class TEST_MULTINOMIAL_ALIAS_SETUP_UT : public testing::Test {};


#define CREATE_NODEDEF(shapes, data_types, datas)                                                   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                                  \
  NodeDefBuilder(node_def.get(), "MultinomialAliasSetup", "MultinomialAliasSetup")                  \
      .Input({"probs", data_types[0], shapes[0], datas[0]})                                         \
      .Output({"j", data_types[1], shapes[1], datas[1]})                                            \
      .Output({"q", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3>
void RunMultinomialAliasSetupKernel(vector<string> data_files,
                   vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);
  uint64_t output_j_size = CalTotalElements(shapes, 1);
  T2 *output_j = new T2[output_j_size];
  uint64_t output_q_size = CalTotalElements(shapes, 2);
  T3 *output_q = new T3[output_q_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output_j,
                          (void *)output_q};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  string data_j_path = ktestcaseFilePath + data_files[1];
  T2 *output_j_exp = new T2[output_j_size];
  status = ReadFile(data_j_path, output_j_exp, output_j_size);
  EXPECT_EQ(status, true);
  string data_q_path = ktestcaseFilePath + data_files[2];
  T3 *output_q_exp = new T3[output_q_size];
  status = ReadFile(data_q_path, output_q_exp, output_q_size);
  EXPECT_EQ(status, true);
  bool compare = CompareResult(output_j, output_j_exp, output_j_size);
  EXPECT_EQ(compare, true);
  compare = CompareResult(output_q, output_q_exp, output_q_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] output_j;
  delete [] output_q;
  delete [] output_j_exp;
  delete [] output_q_exp;
}

TEST_F(TEST_MULTINOMIAL_ALIAS_SETUP_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1024}, {1024}, {1024}};
  vector<string> files{"multinomial_alias_setup/data/multinomial_alias_setup_data_input1_1.txt",
                       "multinomial_alias_setup/data/multinomial_alias_setup_data_output1_1.txt",
                       "multinomial_alias_setup/data/multinomial_alias_setup_data_output2_1.txt"};
  RunMultinomialAliasSetupKernel<float, int64_t, float>(files, data_types, shapes);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_SETUP_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{512}, {512}, {512}};
  vector<string> files{"multinomial_alias_setup/data/multinomial_alias_setup_data_input1_2.txt",
                       "multinomial_alias_setup/data/multinomial_alias_setup_data_output1_2.txt",
                       "multinomial_alias_setup/data/multinomial_alias_setup_data_output2_2.txt"};
  RunMultinomialAliasSetupKernel<double, int64_t, double>(files, data_types, shapes);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_SETUP_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_BOOL,  DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{6}, {6}, {6}};
  bool input[6] = {(bool)1};
  int64_t output1[6] = {(int64_t)1};
  float output2[6] = {(float)1};
  vector<void *> datas = {(void *)input, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_SETUP_UT, INPUT_VALUE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT,  DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3}, {3}, {3}};
  float input[10] = {-1, -1, 0};
  int64_t output1[10] = {(int64_t)1};
  float output2[10] = {(float)1};
  vector<void *> datas = {(void *)input, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MULTINOMIAL_ALIAS_SETUP_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT,  DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2}, {4}, {4}};
  float input[4] = {1, 1, 1, 0};
  int64_t output1[4] = {(int64_t)1};
  float output2[4] = {(float)1};
  vector<void *> datas = {(void *)input, (void *)output1, (void *)output2};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
