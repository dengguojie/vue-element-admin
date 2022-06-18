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

class TEST_ADDCDIV_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Addcdiv", "Addcdiv")             \
      .Input({"input_data", data_types[0], shapes[0], datas[0]})   \
      .Input({"x1", data_types[1], shapes[1], datas[1]})           \
      .Input({"x2", data_types[2], shapes[2], datas[2]})           \
      .Input({"value", data_types[3], shapes[3], datas[3]})        \
      .Output({"y", data_types[4], shapes[4], datas[4]})           

template<typename T>
void RunAddcdivKernel(vector<string> data_files,
                   vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T *input1 = new T[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T *input2 = new T[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  // read data from file for input3
  data_path = ktestcaseFilePath + data_files[2];
  uint64_t input3_size = CalTotalElements(shapes, 2);
  T *input3 = new T[input3_size];
  status = ReadFile(data_path, input3, input3_size);
  EXPECT_EQ(status, true);
  // read data from file for input4
  data_path = ktestcaseFilePath + data_files[3];
  uint64_t input4_size = CalTotalElements(shapes, 3);
  T *input4 = new T[input4_size];
  status = ReadFile(data_path, input4, input4_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 4);
  T *output = new T[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)input3,
                          (void *)input4,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[4];
  T *output_exp = new T[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] input3;
  delete [] input4;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_FLOAT_BB_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 4, 1}, {4, 1, 4}, {1, 4, 4}, {1}, {4, 4, 4}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_1.txt",
                       "addcdiv/data/addcdiv_data_input2_1.txt",
                       "addcdiv/data/addcdiv_data_input3_1.txt",
                       "addcdiv/data/addcdiv_data_input4_1.txt",
                       "addcdiv/data/addcdiv_data_output1_1.txt"};
  RunAddcdivKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_FLOAT_BS_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 4}, {4, 4}, {4, 4}, {1}, {4, 4}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_2.txt",
                       "addcdiv/data/addcdiv_data_input2_2.txt",
                       "addcdiv/data/addcdiv_data_input3_2.txt",
                       "addcdiv/data/addcdiv_data_input4_2.txt",
                       "addcdiv/data/addcdiv_data_output1_2.txt"};
  RunAddcdivKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_FLOAT_SB_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 4}, {4, 1}, {1, 4}, {1}, {4, 4}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_3.txt",
                       "addcdiv/data/addcdiv_data_input2_3.txt",
                       "addcdiv/data/addcdiv_data_input3_3.txt",
                       "addcdiv/data/addcdiv_data_input4_3.txt",
                       "addcdiv/data/addcdiv_data_output1_3.txt"};
  RunAddcdivKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_FLOAT_SS_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4, 4}, {4, 4}, {4, 4}, {1}, {4, 4}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_4.txt",
                       "addcdiv/data/addcdiv_data_input2_4.txt",
                       "addcdiv/data/addcdiv_data_input3_4.txt",
                       "addcdiv/data/addcdiv_data_input4_4.txt",
                       "addcdiv/data/addcdiv_data_output1_4.txt"};
  RunAddcdivKernel<float>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_DOUBLE_SO_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{16, 1024}, {1}, {1}, {1}, {16, 1024}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_5.txt",
                       "addcdiv/data/addcdiv_data_input2_5.txt",
                       "addcdiv/data/addcdiv_data_input3_5.txt",
                       "addcdiv/data/addcdiv_data_input4_5.txt",
                       "addcdiv/data/addcdiv_data_output1_5.txt"};
  RunAddcdivKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_DOUBLE_OS_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1}, {16, 1024}, {16, 1024}, {1}, {16, 1024}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_6.txt",
                       "addcdiv/data/addcdiv_data_input2_6.txt",
                       "addcdiv/data/addcdiv_data_input3_6.txt",
                       "addcdiv/data/addcdiv_data_input4_6.txt",
                       "addcdiv/data/addcdiv_data_output1_6.txt"};
  RunAddcdivKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_DOUBLE_BO_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{16, 1024}, {1}, {16, 1}, {1}, {16, 1024}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_7.txt",
                       "addcdiv/data/addcdiv_data_input2_7.txt",
                       "addcdiv/data/addcdiv_data_input3_7.txt",
                       "addcdiv/data/addcdiv_data_input4_7.txt",
                       "addcdiv/data/addcdiv_data_output1_7.txt"};
  RunAddcdivKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_DOUBLE_DIVX_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1}, {1}, {1,2}, {1}, {1,2}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_8.txt",
                       "addcdiv/data/addcdiv_data_input2_8.txt",
                       "addcdiv/data/addcdiv_data_input3_8.txt",
                       "addcdiv/data/addcdiv_data_input4_8.txt",
                       "addcdiv/data/addcdiv_data_output1_8.txt"};
  RunAddcdivKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_DOUBLE_DIVY_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1}, {1,2}, {1}, {1}, {1,2}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_9.txt",
                       "addcdiv/data/addcdiv_data_input2_9.txt",
                       "addcdiv/data/addcdiv_data_input3_9.txt",
                       "addcdiv/data/addcdiv_data_input4_9.txt",
                       "addcdiv/data/addcdiv_data_output1_9.txt"};
  RunAddcdivKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_DOUBLE_ADDX_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1}, {1,3}, {1,3}, {1}, {1,3}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_10.txt",
                       "addcdiv/data/addcdiv_data_input2_10.txt",
                       "addcdiv/data/addcdiv_data_input3_10.txt",
                       "addcdiv/data/addcdiv_data_input4_10.txt",
                       "addcdiv/data/addcdiv_data_output1_10.txt"};
  RunAddcdivKernel<double>(files, data_types, shapes);
}

TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_DOUBLE_ADDY_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1,3}, {1}, {1}, {1}, {1,3}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_11.txt",
                  "addcdiv/data/addcdiv_data_input2_11.txt",
                  "addcdiv/data/addcdiv_data_input3_11.txt",
                  "addcdiv/data/addcdiv_data_input4_11.txt",
                  "addcdiv/data/addcdiv_data_output1_11.txt"};
  RunAddcdivKernel<double>(files, data_types, shapes);
}
TEST_F(TEST_ADDCDIV_UT, DATA_TYPE_FLOAT16_OB_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1}, {16, 1024}, {1, 1024}, {1}, {16, 1024}};
  vector<string> files{"addcdiv/data/addcdiv_data_input1_12.txt",
                  "addcdiv/data/addcdiv_data_input2_12.txt",
                  "addcdiv/data/addcdiv_data_input3_12.txt",
                  "addcdiv/data/addcdiv_data_input4_12.txt",
                  "addcdiv/data/addcdiv_data_output1_12.txt"};
  RunAddcdivKernel<Eigen::half>(files, data_types, shapes);
}


// // exception instance
TEST_F(TEST_ADDCDIV_UT, ADD_BCAST_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2, 3}, {2, 2, 2}, {2, 2, 2}, {1}, {2, 2, 3}};
  float input1[12] = {8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f};
  float input2[8] = {8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f};
  float input3[8] = {8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f};
  float input4[1] = {8.0f};
  float output[12] = {8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)input4, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADDCDIV_UT, DIV_BCAST_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2,2,2}, {2,2,2}, {2,2,3}, {1}, {2, 2,3}};
  float input1[8] = {8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f};
  float input2[8] = {8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f};
  float input3[12] = {8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f};
  float input4[1] = {8.0f};
  float output[12] = {8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f,8.0f};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)input4, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADDCDIV_UT, INPUT_DIV_ZEROD_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2,2}, {2,2}, {2,2}, {1}, {2, 2}};
  double input1[4] = {(double)0.0d};
  double input2[4] = {(double)0.0d};
  double input3[4] = {(double)0.0d};
  double input4[1] = {(double)0.0d};
  double output[4] = {(double)0.0d};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)input4, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADDCDIV_UT, INPUT_DIV_ZEROF_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2,2}, {2,2}, {2,2}, {1}, {2, 2}};
  float input1[4] = {(float)0.0f};
  float input2[4] = {(float)0.0f};
  float input3[4] = {(float)0.0f};
  float input4[1] = {(float)0.0f};
  float output[4] = {(float)0.0f};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)input4, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADDCDIV_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}, {1}, {2, 11}};
  float input1[22] = {(float)1.0f};
  double input2[22] = {(double)1.0d};
  float input3[22] = {(float)1.0f};
  float input4[1] = {(float)1.0f};
  float output[22] = {(float)1.0f};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADDCDIV_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}, {1}, {2, 11}};
  float output[22] = {(float)1.0f};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADDCDIV_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL, DT_BOOL,};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}, {1}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool input3[22] = {(bool)0};
  bool input4[1] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3, (void *)input4, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

