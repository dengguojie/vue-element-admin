#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_SUB_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Sub", "Sub")                     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3>
void RunSubKernel(vector<string> data_files,
               vector<DataType> data_types,
               vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3 *output_exp = new T3[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_SUB_UT, FLOAT_4_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_1.txt",
                       "sub/data/sub_data_input2_1.txt",
                       "sub/data/sub_data_output1_1.txt"};
  RunSubKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, BROADCAST_2_4_4_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2,4}, {4}, {2,4}};
  vector<string> files{"sub/data/sub_data_input1_2.txt",
                       "sub/data/sub_data_input2_2.txt",
                       "sub/data/sub_data_output1_2.txt"};
  RunSubKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, X_SCALAR_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_3.txt",
                       "sub/data/sub_data_input2_3.txt",
                       "sub/data/sub_data_output1_3.txt"};
  RunSubKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, Y_SCALAR_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{4}, {}, {4}};
  vector<string> files{"sub/data/sub_data_input1_4.txt",
                       "sub/data/sub_data_input2_4.txt",
                       "sub/data/sub_data_output1_4.txt"};
  RunSubKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, TWO_SCALAR_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{}, {}, {}};
  vector<string> files{"sub/data/sub_data_input1_5.txt",
                       "sub/data/sub_data_input2_5.txt",
                       "sub/data/sub_data_output1_5.txt"};
  RunSubKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, Double_SUCCESS) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_6.txt",
                       "sub/data/sub_data_input2_6.txt",
                       "sub/data/sub_data_output1_6.txt"};
  RunSubKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, Float16_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_7.txt",
                       "sub/data/sub_data_input2_7.txt",
                       "sub/data/sub_data_output1_7.txt"};
  RunSubKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, Uint8_SUCCESS) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_8.txt",
                       "sub/data/sub_data_input2_8.txt",
                       "sub/data/sub_data_output1_8.txt"};
  RunSubKernel<uint8_t, uint8_t, uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, Int8_SUCCESS) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_9.txt",
                       "sub/data/sub_data_input2_9.txt",
                       "sub/data/sub_data_output1_9.txt"};
  RunSubKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, Int16_SUCCESS) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_10.txt",
                       "sub/data/sub_data_input2_10.txt",
                       "sub/data/sub_data_output1_10.txt"};
  RunSubKernel<int16_t, int16_t, int16_t>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, Uint16_SUCCESS) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_11.txt",
                       "sub/data/sub_data_input2_11.txt",
                       "sub/data/sub_data_output1_11.txt"};
  RunSubKernel<uint16_t, uint16_t, uint16_t>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, int32_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_12.txt",
                       "sub/data/sub_data_input2_12.txt",
                       "sub/data/sub_data_output1_12.txt"};
  RunSubKernel<int32_t, int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_SUB_UT, Uint32_Failed) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  uint32_t input0[8] = {5, 4, 6, 7};
  uint32_t input1[4] = {5, 2, 5, 10};
  uint32_t output[40] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SUB_UT, int64_SUCCESS) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{4}, {4}, {4}};
  vector<string> files{"sub/data/sub_data_input1_13.txt",
                       "sub/data/sub_data_input2_13.txt",
                       "sub/data/sub_data_output1_13.txt"};
  RunSubKernel<int64_t, int64_t, int64_t>(files, data_types, shapes);
}
