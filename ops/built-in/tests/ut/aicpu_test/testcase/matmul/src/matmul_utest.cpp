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

class TEST_MATMUL_UT : public testing::Test {};

#define CREATE_NODEDEF1(shapes, data_types, datas)                 \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "MatMul", "MatMul")               \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})           \
      .Attr("transpose_x1", false)                                 \
      .Attr("transpose_x2", false)

#define CREATE_NODEDEF2(shapes, data_types, datas)                 \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "MatMul", "MatMul")               \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})           \
      .Attr("transpose_x1", true)                                  \
      .Attr("transpose_x2", false)

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3>
void RunMatMulKernel(vector<string> data_files,
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

  CREATE_NODEDEF1(shapes, data_types, datas);
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

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3>
void RunMatMulKernelV2(vector<string> data_files,
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

  CREATE_NODEDEF2(shapes, data_types, datas);
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

TEST_F(TEST_MATMUL_UT, Not_Transpose_x1_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3}, {3, 2}, {2, 2}};
  vector<string> files{"matmul/data/matmul_data_input1_1.txt",
                       "matmul/data/matmul_data_input2_1.txt",
                       "matmul/data/matmul_data_output1_1.txt"};
  RunMatMulKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_MATMUL_UT, Transpose_x1_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 2}, {3, 2}, {2, 2}};
  vector<string> files{"matmul/data/matmul_data_input1_2.txt",
                       "matmul/data/matmul_data_input2_2.txt",
                       "matmul/data/matmul_data_output1_2.txt"};
  RunMatMulKernelV2<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_MATMUL_UT, Double_SUCCESS) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 3}, {3, 2}, {2, 2}};
  vector<string> files{"matmul/data/matmul_data_input1_3.txt",
                       "matmul/data/matmul_data_input2_3.txt",
                       "matmul/data/matmul_data_output1_3.txt"};
  RunMatMulKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_MATMUL_UT, Int32_SUCCESS) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {3, 2}, {2, 2}};
  vector<string> files{"matmul/data/matmul_data_input1_4.txt",
                       "matmul/data/matmul_data_input2_4.txt",
                       "matmul/data/matmul_data_output1_4.txt"};
  RunMatMulKernel<int32_t, int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_MATMUL_UT, Float16_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2, 3}, {3, 2}, {2, 2}};
  vector<string> files{"matmul/data/matmul_data_input1_5.txt",
                       "matmul/data/matmul_data_input2_5.txt",
                       "matmul/data/matmul_data_output1_5.txt"};
  RunMatMulKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_MATMUL_UT, Uint32_Failed) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {3, 2}, {2, 2}};
  uint32_t input0[6] = {1, 2, 3, 4, 5, 6};
  uint32_t input1[6] = {7, 8, 9, 10, 11, 12};
  uint32_t output[40] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};
  CREATE_NODEDEF1(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}