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
#include <complex>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_Cross_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, dim_select)      \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Cross", "Cross")                 \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})           \
      .Attr("dim", dim_select)

#define CREATE_NODEDEF_WITHOUT_DIM(shapes, data_types, datas)      \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Cross", "Cross")                 \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})           

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunCrossKernel1(vector<string> data_files, vector<DataType> data_types,
                       vector<vector<int64_t>>& shapes, int64_t dim_dim) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1* input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2* input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3* output = new T3[output_size];
  vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

  CREATE_NODEDEF(shapes, data_types, datas, dim_dim);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3* output_exp = new T3[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

template <typename T1, typename T2, typename T3>
void RunCrossKernel2(vector<string> data_files, vector<DataType> data_types,
                       vector<vector<int64_t>>& shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1* input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2* input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3* output = new T3[output_size];
  vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

  CREATE_NODEDEF_WITHOUT_DIM(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3* output_exp = new T3[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{2, 3, 3}, {2, 3, 3}, {2, 3, 3}};
  vector<string> files{"cross/data/Cross_data_input1_1.txt",
                       "cross/data/Cross_data_input2_1.txt",
                       "cross/data/Cross_data_output1_1.txt"};
  int64_t ddim = 1;
  RunCrossKernel1<int16_t, int16_t, int16_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 3, 3}, {2, 3, 3}, {2, 3, 3}};
  vector<string> files{"cross/data/Cross_data_input1_2.txt",
                       "cross/data/Cross_data_input2_2.txt",
                       "cross/data/Cross_data_output1_2.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<int64_t, int64_t, int64_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_INT64_SUCC1) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 3, 3, 3}, {2, 3, 3, 3}, {2, 3, 3, 3}};
  vector<string> files{"cross/data/Cross_data_input1_3.txt",
                       "cross/data/Cross_data_input2_3.txt",
                       "cross/data/Cross_data_output1_3.txt"};
  int64_t ddim = 1;
  RunCrossKernel1<int64_t, int64_t, int64_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3, 2, 3}, {3, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_4.txt",
                       "cross/data/Cross_data_input2_4.txt",
                       "cross/data/Cross_data_output1_4.txt"};
  int64_t ddim = 0;
  RunCrossKernel1<int8_t, int8_t, int8_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3, 2, 3}, {3, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_5.txt",
                       "cross/data/Cross_data_input2_5.txt",
                       "cross/data/Cross_data_output1_5.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<int32_t, int32_t, int32_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_6.txt",
                       "cross/data/Cross_data_input2_6.txt",
                       "cross/data/Cross_data_output1_6.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<uint8_t, uint8_t, uint8_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_7.txt",
                       "cross/data/Cross_data_input2_7.txt",
                       "cross/data/Cross_data_output1_7.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<uint16_t, uint16_t, uint16_t>(files, data_types, shapes,
                                                  ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_8.txt",
                       "cross/data/Cross_data_input2_8.txt",
                       "cross/data/Cross_data_output1_8.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<uint32_t, uint32_t, uint32_t>(files, data_types, shapes,
                                                  ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_9.txt",
                       "cross/data/Cross_data_input2_9.txt",
                       "cross/data/Cross_data_output1_9.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<uint64_t, uint64_t, uint64_t>(files, data_types, shapes,
                                                  ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_FLOAT64_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_10.txt",
                       "cross/data/Cross_data_input2_10.txt",
                       "cross/data/Cross_data_output1_10.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<float, float, float>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_FLOAT32_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_11.txt",
                       "cross/data/Cross_data_input2_11.txt",
                       "cross/data/Cross_data_output1_11.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<float, float, float>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_INT8_SUCC1) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {
      {3, 2, 2, 3}, {3, 2, 2, 3}, {3, 2, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_12.txt",
                       "cross/data/Cross_data_input2_12.txt",
                       "cross/data/Cross_data_output1_12.txt"};
  int64_t ddim = -1;
  RunCrossKernel1<int8_t, int8_t, int8_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_INT16_SUCC1) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {
      {3, 2, 3, 3}, {3, 2, 3, 3}, {3, 2, 3, 3}};
  vector<string> files{"cross/data/Cross_data_input1_13.txt",
                       "cross/data/Cross_data_input2_13.txt",
                       "cross/data/Cross_data_output1_13.txt"};
  int64_t ddim = -2;
  RunCrossKernel1<int16_t, int16_t, int16_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_INT32_SUCC1) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {
      {3, 3, 2, 3}, {3, 3, 2, 3}, {3, 3, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_14.txt",
                       "cross/data/Cross_data_input2_14.txt",
                       "cross/data/Cross_data_output1_14.txt"};
  int64_t ddim = -3;
  RunCrossKernel1<int32_t, int32_t, int32_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_UINT8_SUCC1) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {
      {3, 2, 2, 3}, {3, 2, 2, 3}, {3, 2, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_15.txt",
                       "cross/data/Cross_data_input2_15.txt",
                       "cross/data/Cross_data_output1_15.txt"};
  int64_t ddim = 0;
  RunCrossKernel1<uint8_t, uint8_t, uint8_t>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_DOUBLE_SUCC1) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3, 2, 3}, {3, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_16.txt",
                       "cross/data/Cross_data_input2_16.txt",
                       "cross/data/Cross_data_output1_16.txt"};
  int64_t ddim = 0;
  RunCrossKernel1<double, double, double>(files, data_types, shapes, ddim);
}

TEST_F(TEST_Cross_UT, DATA_TYPE_FLOAT16_SUCC1) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_19.txt",
                       "cross/data/Cross_data_input2_19.txt",
                       "cross/data/Cross_data_output1_19.txt"};
  int64_t ddim = 2;
  RunCrossKernel1<Eigen::half, Eigen::half, Eigen::half>(files, data_types,
                                                           shapes, ddim);
}

TEST_F(TEST_Cross_UT, DEFAULT_DIM) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{2, 2, 3}, {2, 2, 3}, {2, 2, 3}};
  vector<string> files{"cross/data/Cross_data_input1_20.txt",
                       "cross/data/Cross_data_input2_20.txt",
                       "cross/data/Cross_data_output1_20.txt"};
  RunCrossKernel2<uint8_t, uint8_t, uint8_t>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_Cross_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  int32_t output[6];
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  int64_t ddim = 2;
  CREATE_NODEDEF(shapes, data_types, datas, ddim);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, DIM_RANGE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  int32_t input1[6];
  int32_t input2[6];
  int32_t output[6];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  int64_t ddim = 3;
  CREATE_NODEDEF(shapes, data_types, datas, ddim);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, DIM_VALUE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};
  int32_t input1[6];
  int32_t input2[6];
  int32_t output[6];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  int64_t ddim = 0;
  CREATE_NODEDEF(shapes, data_types, datas, ddim);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, SHAPE_SIZE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 2, 3, 4}, {1, 2, 3}};
  int32_t input1[6];
  int32_t input2[24];
  int32_t output[6];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  int64_t ddim = 2;
  CREATE_NODEDEF(shapes, data_types, datas, ddim);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Cross_UT, SHAPE_VALUE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1, 2, 3}, {1, 3, 3}, {1, 2, 3}};
  int32_t input1[6];
  int32_t input2[9];
  int32_t output[6];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  int64_t ddim = 2;
  CREATE_NODEDEF(shapes, data_types, datas, ddim);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}