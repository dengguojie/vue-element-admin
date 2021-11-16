#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_FLOORDIV_UT : public testing::Test {};

template <typename T>
void CalcExpectWithSameShape(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  DataType datatype = input1->GetDataType();
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num == input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      expect_out[j] = Eigen::numext::floor(input0_data[j] / input1_data[j]);
    }
  }
}

template <typename T>
void CalcExpectWithDiffShape(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  DataType datatype = input1->GetDataType();
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num > input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      int64_t i = j % input1_num;
      expect_out[j] = Eigen::numext::floor(input0_data[j] / input1_data[i]);
    }
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "FloorDiv", "FloorDiv")           \      
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunFloorDivKernel(vector<string> data_files, vector<DataType> data_types,
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
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3 *output_exp = new T3[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input1;
  delete[] input2;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}};
  vector<string> files{"floordiv/data/floordiv_data_input1_1.txt",
                       "floordiv/data/floordiv_data_input2_1.txt",
                       "floordiv/data/floordiv_data_output1_1.txt"};
  RunFloorDivKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{4}, {3, 4}, {3, 4}};
  vector<string> files{"floordiv/data/floordiv_data_input1_2.txt",
                       "floordiv/data/floordiv_data_input2_2.txt",
                       "floordiv/data/floordiv_data_output1_2.txt"};
  RunFloorDivKernel<int16_t, int16_t, int16_t>(files, data_types, shapes);
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4}, {4}, {3, 4}};
  vector<string> files{"floordiv/data/floordiv_data_input1_3.txt",
                       "floordiv/data/floordiv_data_input2_3.txt",
                       "floordiv/data/floordiv_data_output1_3.txt"};
  RunFloorDivKernel<int32_t, int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}};
  vector<string> files{"floordiv/data/floordiv_data_input1_4.txt",
                       "floordiv/data/floordiv_data_input2_4.txt",
                       "floordiv/data/floordiv_data_output1_4.txt"};
  RunFloorDivKernel<int64_t, int64_t, int64_t>(files, data_types, shapes);
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"floordiv/data/floordiv_data_input1_5.txt",
                       "floordiv/data/floordiv_data_input2_5.txt",
                       "floordiv/data/floordiv_data_output1_5.txt"};
  RunFloorDivKernel<uint8_t, uint8_t, uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{1}, {3, 4, 5}, {3, 4, 5}};
  vector<string> files{"floordiv/data/floordiv_data_input1_6.txt",
                       "floordiv/data/floordiv_data_input2_6.txt",
                       "floordiv/data/floordiv_data_output1_6.txt"};
  RunFloorDivKernel<uint16_t, uint16_t, uint16_t>(files, data_types, shapes);
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1}, {3, 4, 5}, {3, 4, 5}};
  vector<string> files{"floordiv/data/floordiv_data_input1_7.txt",
                       "floordiv/data/floordiv_data_input2_7.txt",
                       "floordiv/data/floordiv_data_output1_7.txt"};
  RunFloorDivKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types,
                                                           shapes);
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1}, {10, 12, 20}, {10, 12, 20}};
  vector<string> files{"floordiv/data/floordiv_data_input1_8.txt",
                       "floordiv/data/floordiv_data_input2_8.txt",
                       "floordiv/data/floordiv_data_output1_8.txt"};
  RunFloorDivKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_FLOORDIV_UT, FILE_DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1}, {3, 4, 5}, {3, 4, 5}};
  vector<string> files{"floordiv/data/floordiv_data_input1_9.txt",
                       "floordiv/data/floordiv_data_input2_9.txt",
                       "floordiv/data/floordiv_data_output1_9.txt"};
  RunFloorDivKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t output[22] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  int64_t input2[22] = {(int64_t)0};
  int32_t output[22] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  int32_t input1[12] = {(int32_t)1};
  int32_t input2[16] = {(int32_t)1};
  int32_t output[16] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_X_ONE_ELEMENT_ZERO_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1}, {2, 2}, {2, 2}};
  int32_t input1[1] = {(int32_t)1};
  int32_t input2[4] = {(int32_t)1};
  input2[1] = (int32_t)0;
  int32_t output[4] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_Y_ONE_ELEMENT_ZERO_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2}, {1}, {2, 2}};
  int32_t input1[4] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  int32_t output[4] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_SAME_SAHPE_ZERO_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  int32_t input1[4] = {(int32_t)1};
  int32_t input2[4] = {(int32_t)1};
  input2[1] = (int32_t)0;
  int32_t output[4] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_BCAST_PARALLEL_ZERO_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{16, 64, 64}, {64, 64}, {16, 64, 64}};
  int64_t input1[65536] = {(int64_t)1};
  int64_t input2[4096] = {(int64_t)1};
  input2[1] = (int64_t)0;
  int64_t output[65536] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_BCAST_ZERO_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {4, 5}, {3, 4, 5}};
  int64_t input1[60] = {(int64_t)1};
  int64_t input2[20] = {(int64_t)1};
  input2[1] = (int64_t)0;
  int64_t output[60] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_FLOORDIV_UT, INPUT_NOBCAST_PARALLEL_ZERO_EXCEPTION) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{16, 64, 64}, {16, 64, 64}, {16, 64, 64}};
  int64_t input1[65536] = {(int64_t)1};
  int64_t input2[65526] = {(int64_t)1};
  input2[1] = (int64_t)0;
  int64_t output[65536] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}