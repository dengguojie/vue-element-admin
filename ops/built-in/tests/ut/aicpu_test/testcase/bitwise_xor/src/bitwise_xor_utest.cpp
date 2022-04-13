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

class TEST_BTWISE_XOR_UT : public testing::Test {};

template <typename T>
void CalcExpectWithSameShape(const NodeDef &node_def, bool expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num == input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      expect_out[j] = input0_data[j] ^ input1_data[j];
    }
  }
}

template <typename T>
void CalcExpectWithDiffShape(const NodeDef &node_def, bool expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num > input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      auto tmp_num = (input1_num == 0) ? 1 :input1_num;
      int64_t i = j % input1_num;
      expect_out[j] = input0_data[j] ^ input1_data[i];
    }
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "BitwiseXor", "BitwiseXor")       \
      .Input({"x1", (data_types)[0], (shapes)[0], (datas)[0]})     \
      .Input({"x2", (data_types)[1], (shapes)[1], (datas)[1]})     \
      .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunBitwiseXorKernel(vector<string> data_files, vector<DataType> &data_types,
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

// only generate input data by SetRandomValue,
// and calculate output by youself function
template <typename T1, typename T2, typename T3>
void RunBitwiseXorKernel2(vector<DataType> &data_types,
                          vector<vector<int64_t>> &shapes) {
  // gen data use SetRandomValue for input1
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  SetRandomValue<T1>(input1, input1_size);

  // gen data use SetRandomValue for input2
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  SetRandomValue<T2>(input2, input2_size);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // calculate output_exp
  T3 *output_exp = new T3[output_size];
  if (input1_size == input2_size) {
    CalcExpectWithSameShape<T1>(*node_def.get(), output_exp);
  } else {
    CalcExpectWithDiffShape<T1>(*node_def.get(), output_exp);
  }

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input1;
  delete[] input2;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_BTWISE_XOR_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{7, 12}, {12}, {7, 12}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_1.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_1.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_1.txt"};
  RunBitwiseXorKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{12, 6}, {6}, {12, 6}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_2.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_2.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_2.txt"};
  RunBitwiseXorKernel<int16_t, int16_t, int16_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_3.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_3.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_3.txt"};
  RunBitwiseXorKernel<int32_t, int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{13, 10, 4}, {13, 10, 4}, {13, 10, 4}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_4.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_4.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_4.txt"};
  RunBitwiseXorKernel<int64_t, int64_t, int64_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{7, 12}, {12}, {7, 12}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_5.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_5.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_5.txt"};
  RunBitwiseXorKernel<uint8_t, uint8_t, uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_6.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_6.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_6.txt"};
  RunBitwiseXorKernel<uint16_t, uint16_t, uint16_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_7.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_7.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_7.txt"};
  RunBitwiseXorKernel<uint32_t, uint32_t, uint32_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_8.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_8.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_8.txt"};
  RunBitwiseXorKernel<uint64_t, uint64_t, uint64_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, X_ONE_ELEMENT_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{1}, {12}, {1, 12}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_9.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_9.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_9.txt"};
  RunBitwiseXorKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, Y_ONE_ELEMENT_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{12}, {1}, {1, 12}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_10.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_10.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_10.txt"};
  RunBitwiseXorKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATASIZE_BCAST_64KB_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{4, 32, 64}, {4, 1, 64}, {4, 32, 64}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_11.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_11.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_11.txt"};
  RunBitwiseXorKernel<int64_t, int64_t, int64_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATASIZE_BCAST_512B_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{4, 32, 4}, {4, 1, 4}, {4, 32, 4}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_12.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_12.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_12.txt"};
  RunBitwiseXorKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATASIZE_NO_BCAST_512KB_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{4, 32, 512}, {4, 32, 512}, {4, 32, 512}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_13.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_13.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_13.txt"};
  RunBitwiseXorKernel<int64_t, int64_t, int64_t>(files, data_types, shapes);
}

TEST_F(TEST_BTWISE_XOR_UT, DATASIZE_NO_BCAST_2K_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {
      {4, 32, 4, 4}, {4, 32, 4, 4}, {4, 32, 4, 4}};
  vector<string> files{"bitwise_xor/data/bitwise_xor_data_input1_14.txt",
                       "bitwise_xor/data/bitwise_xor_data_input2_14.txt",
                       "bitwise_xor/data/bitwise_xor_data_output1_14.txt"};
  RunBitwiseXorKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_BTWISE_XOR_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  int32_t input1[12] = {(int32_t)1};
  int32_t input2[16] = {(int32_t)0};
  int32_t output[16] = {(int32_t)1};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BTWISE_XOR_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  int64_t input2[22] = {(int64_t)0};
  int64_t output[22] = {(int32_t)1};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BTWISE_XOR_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool output[22] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BTWISE_XOR_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)1};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}