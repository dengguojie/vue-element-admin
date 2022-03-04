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

class TEST_RIGHTSHIFT_UT : public testing::Test {};

template <typename T>
void RightCalcExpectWithSameShape(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  
  T *in1_clamped = new T[input1_num];
  for (int64_t i = 0; i < input1_num; i++) {
    in1_clamped[i] = input1_data[i];
    if (in1_clamped[i] < 0) {
      in1_clamped[i] = 0;
    } else if (in1_clamped[i] > sizeof(T) * CHAR_BIT -1) {
      in1_clamped[i] = sizeof(T) * CHAR_BIT -1;
    }
  }

  if (input0_num == input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      expect_out[j] = input0_data[j] >> in1_clamped[j];
    }
  }
  delete [] in1_clamped;
}

template <typename T>
void RightCalcExpectWithDiffShape(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();

  T *in1_clamped = new T[input1_num];
  for (int64_t i = 0; i < input1_num; i++) {
    in1_clamped[i] = input1_data[i];
    if (in1_clamped[i] < 0) {
      in1_clamped[i] = 0;
    } else if (in1_clamped[i] > sizeof(T) * CHAR_BIT -1) {
      in1_clamped[i] = sizeof(T) * CHAR_BIT -1;
    }
  }

  if (input0_num > input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      int64_t i = j % input1_num;
      expect_out[j] = input0_data[j] >> in1_clamped[i];
    }
  } else {
    for (int64_t j = 0; j < input1_num; ++j) {
      int64_t i = j % input0_num;
      expect_out[j] = input0_data[i] >> in1_clamped[j];
    }
  }
  delete [] in1_clamped;
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "RightShift", "RightShift")       \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Input({"y", data_types[1], shapes[1], datas[1]})            \
      .Output({"z", data_types[2], shapes[2], datas[2]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunRightShiftKernel(vector<string> data_files, vector<DataType> data_types,
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
void RunRightShiftKernel2(vector<DataType> data_types,
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
    RightCalcExpectWithSameShape<T1>(*node_def.get(), output_exp);
  } else {
    RightCalcExpectWithDiffShape<T1>(*node_def.get(), output_exp);
  }

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input1;
  delete[] input2;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{6, 12}, {12}, {6, 12}};
  vector<string> files{"rightshift/data/rightshift_data_input1_1.txt",
                       "rightshift/data/rightshift_data_input2_1.txt",
                       "rightshift/data/rightshift_data_output1_1.txt"};
  RunRightShiftKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{1024, 8}, {1024, 8}, {1024, 8}};
  vector<string> files{"rightshift/data/rightshift_data_input1_2.txt",
                       "rightshift/data/rightshift_data_input2_2.txt",
                       "rightshift/data/rightshift_data_output1_2.txt"};
  RunRightShiftKernel<int16_t, int16_t, int16_t>(files, data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1024}, {4, 1024}, {4, 1024}};
  vector<string> files{"rightshift/data/rightshift_data_input1_3.txt",
                       "rightshift/data/rightshift_data_input2_3.txt",
                       "rightshift/data/rightshift_data_output1_3.txt"};
  RunRightShiftKernel<int32_t, int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_INT64_SUC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}, {15, 12, 30}};
  vector<string> files{"rightshift/data/rightshift_data_input1_4.txt",
                       "rightshift/data/rightshift_data_input2_4.txt",
                       "rightshift/data/rightshift_data_output1_4.txt"};
  RunRightShiftKernel<int64_t, int64_t, int64_t>(files, data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{6, 12}, {12}, {6, 12}};
  vector<string> files{"rightshift/data/rightshift_data_input1_5.txt",
                       "rightshift/data/rightshift_data_input2_5.txt",
                       "rightshift/data/rightshift_data_output1_5.txt"};
  RunRightShiftKernel<uint8_t, uint8_t, uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{5, 12}, {5, 12}, {5, 12}};
  vector<string> files{"rightshift/data/rightshift_data_input1_6.txt",
                       "rightshift/data/rightshift_data_input2_6.txt",
                       "rightshift/data/rightshift_data_output1_6.txt"};
  RunRightShiftKernel<uint16_t, uint16_t, uint16_t>(files, data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};
  vector<string> files{"rightshift/data/rightshift_data_input1_7.txt",
                       "rightshift/data/rightshift_data_input2_7.txt",
                       "rightshift/data/rightshift_data_output1_7.txt"};
  RunRightShiftKernel<uint32_t, uint32_t, uint32_t>(files, data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT64_SUC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}, {15, 12, 30}};
  vector<string> files{"rightshift/data/rightshift_data_input1_8.txt",
                       "rightshift/data/rightshift_data_input2_8.txt",
                       "rightshift/data/rightshift_data_output1_8.txt"};
  RunRightShiftKernel<uint64_t, uint64_t, uint64_t>(files, data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT16_SUCC2) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  RunRightShiftKernel2<uint16_t, uint16_t, uint16_t>(data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT32_SUCC2) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{1}, {3}, {3}};
  RunRightShiftKernel2<uint32_t, uint32_t, uint32_t>(data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT64_SUCC2) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3}, {1}, {3}};
  RunRightShiftKernel2<uint64_t, uint64_t, uint64_t>(data_types, shapes);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT64_DIFFTRENT_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  uint64_t input1[6] = {100, 3, 20, 14, 50, 100};
  uint64_t input2[6] = {2, 3, 4};
  uint64_t output[6] = {0, 0, 0, 0, 0, 0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint64_t output_exp[6] = {25, 0, 1, 3, 7, 6};
  RightCalcExpectWithDiffShape<uint64_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_RIGHTSHIFT_UT, DATA_TYPE_UINT64_SAME_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  uint64_t input1[6] = {100, 3, 9, 14, 6, 8};
  uint64_t input2[6] = {3, 5, 2, 3, 1, 2};
  uint64_t output[6] = {0, 0, 0, 0, 0, 0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  uint64_t output_exp[6] = {12, 0, 2, 1, 3, 2};
  RightCalcExpectWithSameShape<uint64_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

// exception instance
TEST_F(TEST_RIGHTSHIFT_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[12] = {(int32_t)0};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RIGHTSHIFT_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  int64_t input2[22] = {(int64_t)0};
  int64_t output[22] = {(int64_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RIGHTSHIFT_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int64_t output[22] = {(int64_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RIGHTSHIFT_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_RIGHTSHIFT_UT, INPUT2_NEGATIVE_OR_GREATER) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  int32_t input1[6] = {10, 10 , 10, 10, 10, 10};
  int32_t input2[6] = {2, -2, 32};
  int32_t output[6] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int32_t output_exp[6] = {0};
  RightCalcExpectWithDiffShape<int32_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}