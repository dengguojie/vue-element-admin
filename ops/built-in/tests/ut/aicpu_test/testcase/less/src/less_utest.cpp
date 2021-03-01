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

class TEST_LESS_UT : public testing::Test {};

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
      expect_out[j] = input0_data[j] < input1_data[j] ? true : false;
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
      int64_t i = j % input1_num;
      expect_out[j] = input0_data[j] < input1_data[i] ? true : false;
    }
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Less", "Less")                   \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_LESS_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  int32_t input1[12] = {(int32_t)1};
  int32_t input2[16] = {(int32_t)0};
  bool output[16] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LESS_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  int64_t input2[22] = {(int64_t)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LESS_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LESS_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LESS_UT, BROADCAST_INPUT_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "less/data/less_data_input1_1.txt";
  constexpr uint64_t input1_size = 3 * 12;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "less/data/less_data_input2_1.txt";
  constexpr uint64_t input2_size = 12;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 3 * 12;
  bool output[output_size] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "less/data/less_data_output1_1.txt";
  bool output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, SAME_INPUT_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{13, 10, 4}, {13, 10, 4}, {13, 10, 4}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "less/data/less_data_input1_2.txt";
  constexpr uint64_t input1_size = 13 * 10 * 4;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "less/data/less_data_input2_2.txt";
  constexpr uint64_t input2_size = 13 * 10 * 4;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 13 * 10 * 4;
  bool output[output_size] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "less/data/less_data_output1_2.txt";
  bool output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, INPUT_FILE_DTYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}, {15, 12, 30}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "less/data/less_data_input1_3.txt";
  constexpr uint64_t input1_size = 15 * 12 * 30;
  float input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "less/data/less_data_input2_3.txt";
  constexpr uint64_t input2_size = 15 * 12 * 30;
  float input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 15 * 12 * 30;
  bool output[output_size] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "less/data/less_data_output1_3.txt";
  bool output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{7, 12, 30}, {7, 12, 30}, {7, 12, 30}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "less/data/less_data_input1_4.txt";
  constexpr uint64_t input1_size = 7 * 12 * 30;
  double input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "less/data/less_data_input2_4.txt";
  constexpr uint64_t input2_size = 7 * 12 * 30;
  double input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 7 * 12 * 30;
  bool output[output_size] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "less/data/less_data_output1_4.txt";
  bool output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2, 11}, {1}, {2, 11}};
  Eigen::half input1[22];
  SetRandomValue<Eigen::half>(input1, 22);
  Eigen::half input2[1];
  SetRandomValue<Eigen::half>(input2, 1);
  bool output[22] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[22] = {false};
  CalcExpectWithDiffShape<Eigen::half>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 22);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {1}, {2, 11}};
  float input1[22];
  SetRandomValue<float>(input1, 22);
  float input2[1];
  SetRandomValue<float>(input2, 1);
  bool output[22] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[22] = {false};
  CalcExpectWithDiffShape<float>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 22);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  double input1[6] = {2.3, 3.4, 9.2, 4.7, 6.8, 8.4};
  double input2[3] = {3.7, 5.6};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[22] = {false};
  CalcExpectWithDiffShape<double>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  int8_t input1[6] = {(int8_t)100, (int8_t)3, (int8_t)9,
                      (int8_t)4, (int8_t)6, (int8_t)8};
  int8_t input2[3] = {(int8_t)3, (int8_t)5, (int8_t)9};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithDiffShape<int8_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  int16_t input1[6] = {(int16_t)100, (int16_t)3, (int16_t)9,
                      (int16_t)4, (int16_t)6, (int16_t)8};
  int16_t input2[3] = {(int16_t)3, (int16_t)5, (int16_t)9};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithDiffShape<int16_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  int32_t input1[6] = {100, 3, 9, 4, 6, 8};
  int32_t input2[3] = {3, 5, 9};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithDiffShape<int32_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  int64_t input1[6] = {100, 3, 9, 4, 6, 8};
  int64_t input2[3] = {3, 5, 9};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithDiffShape<int64_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  uint8_t input1[6] = {(uint8_t)100, (uint8_t)3, (uint8_t)9,
                       (uint8_t)4, (uint8_t)6, (uint8_t)8};
  uint8_t input2[3] = {(uint8_t)3, (uint8_t)5, (uint8_t)9};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithDiffShape<uint8_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  uint16_t input1[6] = {(uint16_t)100, (uint16_t)3, (uint16_t)9,
                        (uint16_t)4, (uint16_t)6, (uint16_t)8};
  uint16_t input2[3] = {(uint16_t)3, (uint16_t)5, (uint16_t)9};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithDiffShape<uint16_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  uint32_t input1[6] = {100, 3, 9, 4, 6, 8};
  uint32_t input2[3] = {3, 5, 9};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithDiffShape<uint32_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {3}, {2, 3}};
  uint64_t input1[6] = {100, 3, 9, 4, 6, 8};
  uint64_t input2[3] = {3, 5, 9};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithDiffShape<uint64_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_LESS_UT, DATA_TYPE_UINT64_SAME_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}, {2, 3}};
  uint64_t input1[6] = {100, 3, 9, 4, 6, 8};
  uint64_t input2[6] = {3, 5, 9, 6, 0, 4};
  bool output[6] = {false};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool output_exp[6] = {false};
  CalcExpectWithSameShape<uint64_t>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, 6);
  EXPECT_EQ(compare, true);
}