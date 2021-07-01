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

class TEST_ADD_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Add", "Add")                     \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]});

TEST_F(TEST_ADD_UT, BROADCAST_INPUT_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "add/data/add_data_input1_1.txt";
  int32_t input1[36] = {0};
  bool status = ReadFile(data_path, input1, 36);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "add/data/add_data_input2_1.txt";
  int32_t input2[12] = {0};
  status = ReadFile(data_path, input2, 12);
  EXPECT_EQ(status, true);

  int32_t output[36] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "add/data/add_data_output1_1.txt";
  int32_t output_exp[36] = {0};
  status = ReadFile(data_path, output_exp, 36);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, 36);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, SAME_INPUT_SHAPE_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{18, 20, 6}, {18, 20, 6}, {18, 20, 6}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "add/data/add_data_input1_2.txt";
  constexpr uint64_t input1_size = 18 * 20 * 6;
  int32_t input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "add/data/add_data_input2_2.txt";
  constexpr uint64_t input2_size = 18 * 20 * 6;
  int32_t input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 18 * 20 * 6;
  int32_t output[output_size] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "add/data/add_data_output1_2.txt";
  int32_t output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, FLOAT_DTYPE_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}, {15, 12, 30}};

  // read data from file for input1
  string data_path = ktestcaseFilePath + "add/data/add_data_input1_3.txt";
  constexpr uint64_t input1_size = 15 * 12 * 30;
  float input1[input1_size] = {0};
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + "add/data/add_data_input2_3.txt";
  constexpr uint64_t input2_size = 15 * 12 * 30;
  float input2[input2_size] = {0};
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t output_size = 15 * 12 * 30;
  float output[output_size] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "add/data/add_data_output1_3.txt";
  float output_exp[output_size] = {0};
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT8_SCALAR_ADD_SCALAR_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{}, {}, {}};

  int8_t input1[1] = {1};
  int8_t input2[1] = {2};
  int8_t output[1] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int8_t output_exp[1] = {3};
  bool compare = CompareResult(output, output_exp, 1);
}

TEST_F(TEST_ADD_UT, INT16_SCALAR_ADD_VECTOR_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{}, {2}, {2}};

  int16_t input1[1] = {1};
  int16_t input2[2] = {2, 3};
  int16_t output[2] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int16_t output_exp[2] = {3, 4};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT32_VECTOR_ADD_SCALAR_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2}, {1}, {2}};

  int32_t input1[2] = {2, 5};
  int32_t input2[1] = {3};
  int32_t output[2] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int32_t output_exp[2] = {5, 8};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT64_VECTOR_ADD_VECTOR_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 1, 1}, {2, 1, 1}, {2, 1, 1}};

  int64_t input1[2] = {2, 5};
  int64_t input2[2] = {3, 5};
  int64_t output[2] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int64_t output_exp[2] = {5, 10};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT64_VECTOR_ADD_VECTOR_BROADCAST_0) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{1}, {2, 1, 1, 1}, {2, 1, 1, 1}};

  int64_t input1[1] = {2};
  int64_t input2[2] = {3, 5};
  int64_t output[2] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int64_t output_exp[2] = {5, 7};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT16_VECTOR_ADD_VECTOR_BROADCAST_1) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{2, 1, 1, 1}, {1}, {2, 1, 1, 1}};

  int16_t input1[2] = {2, 3};
  int16_t input2[1] = {3};
  int16_t output[2] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int16_t output_exp[2] = {5, 6};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT16_VECTOR_ADD_VECTOR_DIM_7) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{2, 1, 1, 1, 1, 1, 1}, {1}, {2, 1, 1, 1, 1, 1, 1}};

  int16_t input1[2] = {2, 3};
  int16_t input2[1] = {3};
  int16_t output[2] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int16_t output_exp[2] = {5, 6};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT16_VECTOR_ADD_VECTOR_DIM_8) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{2, 1, 1, 1, 1, 1, 1, 1}, {1}, {2, 1, 1, 1, 1, 1, 1, 1}};

  int16_t input1[2] = {2, 3};
  int16_t input2[1] = {3};
  int16_t output[2] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int16_t output_exp[2] = {5, 6};
  bool compare = CompareResult(output, output_exp, 2);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, INT16_VECTOR_ADD_VECTOR_INVALID_DIM) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{2, 1, 1, 1, 1, 1, 1, 1, 1}, {1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}};

  int16_t input1[2] = {2, 3};
  int16_t input2[1] = {3};
  int16_t output[2] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, INT32_VECTOR_ADD_VECTOR_BROADCAST_BOTH) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{1, 2}, {2, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 2}};

  int32_t input1[2] = {2, 3};
  int32_t input2[2] = {3, 10};
  int32_t output[4] = {0};
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int32_t output_exp[4] = {5, 6, 12, 13};
  bool compare = CompareResult(output, output_exp, 4);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_ADD_UT, UNDEFINED_DTYPE_EXCEPTION) {
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<DataType> data_types = {DT_UNDEFINED, DT_UNDEFINED, DT_UNDEFINED};
    vector<int32_t> input1 = {1};
    vector<int32_t> input2 = {1};
    vector<int32_t> output = {0};
    vector<void *> datas = { (void *)input1.data(),
                             (void *)input2.data(),
                             (void *)output.data() };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, DTYPE_BOOL_UNSORPORT) {
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
    bool input1[1] = {false};
    bool input2[1] = {true};
    bool output[1] = {false};
    vector<void *> datas = { (void *)input1,
                             (void *)input2,
                             (void *)output };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, INPUT_DTYPE_DISMATCH) {
    vector<vector<int64_t>> shapes = {{}, {}, {}};
    vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_DOUBLE};
    float input1[1] = {1.0F};
    double input2[1] = {2.0};
    double output[1] = {0.0};
    vector<void *> datas = { (void *)input1,
                             (void *)input2,
                             (void *)output };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, BROADCAST_MAX_SIZE_DISMATCH) {
    vector<vector<int64_t>> shapes = {{1}, {1}, {1, 1}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    float input1[1] = {1.0F};
    float input2[1] = {2.0F};
    float output[1] = {0.0F};
    vector<void *> datas = { (void *)input1,
                             (void *)input2,
                             (void *)output };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, BROADCAST_SHAPE_DISMATCH) {
    vector<vector<int64_t>> shapes = {{1, 2}, {1, 3}, {1, 1}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    float input1[2] = {1.0F};
    float input2[3] = {2.0F};
    float output[1] = {0.0F};
    vector<void *> datas = { (void *)input1,
                             (void *)input2,
                             (void *)output };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_ADD_UT, BROADCAST_NOT_SUPPORT) {
    vector<vector<int64_t>> shapes = {{1, 2}, {1, 3}, {1, 3}};
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    float input1[2] = {1.0F};
    float input2[3] = {2.0F};
    float output[3] = {0.0F};
    vector<void *> datas = { (void *)input1,
                             (void *)input2,
                             (void *)output };
    CREATE_NODEDEF(shapes, data_types, datas);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}