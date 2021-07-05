#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/kernel_util.h"

#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_SCATTER_ELEMENTS_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, axis)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();       \
  NodeDefBuilder(node_def.get(), "ScatterElements", "ScatterElements")   \
      .Input({"x1", data_types[0], shapes[0], datas[0]})                 \
      .Input({"x2", data_types[1], shapes[1], datas[1]})                 \
      .Input({"x3", data_types[2], shapes[2], datas[2]})                 \
      .Attr("axis", axis)                                                \
      .Output({"y", data_types[3], shapes[3], datas[3]})

#define CREATE_NODEDEF_NO_AXIS(shapes, data_types, datas)                \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();       \
  NodeDefBuilder(node_def.get(), "ScatterElements", "ScatterElements")   \
      .Input({"x1", data_types[0], shapes[0], datas[0]})                 \
      .Input({"x2", data_types[1], shapes[1], datas[1]})                 \
      .Input({"x3", data_types[2], shapes[2], datas[2]})                 \
      .Output({"y", data_types[3], shapes[3], datas[3]})

template <typename T1, typename T2, typename T3, typename T4>
void RunScatterElementsKernel(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  // read data from file
  string data_path = ktestcaseFilePath  + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath  + data_files[2];
  uint64_t input3_size = CalTotalElements(shapes, 2);
  T3 *input3 = new T3[input3_size];
  status = ReadFile(data_path, input3, input3_size);
  EXPECT_EQ(status, true);
  // read output
  uint64_t output_size = CalTotalElements(shapes, 3);
  T4 *output = new T4[output_size];
  //read axis from file
  data_path = ktestcaseFilePath  + data_files[4];
  int64_t *axis = new int64_t[1];
  status = ReadFile(data_path, axis, 1);
  EXPECT_EQ(status, true);
  int64_t axis_value = *axis;
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, axis_value);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  // read expect data
  data_path = ktestcaseFilePath + data_files[3];
  T4 *output_exp = new T4[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input1;
  delete[] input2;
  delete[] input3;
  delete[] output;
  delete[] output_exp;
  delete[] axis;
}

#define ADD_CASE(data_type, indices_type, data_real_type, indices_real_type,   \
                 case_name)                                                    \
  TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_DTYPE_##data_type) {                  \
    vector<DataType> data_types = {data_type, indices_type, data_type,         \
                                   data_type};                                 \
    vector<vector<int64_t>> shapes = {                                         \
        {10, 5, 10, 5, 5}, {1, 2, 1, 1, 2}, {1, 2, 1, 1, 2}, {10, 5, 10, 5, 5}};\
    string pre_input1 = "scatter_elements/data/scatter_elements_data_input1_"; \
    string pre_input2 = "scatter_elements/data/scatter_elements_data_input2_"; \
    string pre_input3 = "scatter_elements/data/scatter_elements_data_input3_"; \
    string pre_output = "scatter_elements/data/scatter_elements_data_output_"; \
    string pre_attr = "scatter_elements/data/scatter_elements_data_attr_";     \
    vector<string> files{pre_input1 + case_name, pre_input2 + case_name,       \
                         pre_input3 + case_name, pre_output + case_name,       \
                         pre_attr + case_name};                                \
    RunScatterElementsKernel<data_real_type, indices_real_type,                \
                             data_real_type, data_real_type>(                  \
        files, data_types, shapes);                                            \
  }



TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_DTYPE_DT_FLOAT16) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1, 5}, {1, 2}, {1, 2}, {1, 5}};
  Eigen::half input_x1[5] = {Eigen::half(1.0), Eigen::half(2.0),
                             Eigen::half(3.0), Eigen::half(4.0),
                             Eigen::half(5.0)};
  int32_t input_x2[2] = {1, 3};
  Eigen::half input_x3[2] = {Eigen::half(1.1), Eigen::half(2.1)};
  Eigen::half output_y[5] = {Eigen::half(0.0)};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF(shapes, data_types, datas, 1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  Eigen::half output_exp[5] = {Eigen::half(1.0), Eigen::half(1.1),
                               Eigen::half(3.0), Eigen::half(2.1),
                              Eigen::half(5.0)};
  bool compare = CompareResult(output_y, output_exp, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_DTYPE_DT_FLOAT_NEGTIVE_INDICES) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {1, 2}, {1, 2}, {1, 5}};
  float input_x1[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  int64_t input_x2[2] = {1, -3};
  float input_x3[2] = {1.1, 2.1};
  float output_y[5] = {0.0};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF(shapes, data_types, datas, 1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  float output_exp[5] = {1.0, 1.1, 2.1, 4.0, 5.0};
  bool compare = CompareResult(output_y, output_exp, 5);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_DTYPE_DT_DOUBLE_WITHOUT_AXIS) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 3}, {2, 3}, {2, 3}, {3, 3}};
  double input_x1[9] = {0};
  int32_t input_x2[6] = {1, 0, 2, 0, 2 , 1};
  double input_x3[6] = {1.0, 1.1, 1.2, 2.0, 2.1, 2.2};
  double output_y[9] = {0};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF_NO_AXIS(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  double output_exp[9] = {2.0, 1.1, 0.0, 1.0, 0.0, 2.2, 0.0, 2.1, 1.2};
  bool compare = CompareResult(output_y, output_exp, 9);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_DTYPE_DT_BOOL_WITH_ONE_DIM) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  bool input_x1[1] = {true};
  int32_t input_x2[1] = {0};
  bool input_x3[1] = {false};
  bool output_y[1] = {true};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF_NO_AXIS(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  bool output_exp[1] = {false};
  bool compare = CompareResult(output_y, output_exp, 1);
  EXPECT_EQ(compare, true);
}

ADD_CASE(DT_INT8, DT_INT32, int8_t, int32_t, "DT_INT8");
ADD_CASE(DT_INT16, DT_INT64, int16_t, int64_t, "DT_INT16");
ADD_CASE(DT_INT32, DT_INT32, int32_t, int32_t, "DT_INT32");
ADD_CASE(DT_INT64, DT_INT64, int64_t, int64_t, "DT_INT64");
ADD_CASE(DT_UINT8, DT_INT32, uint8_t, int32_t, "DT_INT64");
ADD_CASE(DT_UINT16, DT_INT32, uint16_t, int32_t, "DT_UINT16");
ADD_CASE(DT_UINT32, DT_INT32, uint32_t, int32_t, "DT_UINT32");
ADD_CASE(DT_UINT64, DT_INT64, uint64_t, int64_t, "DT_UINT64");
ADD_CASE(DT_COMPLEX64, DT_INT64, complex<float>, int64_t, "DT_COMPLEX64");
ADD_CASE(DT_COMPLEX128, DT_INT64, complex<double>, int64_t, "DT_COMPLEX128");

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_INDICES_EXCEPTION_1) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5, 1}, {1, 2}, {1, 2}, {1, 5, 1}};
  float input_x1[5] = {1.0};
  int64_t input_x2[2] = {1, -3};
  float input_x3[2] = {1.1, 2.1};
  float output_y[5] = {0.0};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF(shapes, data_types, datas, 1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_INDICES_EXCEPTION_2) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {1}, {1, 2}, {1, 5}};
  float input_x1[5] = {1.0};
  int64_t input_x2[2] = {1};
  float input_x3[2] = {1.1, 2.1};
  float output_y[5] = {0.0};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF(shapes, data_types, datas, 1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_DATATYPE_EXCEPTION_1) {
vector<DataType> data_types = {DT_STRING, DT_INT64, DT_STRING, DT_STRING};
  vector<vector<int64_t>> shapes = {{1, 1}, {1, 1}, {1, 1}, {1, 1}};
  string input_x1[1] = {"ai_test"};
  int64_t input_x2[1] = {1};
  string input_x3[1] = {"scatter"};
  string output_y[1] = {"0.0"};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF(shapes, data_types, datas, 0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_DATATYPE_EXCEPTION_2) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT8, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {1, 1}, {1, 1}, {1, 5}};
  float input_x1[5] = {1.0};
  int8_t input_x2[2] = {1};
  float input_x3[2] = {1.1};
  float output_y[5] = {0.0};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF_NO_AXIS(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_NULL_PTR_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {1}, {1, 2}, {1, 5}};
  float output_y[5] = {0.0};
  vector<void *> datas = {(void *)nullptr,
                          (void *)nullptr,
                          (void *)nullptr,
                          (void *)output_y};
  CREATE_NODEDEF_NO_AXIS(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_OUT_OF_BOUND_EXCEPTION_1) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {1, 2}, {1, 2}, {1, 5}};
  float input_x1[5] = {1.0};
  int64_t input_x2[2] = {1, -30};
  float input_x3[2] = {1.1, 2.1};
  float output_y[5] = {0.0};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF(shapes, data_types, datas, 1);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SCATTER_ELEMENTS_UT, INPUT_OUT_OF_BOUND_EXCEPTION_2) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 5}, {1, 2}, {1, 2}, {1, 5}};
  float input_x1[5] = {1.0};
  int64_t input_x2[2] = {1, -3};
  float input_x3[2] = {1.1, 2.1};
  float output_y[5] = {0.0};
  vector<void *> datas = {(void *)input_x1,
                          (void *)input_x2,
                          (void *)input_x3,
                          (void *)output_y};
  CREATE_NODEDEF(shapes, data_types, datas, 100);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
