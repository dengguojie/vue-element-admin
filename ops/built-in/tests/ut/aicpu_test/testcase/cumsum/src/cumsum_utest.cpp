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

class TEST_CUMSUM_UT : public testing::Test {};

template <typename T>
void CalcExpectWithType(const NodeDef &node_def, bool exclusive, bool reverse,
                        T expect_out[]) {
  auto input_data = reinterpret_cast<T *>(node_def.MutableInputs(0)->GetData());
  auto axis_data =
      reinterpret_cast<int32_t *>(node_def.MutableInputs(1)->GetData());
  int32_t axis = *axis_data;
  auto shape = node_def.MutableInputs(0)->GetTensorShape();
  const int64_t rank = shape->GetDims();
  if (axis < 0) axis += shape->GetDims();
  size_t inner = 1;
  size_t outer = 1;
  size_t depth = 1;
  for (int32_t i = 0; i < rank; ++i) {
    if (i < axis)
      inner *= shape->GetDimSize(i);
    else if (i > axis)
      outer *= shape->GetDimSize(i);
    else
      depth = shape->GetDimSize(i);
  }
  for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
    size_t outer_index_adj;
    if (reverse)
      outer_index_adj = (outer - 1) - outer_index;
    else
      outer_index_adj = outer_index;
    for (size_t inner_index = 0; inner_index < inner; inner_index++) {
      // T accumulator = 0;
      auto accumulator = static_cast<T>(0);
      size_t inner_index_adj;
      if (reverse)
        inner_index_adj = (inner - 1) - inner_index;
      else
        inner_index_adj = inner_index;
      for (size_t depth_index = 0; depth_index < depth; depth_index++) {
        size_t depth_index_adj;
        if (reverse)
          depth_index_adj = (depth - 1) - depth_index;
        else
          depth_index_adj = depth_index;
        size_t index = outer_index_adj;
        index += inner_index_adj * depth * outer;
        index += depth_index_adj * outer;
        if (exclusive) {
          expect_out[index] = accumulator;
          accumulator += input_data[index];
        } else {
          accumulator += input_data[index];
          expect_out[index] = accumulator;
        }
      }
    }
  }
}
#define CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse) \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();    \
  NodeDefBuilder(node_def.get(), "Cumsum", "Cumsum")                  \
      .Input({"x", data_types[0], shapes[0], datas[0]})               \
      .Input({"axis", data_types[1], shapes[1], datas[1]})            \
      .Output({"y", data_types[2], shapes[2], datas[2]})              \
      .Attr("exclusive", exclusive)                                   \
      .Attr("reverse", reverse)

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunCumsumKernel(vector<string> data_files, vector<DataType> data_types,
                     vector<vector<int64_t>> &shapes, bool exclusive,
                     bool reverse) {
  // read data from file for input1
  string data_path_1 = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 *input = new T1[input_size];
  bool status = ReadFile(data_path_1, input, input_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  string data_path_2 = ktestcaseFilePath + data_files[1];
  uint64_t axis_size = CalTotalElements(shapes, 1);
  T2 *axis = new T2[axis_size];
  status = ReadFile(data_path_2, axis, axis_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input, (void *)axis, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  string data_path_3 = ktestcaseFilePath + data_files[2];
  T3 *output_exp = new T3[output_size];
  status = ReadFile(data_path_3, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input;
  delete[] axis;
  delete[] output;
  delete[] output_exp;
}
// only generate input data by SetRandomValue,
// and calculate output by youself function
template <typename T1, typename T2, typename T3>
void RunCumsumKernel2(vector<DataType> data_types,
                      vector<vector<int64_t>> &shapes, bool exclusive,
                      bool reverse) {
  // gen data use SetRandomValue for input
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 *input = new T1[input_size];
  SetRandomValue<T1>(input, input_size);

  // gen data use SetRandomValue for axis
  uint64_t axis_size = CalTotalElements(shapes, 1);
  T2 *axis = new T2[axis_size];
  SetRandomValue<T2>(axis, axis_size, -3.0, 3.0);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input, (void *)axis, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // calculate output_exp
  T3 *output_exp = new T3[output_size];
  CalcExpectWithType<T1>(*node_def.get(), exclusive, reverse, output_exp);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input;
  delete[] axis;
  delete[] output;
  delete[] output_exp;
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_EF_RF) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float16.txt",
                       "cumsum/data/cumsum_data_axis_int32_float16.txt",
                       "cumsum/data/cumsum_data_output_float16_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<Eigen::half, int32_t, Eigen::half>(files, data_types, shapes,
                                                     exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_ET_RF) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float16.txt",
                       "cumsum/data/cumsum_data_axis_int32_float16.txt",
                       "cumsum/data/cumsum_data_output_float16_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<Eigen::half, int32_t, Eigen::half>(files, data_types, shapes,
                                                     exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_EF_RT) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float16.txt",
                       "cumsum/data/cumsum_data_axis_int32_float16.txt",
                       "cumsum/data/cumsum_data_output_float16_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<Eigen::half, int32_t, Eigen::half>(files, data_types, shapes,
                                                     exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_ET_RT) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float16.txt",
                       "cumsum/data/cumsum_data_axis_int32_float16.txt",
                       "cumsum/data/cumsum_data_output_float16_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<Eigen::half, int32_t, Eigen::half>(files, data_types, shapes,
                                                     exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT_EF_RF) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float32.txt",
                       "cumsum/data/cumsum_data_axis_int32_float32.txt",
                       "cumsum/data/cumsum_data_output_float32_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<float, int32_t, float>(files, data_types, shapes, exclusive,
                                         reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT_ET_RF) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float32.txt",
                       "cumsum/data/cumsum_data_axis_int32_float32.txt",
                       "cumsum/data/cumsum_data_output_float32_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<float, int32_t, float>(files, data_types, shapes, exclusive,
                                         reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT_EF_RT) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float32.txt",
                       "cumsum/data/cumsum_data_axis_int32_float32.txt",
                       "cumsum/data/cumsum_data_output_float32_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<float, int32_t, float>(files, data_types, shapes, exclusive,
                                         reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT_ET_RT) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float32.txt",
                       "cumsum/data/cumsum_data_axis_int32_float32.txt",
                       "cumsum/data/cumsum_data_output_float32_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<float, int32_t, float>(files, data_types, shapes, exclusive,
                                         reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_DOUBLE_EF_RF) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float64.txt",
                       "cumsum/data/cumsum_data_axis_int32_float64.txt",
                       "cumsum/data/cumsum_data_output_float64_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<double, int32_t, double>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_DOUBLE_ET_RF) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float64.txt",
                       "cumsum/data/cumsum_data_axis_int32_float64.txt",
                       "cumsum/data/cumsum_data_output_float64_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<double, int32_t, double>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_DOUBLE_EF_RT) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float64.txt",
                       "cumsum/data/cumsum_data_axis_int32_float64.txt",
                       "cumsum/data/cumsum_data_output_float64_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<double, int32_t, double>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_DOUBLE_ET_RT) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_float64.txt",
                       "cumsum/data/cumsum_data_axis_int32_float64.txt",
                       "cumsum/data/cumsum_data_output_float64_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<double, int32_t, double>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT8_EF_RF) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int8.txt",
                       "cumsum/data/cumsum_data_axis_int32_int8.txt",
                       "cumsum/data/cumsum_data_output_int8_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<int8_t, int32_t, int8_t>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT8_ET_RF) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int8.txt",
                       "cumsum/data/cumsum_data_axis_int32_int8.txt",
                       "cumsum/data/cumsum_data_output_int8_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<int8_t, int32_t, int8_t>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT8_EF_RT) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int8.txt",
                       "cumsum/data/cumsum_data_axis_int32_int8.txt",
                       "cumsum/data/cumsum_data_output_int8_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<int8_t, int32_t, int8_t>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT8_ET_RT) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int8.txt",
                       "cumsum/data/cumsum_data_axis_int32_int8.txt",
                       "cumsum/data/cumsum_data_output_int8_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<int8_t, int32_t, int8_t>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT16_EF_RF) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int16.txt",
                       "cumsum/data/cumsum_data_axis_int32_int16.txt",
                       "cumsum/data/cumsum_data_output_int16_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<int16_t, int32_t, int16_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT16_ET_RF) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int16.txt",
                       "cumsum/data/cumsum_data_axis_int32_int16.txt",
                       "cumsum/data/cumsum_data_output_int16_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<int16_t, int32_t, int16_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT16_EF_RT) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int16.txt",
                       "cumsum/data/cumsum_data_axis_int32_int16.txt",
                       "cumsum/data/cumsum_data_output_int16_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<int16_t, int32_t, int16_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT16_ET_RT) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int16.txt",
                       "cumsum/data/cumsum_data_axis_int32_int16.txt",
                       "cumsum/data/cumsum_data_output_int16_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<int16_t, int32_t, int16_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT32_EF_RF) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int32.txt",
                       "cumsum/data/cumsum_data_axis_int32_int32.txt",
                       "cumsum/data/cumsum_data_output_int32_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<int32_t, int32_t, int32_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT32_ET_RF) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int32.txt",
                       "cumsum/data/cumsum_data_axis_int32_int32.txt",
                       "cumsum/data/cumsum_data_output_int32_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<int32_t, int32_t, int32_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT32_EF_RT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int32.txt",
                       "cumsum/data/cumsum_data_axis_int32_int32.txt",
                       "cumsum/data/cumsum_data_output_int32_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<int32_t, int32_t, int32_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT32_ET_RT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int32.txt",
                       "cumsum/data/cumsum_data_axis_int32_int32.txt",
                       "cumsum/data/cumsum_data_output_int32_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<int32_t, int32_t, int32_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT64_EF_RF) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int64.txt",
                       "cumsum/data/cumsum_data_axis_int32_int64.txt",
                       "cumsum/data/cumsum_data_output_int64_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<int64_t, int32_t, int64_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT64_ET_RF) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int64.txt",
                       "cumsum/data/cumsum_data_axis_int32_int64.txt",
                       "cumsum/data/cumsum_data_output_int64_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<int64_t, int32_t, int64_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT64_EF_RT) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int64.txt",
                       "cumsum/data/cumsum_data_axis_int32_int64.txt",
                       "cumsum/data/cumsum_data_output_int64_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<int64_t, int32_t, int64_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT64_ET_RT) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_int64.txt",
                       "cumsum/data/cumsum_data_axis_int32_int64.txt",
                       "cumsum/data/cumsum_data_output_int64_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<int64_t, int32_t, int64_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT8_EF_RF) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_uint8.txt",
                       "cumsum/data/cumsum_data_axis_int32_uint8.txt",
                       "cumsum/data/cumsum_data_output_uint8_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<uint8_t, int32_t, uint8_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT8_ET_RF) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_uint8.txt",
                       "cumsum/data/cumsum_data_axis_int32_uint8.txt",
                       "cumsum/data/cumsum_data_output_uint8_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<uint8_t, int32_t, uint8_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT8_EF_RT) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_uint8.txt",
                       "cumsum/data/cumsum_data_axis_int32_uint8.txt",
                       "cumsum/data/cumsum_data_output_uint8_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<uint8_t, int32_t, uint8_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT8_ET_RT) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_uint8.txt",
                       "cumsum/data/cumsum_data_axis_int32_uint8.txt",
                       "cumsum/data/cumsum_data_output_uint8_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<uint8_t, int32_t, uint8_t>(files, data_types, shapes,
                                             exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT16_EF_RF) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_uint16.txt",
                       "cumsum/data/cumsum_data_axis_int32_uint16.txt",
                       "cumsum/data/cumsum_data_output_uint16_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<uint16_t, int32_t, uint16_t>(files, data_types, shapes,
                                               exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT16_ET_RF) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_uint16.txt",
                       "cumsum/data/cumsum_data_axis_int32_uint16.txt",
                       "cumsum/data/cumsum_data_output_uint16_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<uint16_t, int32_t, uint16_t>(files, data_types, shapes,
                                               exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT16_EF_RT) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_uint16.txt",
                       "cumsum/data/cumsum_data_axis_int32_uint16.txt",
                       "cumsum/data/cumsum_data_output_uint16_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<uint16_t, int32_t, uint16_t>(files, data_types, shapes,
                                               exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT16_ET_RT) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_uint16.txt",
                       "cumsum/data/cumsum_data_axis_int32_uint16.txt",
                       "cumsum/data/cumsum_data_output_uint16_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<uint16_t, int32_t, uint16_t>(files, data_types, shapes,
                                               exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT32_EF_RF) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel2<uint32_t, int32_t, uint32_t>(data_types, shapes, exclusive,
                                                reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT32_ET_RF) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel2<uint32_t, int32_t, uint32_t>(data_types, shapes, exclusive,
                                                reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT32_EF_RT) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel2<uint32_t, int32_t, uint32_t>(data_types, shapes, exclusive,
                                                reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT32_ET_RT) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel2<uint32_t, int32_t, uint32_t>(data_types, shapes, exclusive,
                                                reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT64_EF_RF) {
  vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel2<uint64_t, int32_t, uint64_t>(data_types, shapes, exclusive,
                                                reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT64_ET_RF) {
  vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel2<uint64_t, int32_t, uint64_t>(data_types, shapes, exclusive,
                                                reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT64_EF_RT) {
  vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel2<uint64_t, int32_t, uint64_t>(data_types, shapes, exclusive,
                                                reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT64_ET_RT) {
  vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel2<uint64_t, int32_t, uint64_t>(data_types, shapes, exclusive,
                                                reverse);
}
// TEST_F(TEST_CUMSUM_UT,DATA_TYPE_QINT8_SUCC){}
// TEST_F(TEST_CUMSUM_UT,DATA_TYPE_QINT32_SUCC){}
// TEST_F(TEST_CUMSUM_UT,DATA_TYPE_QUINT8_SUCC){}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX64_EF_RF) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_complex64.txt",
                       "cumsum/data/cumsum_data_axis_int32_complex64.txt",
                       "cumsum/data/cumsum_data_output_complex64_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<complex<float>, int32_t, complex<float>>(
      files, data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX64_ET_RF) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_complex64.txt",
                       "cumsum/data/cumsum_data_axis_int32_complex64.txt",
                       "cumsum/data/cumsum_data_output_complex64_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<complex<float>, int32_t, complex<float>>(
      files, data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX64_EF_RT) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_complex64.txt",
                       "cumsum/data/cumsum_data_axis_int32_complex64.txt",
                       "cumsum/data/cumsum_data_output_complex64_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<complex<float>, int32_t, complex<float>>(
      files, data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX64_ET_RT) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_complex64.txt",
                       "cumsum/data/cumsum_data_axis_int32_complex64.txt",
                       "cumsum/data/cumsum_data_output_complex64_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<complex<float>, int32_t, complex<float>>(
      files, data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX128_EF_RF) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_complex128.txt",
                       "cumsum/data/cumsum_data_axis_int32_complex128.txt",
                       "cumsum/data/cumsum_data_output_complex128_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<complex<double>, int32_t, complex<double>>(
      files, data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX128_ET_RF) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_complex128.txt",
                       "cumsum/data/cumsum_data_axis_int32_complex128.txt",
                       "cumsum/data/cumsum_data_output_complex128_ET_RF.txt"};
  bool exclusive = true;
  bool reverse = false;
  RunCumsumKernel<complex<double>, int32_t, complex<double>>(
      files, data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX128_EF_RT) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_complex128.txt",
                       "cumsum/data/cumsum_data_axis_int32_complex128.txt",
                       "cumsum/data/cumsum_data_output_complex128_EF_RT.txt"};
  bool exclusive = false;
  bool reverse = true;
  RunCumsumKernel<complex<double>, int32_t, complex<double>>(
      files, data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX128_ET_RT) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  vector<string> files{"cumsum/data/cumsum_data_input_complex128.txt",
                       "cumsum/data/cumsum_data_axis_int32_complex128.txt",
                       "cumsum/data/cumsum_data_output_complex128_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<complex<double>, int32_t, complex<double>>(
      files, data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_DOUBLE_BIG_EF_RF) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1024, 4, 1024}, {1}, {1024, 4, 1024}};
  vector<string> files{"cumsum/data/cumsum_data_input_double.txt",
                       "cumsum/data/cumsum_data_axis_int32_double.txt",
                       "cumsum/data/cumsum_data_output_double_EF_RF.txt"};
  bool exclusive = false;
  bool reverse = false;
  RunCumsumKernel<double, int32_t, double>(files, data_types, shapes, exclusive,
                                           reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_DOUBLE_BIG_ET_RT) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1024, 4, 1024}, {1}, {1024, 4, 1024}};
  vector<string> files{"cumsum/data/cumsum_data_input_double.txt",
                       "cumsum/data/cumsum_data_axis_int32_double.txt",
                       "cumsum/data/cumsum_data_output_double_ET_RT.txt"};
  bool exclusive = true;
  bool reverse = true;
  RunCumsumKernel<double, int32_t, double>(files, data_types, shapes, exclusive,
                                           reverse);
}
// exception instance
TEST_F(TEST_CUMSUM_UT, AXIS_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {2, 2}, {3, 4, 5}};
  int32_t input[60] = {(int32_t)1};
  int32_t axis[4] = {(int32_t)0};
  int32_t output[60] = {(bool)0};
  vector<void *> datas = {(void *)input, (void *)axis, (void *)output};
  bool exclusive = false;
  bool reverse = false;
  CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, AXIS_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 5, 6}, {1}, {4, 5, 6}};
  int32_t input[120] = {(int32_t)1};
  int32_t axis[1] = {1};
  int32_t output[120] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)axis, (void *)output};
  bool exclusive = false;
  bool reverse = false;
  CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{}, {1}, {}};
  int32_t input[6] = {(int32_t)1};
  int32_t axis[1] = {1};
  int32_t output[6] = {(int32_t)0};
  vector<void *> datas = {(void *)input, (void *)axis, (void *)output};
  bool exclusive = false;
  bool reverse = false;
  CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_STRING, DT_INT32, DT_STRING};
  vector<vector<int64_t>> shapes = {{4, 5, 6}, {1}, {4, 5, 6}};
  int32_t input[120] = {'1'};
  int32_t axis[1] = {1};
  int32_t output[120] = {'0'};
  vector<void *> datas = {(void *)input, (void *)axis, (void *)output};
  bool exclusive = false;
  bool reverse = false;
  CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  int32_t axis[1] = {(int32_t)0};
  int32_t output[60] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)axis, (void *)output};
  bool exclusive = false;
  bool reverse = false;
  CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
  bool input[60] = {(bool)1};
  int32_t axis[4] = {(int32_t)0};
  bool output[60] = {(bool)0};
  vector<void *> datas = {(void *)input, (void *)axis, (void *)output};
  bool exclusive = false;
  bool reverse = false;
  CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}