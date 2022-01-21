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

class TEST_REDUCE_SUM_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, keep_dims) \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();    \
  NodeDefBuilder(node_def.get(), "ReduceSum", "ReduceSum")                  \
      .Input({"x", data_types[0], shapes[0], datas[0]})               \
      .Input({"axes", data_types[1], shapes[1], datas[1]})            \
      .Output({"y", data_types[2], shapes[2], datas[2]})              \
      .Attr("keep_dims", keep_dims)

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunReduceSumKernel(vector<string> data_files, vector<DataType> data_types,
                     vector<vector<int64_t>> &shapes, bool keep_dims) {
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

  uint64_t output_size = 1;
  int32_t axis_temp;
  uint64_t input_shape_len = shapes[0].size();
  for (uint64_t i = 0; i < static_cast<int64_t>(input_shape_len); ++i) {
    uint64_t j = 0;
    for (j = 0; j < axis_size; ++j) {
      if (axis[j] < 0) {
        axis_temp = input_shape_len + axis[j];
      } else {
        axis_temp = axis[j];
      }
      if (axis_temp == i) {
        break;
      }
    }
    if (j >= axis_size) {
    shapes[2].push_back(shapes[0][i]);
    output_size = output_size * shapes[0][i];
    }
  }

  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input, (void *)axis, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, keep_dims);
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

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT8_1X) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_int8_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_int8_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_int8_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<int8_t, int32_t, int8_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT16_1X) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_int16_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_int16_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_int16_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<int16_t, int32_t, int16_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT32_1X) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_int32_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_int32_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_int32_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<int32_t, int32_t, int32_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT64_1X) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_int64_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_int64_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_int64_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<int64_t, int32_t, int64_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_UINT64_1X) {
  vector<DataType> data_types = {DT_UINT64, DT_INT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_uint64_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_uint64_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_uint64_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<uint64_t, int64_t, uint64_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_FLOAT16_1X) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_float16_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_float16_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_float16_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<Eigen::half, int32_t, Eigen::half>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_FLOAT32_1X) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_float32_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_float32_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_float32_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<float, int32_t, float>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_FLOAT64_1X) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_float64_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_float64_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_float64_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<double, int32_t, double>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT8_1X_ET) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_int8_1X_ET.txt",
                       "reduce_sum/data/reduce_sum_data_axis_int8_1X_ET.txt",
                       "reduce_sum/data/reduce_sum_data_output_int8_1X_ET.txt"};
  bool keep_dims = true;
  RunReduceSumKernel<int8_t, int32_t, int8_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT8_2X) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {2}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_int8_2X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_int8_2X.txt",
                       "reduce_sum/data/reduce_sum_data_output_int8_2X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<int8_t, int32_t, int8_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT8_2X_ET) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {2}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_int8_2X_ET.txt",
                       "reduce_sum/data/reduce_sum_data_axis_int8_2X_ET.txt",
                       "reduce_sum/data/reduce_sum_data_output_int8_2X_ET.txt"};
  bool keep_dims = true;
  RunReduceSumKernel<int8_t, int32_t, int8_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_COMPLEX64_1X) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_complex64_1X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_complex64_1X.txt",
                       "reduce_sum/data/reduce_sum_data_output_complex64_1X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<complex<float>, int32_t, complex<float>>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_COMPLEX64_2X) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 4, 5}, {2}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_complex64_2X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_complex64_2X.txt",
                       "reduce_sum/data/reduce_sum_data_output_complex64_2X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<complex<float>, int32_t, complex<float>>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_BIG_2X) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{3, 4, 5, 6, 7, 8}, {2}, {}};
  vector<string> files{"reduce_sum/data/reduce_sum_data_input_big_2X.txt",
                       "reduce_sum/data/reduce_sum_data_axis_big_2X.txt",
                       "reduce_sum/data/reduce_sum_data_output_big_2X.txt"};
  bool keep_dims = false;
  RunReduceSumKernel<uint32_t, int32_t, uint32_t>(files, data_types, shapes, keep_dims);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_INT8_SCALAR) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{}, {}, {}};

  int8_t input1[1] = {1};
  int32_t input2[1] = {0};
  int8_t output[1] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  int8_t output_exp[1] = {1};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_REDUCE_SUM_UT, DATA_TYPE_COMPLEX64_SCALAR) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{}, {}, {}};

  complex<float> input1[1] = {(1.2345678 + 2.3456789j)};
  int32_t input2[1] = {0};
  complex<float> output[1] = {0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  complex<float> output_exp[1] = {(1.2345678 + 2.3456789j)};
  bool compare = CompareResult(output, output_exp, 1);
  EXPECT_EQ(compare, true);
}