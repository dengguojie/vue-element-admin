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

class TEST_MATRIX_BAND_PART_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                     \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();    \
  NodeDefBuilder(node_def.get(), "MatrixBandPart", "MatrixBandPart")  \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})         \
      .Input({"num_lower", (data_types)[1], (shapes)[1], (datas)[1]}) \
      .Input({"num_upper", (data_types)[2], (shapes)[2], (datas)[2]}) \
      .Output({"y", (data_types)[3], (shapes)[3], (datas)[3]})

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3, typename T4>
void RunMatrixBandPartKernel(vector<string> data_files,
                             vector<DataType> data_types,
                             vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 input1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 input2[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  // read data from file for input3
  data_path = ktestcaseFilePath + data_files[2];
  uint64_t input3_size = CalTotalElements(shapes, 2);
  T3 input3[input3_size];
  status = ReadFile(data_path, input3, input3_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 3);
  T4 output[output_size];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[3];
  T4 output_exp[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult<T4>(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {}, {}, {6, 8, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_1.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_1.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_1.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_1.txt"};
  RunMatrixBandPartKernel<int8_t, int32_t, int32_t, int8_t>(files, data_types,
                                                            shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {}, {}, {6, 8, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_2.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_2.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_2.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_2.txt"};
  RunMatrixBandPartKernel<int16_t, int32_t, int32_t, int16_t>(files, data_types,
                                                              shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {}, {}, {6, 8, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_3.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_3.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_3.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_3.txt"};
  RunMatrixBandPartKernel<int32_t, int32_t, int32_t, int32_t>(files, data_types,
                                                              shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT32, DT_INT64};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {}, {}, {6, 8, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_4.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_4.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_4.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_4.txt"};
  RunMatrixBandPartKernel<int64_t, int32_t, int32_t, int64_t>(files, data_types,
                                                              shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {}, {}, {6, 8, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_5.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_5.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_5.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_5.txt"};
  RunMatrixBandPartKernel<uint8_t, int32_t, int32_t, uint8_t>(files, data_types,
                                                              shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{6, 8, 8}, {}, {}, {6, 8, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_6.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_6.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_6.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_6.txt"};
  RunMatrixBandPartKernel<uint16_t, int32_t, int32_t, uint16_t>(
      files, data_types, shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT64, DT_INT64, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{6, 7, 8}, {}, {}, {6, 7, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_7.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_7.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_7.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_7.txt"};
  RunMatrixBandPartKernel<Eigen::half, int64_t, int64_t, Eigen::half>(
      files, data_types, shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT64, DT_INT64, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{6, 7, 8}, {}, {}, {6, 7, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_8.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_8.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_8.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_8.txt"};
  RunMatrixBandPartKernel<float, int64_t, int64_t, float>(files, data_types,
                                                          shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT64, DT_INT64, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{64, 64, 64}, {}, {}, {64, 64, 64}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_9.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_9.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_9.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_9.txt"};
  RunMatrixBandPartKernel<double, int64_t, int64_t, double>(files, data_types,
                                                            shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_BOOL_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_INT64, DT_INT64, DT_BOOL};
  vector<vector<int64_t>> shapes = {{6, 7, 8}, {}, {}, {6, 7, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_10.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_10.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_10.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_10.txt"};
  RunMatrixBandPartKernel<bool, int64_t, int64_t, bool>(files, data_types,
                                                        shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT64, DT_INT64,
                                 DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{6, 7, 8}, {}, {}, {6, 7, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_11.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_11.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_11.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_11.txt"};
  RunMatrixBandPartKernel<std::complex<float>, int64_t, int64_t,
                          std::complex<float>>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_INT64,
                                 DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 7, 8}, {}, {}, {6, 7, 8}};
  vector<string> files{
      "matrix_band_part/data/matrix_band_part_data_input1_12.txt",
      "matrix_band_part/data/matrix_band_part_data_input2_12.txt",
      "matrix_band_part/data/matrix_band_part_data_input3_12.txt",
      "matrix_band_part/data/matrix_band_part_data_output1_12.txt"};
  RunMatrixBandPartKernel<std::complex<double>, int64_t, int64_t,
                          std::complex<double>>(files, data_types, shapes);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{12}, {}, {}, {12}};
  int32_t input1[12] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[12] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, NUM_LOWER_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {1}, {}, {4, 4}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, NUM_UPPER_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {1}, {4, 4}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, NUM_LOWER_DATA_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {}, {4, 4}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)6};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, NUM_UPPER_DATA_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {}, {4, 4}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)6};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, DATA_SIZE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {}, {4, 5}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[20] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {}, {4, 4}};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)nullptr,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, INPUT_DTYPE_UNSUPPORT) {
  vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_INT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {}, {4, 4}};
  int32_t input1[16] = {(uint32_t)1};
  int32_t input2[1] = {(int32_t)0};
  int32_t input3[1] = {(int32_t)0};
  uint32_t output[16] = {(uint32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, NUM_LOWER_DTYPE_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_INT16, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {}, {4, 4}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[1] = {(int16_t)0};
  int32_t input3[1] = {(int32_t)0};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_BAND_PART_UT, NUM_UPPER_DTYPE_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT16, DT_INT32};
  vector<vector<int64_t>> shapes = {{4, 4}, {}, {}, {4, 4}};
  int32_t input1[16] = {(int32_t)1};
  int32_t input2[1] = {(int32_t)0};
  int32_t input3[1] = {(int16_t)0};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}