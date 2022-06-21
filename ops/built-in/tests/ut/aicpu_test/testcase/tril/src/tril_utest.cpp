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

class TEST_TRIL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, diagonal)        \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Tril", "Tril")                   \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})      \
      .Output({"y", (data_types)[1], (shapes)[1], (datas)[1]})     \
      .Attr("diagonal", diagonal);

template <typename T>
void RunTrilKernel1(vector<string> data_files, vector<DataType> data_types,
                    vector<vector<int64_t>> &shapes, int diagonal = 0) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T *input = new T[input_size];
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T *output = new T[output_size];
  vector<void *> datas = {(void *)input, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, diagonal);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + data_files[1];
  T *output_exp = new T[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete[] input;
  delete[] output;
  delete[] output_exp;
}

template <typename T>
void RunTrilKernel2(vector<DataType> data_types,
                    vector<vector<int64_t>> &shapes, int diagonal = 0) {
  uint64_t input_size = CalTotalElements(shapes, 0);
  T *input = new T[input_size];
  SetRandomValue<T>(input, input_size);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T *output = new T[output_size];
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, diagonal);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  delete[] input;
  delete[] output;
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 4}, {3, 4}};
  vector<string> files{"tril/data/tril_data_input1_1.txt",
                       "tril/data/tril_data_output1_1.txt"};
  RunTrilKernel1<int32_t>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes{{3, 5}, {3, 5}};
  vector<string> files{"tril/data/tril_data_input1_2.txt",
                       "tril/data/tril_data_output1_2.txt"};
  RunTrilKernel1<int64_t>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{16, 128, 1024}, {16, 128, 1024}};
  vector<string> files{"tril/data/tril_data_input1_3.txt",
                       "tril/data/tril_data_output1_3.txt"};
  RunTrilKernel1<float>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes{{5, 5}, {5, 5}};
  vector<string> files{"tril/data/tril_data_input1_4.txt",
                       "tril/data/tril_data_output1_4.txt"};
  RunTrilKernel1<double>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes{{12, 8}, {12, 8}};
  vector<string> files{"tril/data/tril_data_input1_5.txt",
                       "tril/data/tril_data_output1_5.txt"};
  RunTrilKernel1<int8_t>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes{{10, 20}, {10, 20}};
  vector<string> files{"tril/data/tril_data_input1_6.txt",
                       "tril/data/tril_data_output1_6.txt"};
  RunTrilKernel1<uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes{{9, 6}, {9, 6}};
  vector<string> files{"tril/data/tril_data_input1_7.txt",
                       "tril/data/tril_data_output1_7.txt"};
  RunTrilKernel1<int16_t>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes{{4, 5}, {4, 5}};
  vector<string> files{"tril/data/tril_data_input1_8.txt",
                       "tril/data/tril_data_output1_8.txt"};
  RunTrilKernel1<Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_BOOL_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes{{6, 5}, {6, 5}};
  vector<string> files{"tril/data/tril_data_input1_9.txt",
                       "tril/data/tril_data_output1_9.txt"};
  RunTrilKernel1<bool>(files, data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{6, 13}, {6, 13}};
  RunTrilKernel2<uint16_t>(data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_UINT32_SUCC) {
  vector<DataType> data_types = {DT_UINT32, DT_UINT32};
  vector<vector<int64_t>> shapes = {{6, 13}, {6, 13}};
  RunTrilKernel2<uint32_t>(data_types, shapes);
}

TEST_F(TEST_TRIL_UT, DATA_TYPE_UINT64_SUCC) {
  vector<DataType> data_types = {DT_UINT64, DT_UINT64};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  RunTrilKernel2<uint64_t>(data_types, shapes);
}

TEST_F(TEST_TRIL_UT, ATTR_DIAGONAL_POSITIVE) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes{{8, 10}, {8, 10}};
  vector<string> files{"tril/data/tril_data_input1_10.txt",
                       "tril/data/tril_data_output1_10.txt"};
  RunTrilKernel1<double>(files, data_types, shapes, 7);
}

TEST_F(TEST_TRIL_UT, ATTR_DIAGONAL_NEGATIVE) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes{{12, 10}, {12, 10}};
  vector<string> files{"tril/data/tril_data_input1_11.txt",
                       "tril/data/tril_data_output1_11.txt"};
  RunTrilKernel1<int32_t>(files, data_types, shapes, -9);
}

TEST_F(TEST_TRIL_UT, INPUT_BATCH_MATRIXS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes{{2, 3, 6, 8}, {2, 3, 6, 8}};
  vector<string> files{"tril/data/tril_data_input1_12.txt",
                       "tril/data/tril_data_output1_12.txt"};
  RunTrilKernel1<float>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_TRIL_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3, 3}, {3, 4, 2}};
  int32_t input[27] = {(int32_t)1};
  int32_t output[24] = {(int32_t)1};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TRIL_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{2, 3}, {2, 3}};
  bool input[6] = {(int32_t)1};
  bool output[6] = {(double)0};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TRIL_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool output[22] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TRIL_UT, INPUT_DIM_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3}, {3}};
  double input[27] = {(double)1};
  double output[24] = {(double)1};
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, 0);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
