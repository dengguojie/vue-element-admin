#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_test_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "aicpu_read_file.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class TEST_RANDOM_STANDARD_NORMAL_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, seed1, seed2)                    \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                 \
  if (seed1 == -1) {                                                               \
    NodeDefBuilder(node_def.get(), "RandomStandardNormal", "RandomStandardNormal") \
        .Input({"shape", data_types[0], shapes[0], datas[0]})                      \
        .Output({"y", data_types[1], shapes[1], datas[1]})                         \
        .Attr("dtype", data_types[2]);                                             \
  } else if (seed2 == -1) {                                                        \
    NodeDefBuilder(node_def.get(), "RandomStandardNormal", "RandomStandardNormal") \
        .Input({"shape", data_types[0], shapes[0], datas[0]})                      \
        .Output({"y", data_types[1], shapes[1], datas[1]})                         \
        .Attr("seed", seed1)                                                       \
        .Attr("dtype", data_types[2]);                                             \
  } else {                                                                         \
    NodeDefBuilder(node_def.get(), "RandomStandardNormal", "RandomStandardNormal") \
        .Input({"shape", data_types[0], shapes[0], datas[0]})                      \
        .Output({"y", data_types[1], shapes[1], datas[1]})                         \
        .Attr("seed", seed1)                                                       \
        .Attr("seed2", seed2)                                                      \
        .Attr("dtype", data_types[2]);                                             \
  }

// read input and output data from files which generate by your python file
template <typename T>
void RunRandomKernel(int64_t seed, int64_t seed2, int64_t* input, vector<DataType> data_types, KernelStatus expect_status,
                     const string& data_path) {
  vector<vector<int64_t>> shapes = {{2}, {input[0], input[1]}};
  uint64_t output_size = CalTotalElements(shapes, 1);
  T* output = new T[output_size];
  vector<void*> datas = {(void*)input, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas, seed, seed2);
  RUN_KERNEL(node_def, HOST, expect_status);
  // read data from file for expect ouput
  if (!data_path.empty()) {
    T* output_exp = new T[output_size];
    bool status = ReadFile(data_path, output_exp, output_size);
    EXPECT_EQ(status, true);
    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] output_exp;
  }
  delete[] output;
}

TEST_F(TEST_RANDOM_STANDARD_NORMAL_UT, DATA_TYPE_FLOAT_SUCC) {
  string data_path = ktestcaseFilePath + "random_standard_normal/data/random_standard_normal_data_output_float.txt";
  vector<DataType> data_types = {DT_INT64, DT_FLOAT, DT_FLOAT};
  int64_t input[2] = {100, 100};
  RunRandomKernel<float>(10, 5, input, data_types, KERNEL_STATUS_OK, data_path);
}

TEST_F(TEST_RANDOM_STANDARD_NORMAL_UT, DATA_TYPE_FLOAT_1024_SUCC) {
  string data_path = ktestcaseFilePath + "random_standard_normal/data/random_standard_normal_data_output_float_1024.txt";
  vector<DataType> data_types = {DT_INT64, DT_FLOAT, DT_FLOAT};
  int64_t input[2] = {1, 1024};
  RunRandomKernel<float>(10, 5, input, data_types, KERNEL_STATUS_OK, data_path);
}

TEST_F(TEST_RANDOM_STANDARD_NORMAL_UT, DATA_TYPE_DOUBLE_SUCC) {
  string data_path = ktestcaseFilePath + "random_standard_normal/data/random_standard_normal_data_output_double.txt";
  vector<DataType> data_types = {DT_INT64, DT_DOUBLE, DT_DOUBLE};
  int64_t input[2] = {100, 100};
  RunRandomKernel<double>(10, 5, input, data_types, KERNEL_STATUS_OK, data_path);
}

TEST_F(TEST_RANDOM_STANDARD_NORMAL_UT, DATA_TYPE_HALF_SUCC) {
  string data_path = ktestcaseFilePath + "random_standard_normal/data/random_standard_normal_data_output_half.txt";
  vector<DataType> data_types = {DT_INT64, DT_FLOAT16, DT_FLOAT16};
  int64_t input[2] = {100, 100};
  RunRandomKernel<Eigen::half>(10, 5, input, data_types, KERNEL_STATUS_OK, data_path);
}

TEST_F(TEST_RANDOM_STANDARD_NORMAL_UT, ZERO_SEED1_ZERO_SEED2) {
  vector<DataType> data_types = {DT_INT64, DT_DOUBLE, DT_DOUBLE};
  int64_t input[2] = {100, 100};
  RunRandomKernel<int64_t>(0, 0, input, data_types, KERNEL_STATUS_OK, "");
}

TEST_F(TEST_RANDOM_STANDARD_NORMAL_UT, DATA_TYPE_NOT_MATCH) {
  vector<DataType> data_types = {DT_INT64, DT_DOUBLE, DT_INT64};
  int64_t input[2] = {100, 100};
  RunRandomKernel<int64_t>(-1, -1, input, data_types, KERNEL_STATUS_PARAM_INVALID, "");
}

TEST_F(TEST_RANDOM_STANDARD_NORMAL_UT, DATA_TYPE_NOT_SUPPORT) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  int64_t input[2] = {100, 100};
  RunRandomKernel<int64_t>(-1, -1, input, data_types, KERNEL_STATUS_PARAM_INVALID, "");
}
