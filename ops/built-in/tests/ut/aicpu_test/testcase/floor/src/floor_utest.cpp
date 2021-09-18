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

class TEST_FLOOR_UT : public testing::Test {};

template <typename T>
void CalcExpectWithSameShape(const NodeDef &node_def, bool expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  int64_t input0_num = input0->NumElements();
    for (int64_t j = 0; j < input0_num; ++j) {
      expect_out[j] = Eigen::numext::floor(input0_data[j]);
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Floor", "Floor")                   \
      .Input({"x", data_types[0], shapes[0], datas[0]})           \
      .Output({"y", data_types[1], shapes[1], datas[1]})

// read input and output data from files which generate by your python file
template<typename T1, typename T2>
void RunFloorKernel(vector<string> data_files,
                   vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] output;
  delete [] output_exp;
}

// only generate input data by SetRandomValue,
// and calculate output by youself function
template<typename T1, typename T2>
void RunFloorKernel2(vector<DataType> data_types,
                    vector<vector<int64_t>> &shapes) {
  // gen data use SetRandomValue for input1
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  SetRandomValue<T1>(input1, input1_size);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // calculate output_exp
  T2 *output_exp = new T2[output_size];
  CalcExpectWithSameShape<T1>(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_FLOOR_UT, DATA_TYPE_FLOAT_SUCC_1D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15}, {15}};
  vector<string> files{"floor/data/floor_data_input1_1.txt",
                       "floor/data/floor_data_output1_1.txt"};
  RunFloorKernel<float, float>(files, data_types, shapes);
}

TEST_F(TEST_FLOOR_UT, DATA_TYPE_DOUBLE_SUCC_1D) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{7}, {7}};
  vector<string> files{"floor/data/floor_data_input1_6.txt",
                       "floor/data/floor_data_output1_6.txt"};
  RunFloorKernel<double, double>(files, data_types, shapes);
}

TEST_F(TEST_FLOOR_UT, DATA_TYPE_FLOAT16_SUCC_1D) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{12}, {12}};
  vector<string> files{"floor/data/floor_data_input1_11.txt",
                       "floor/data/floor_data_output1_11.txt"};
  RunFloorKernel<Eigen::half, Eigen::half>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_FLOOR_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}