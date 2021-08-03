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

class TEST_GREATER_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Greater", "Greater")             \
      .Input({"x1", data_types[0], shapes[0], datas[0]})           \
      .Input({"x2", data_types[1], shapes[1], datas[1]})           \
      .Output({"y", data_types[2], shapes[2], datas[2]})

#define ADD_CASE(base_type, aicpu_type)                                      \
  TEST_F(TEST_GREATER_UT, TestGreater_##aicpu_type) {                        \
    vector<DataType> data_types = {aicpu_type, aicpu_type, DT_BOOL};         \
    vector<vector<int64_t>> shapes = {{2, 4}, {4}, {2, 4}};                  \
    base_type input0[8] = {(base_type)5, (base_type)4, (base_type)6,         \
                           (base_type)7, (base_type)5, (base_type)4,         \
                           (base_type)6, (base_type)7};                      \
    base_type input1[4] = {(base_type)5, (base_type)2, (base_type)5,         \
                           (base_type)10};                                   \
    bool output[40] = {0};                                                   \
    vector<void *> datas = {(void *)input0, (void *)input1, (void *)output}; \
    CREATE_NODEDEF(shapes, data_types, datas);                               \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                            \
    bool output_exp[8] = {0, 1, 1, 0, 0, 1, 1, 0};                           \
    EXPECT_EQ(CompareResult<bool>(output, output_exp, 4), true);        \
  }

ADD_CASE(Eigen::half, DT_FLOAT16)
ADD_CASE(float, DT_FLOAT)
ADD_CASE(double, DT_DOUBLE)
ADD_CASE(int8_t, DT_INT8)
ADD_CASE(int16_t, DT_INT16)
ADD_CASE(int32_t, DT_INT32)
ADD_CASE(int64_t, DT_INT64)
ADD_CASE(uint8_t, DT_UINT8)
ADD_CASE(uint16_t, DT_UINT16)
ADD_CASE(uint32_t, DT_UINT32)
ADD_CASE(uint64_t, DT_UINT64)

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3>
void RunGreaterKernel(vector<string> data_files,
               vector<DataType> data_types,
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
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3 *output_exp = new T3[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_GREATER_UT, X_SCALAR_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{}, {4}, {4}};
  vector<string> files{"greater/data/greater_data_input1_1.txt",
                       "greater/data/greater_data_input2_1.txt",
                       "greater/data/greater_data_output1_1.txt"};
  RunGreaterKernel<float, float, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_UT, Y_SCALAR_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{4}, {}, {4}};
  vector<string> files{"greater/data/greater_data_input1_2.txt",
                       "greater/data/greater_data_input2_2.txt",
                       "greater/data/greater_data_output1_2.txt"};
  RunGreaterKernel<float, float, bool>(files, data_types, shapes);
}

TEST_F(TEST_GREATER_UT, TWO_SCALAR_SUCCESS) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{}, {}, {}};
  vector<string> files{"greater/data/greater_data_input1_3.txt",
                       "greater/data/greater_data_input2_3.txt",
                       "greater/data/greater_data_output1_3.txt"};
  RunGreaterKernel<float, float, bool>(files, data_types, shapes);
}
