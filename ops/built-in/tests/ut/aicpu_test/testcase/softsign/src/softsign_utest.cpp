#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#include <cmath>
#undef private
#undef protected
#include "Eigen/SVD"
#include "Eigen/Core"
#include "Eigen/Dense"
using namespace std;
using namespace aicpu;
const float alpha=1.0;
class TEST_SOFTSIGN_UT : public testing::Test {};
template <typename T>
void Softsign_float16(const NodeDef &node_def,T expect_out[]){
  auto num0 = static_cast<T>(0);
  auto num1 = static_cast<T>(1);
  auto num2 = static_cast<T>(-1);
  auto input = node_def.MutableInputs(0);
  T *input0_data = (T *)input->GetData();
  int64_t input0_num = input->NumElements();
  for(int64_t j = 0; j < input0_num;++j){
     expect_out[j] = ((input0_data[j]) < num0) ? (input0_data[j]/(input0_data[j]*num2+num1)) : (input0_data[j]/(input0_data[j]+num1));
  }
}
#define CREATE_NODEDEF(shapes, data_type, datas, alpha)            \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Softsign", "Softsign")           \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})      \
      .Output({"y", (data_types)[1], (shapes)[1], (datas)[1]})     \
      .Attr("alpha", (alpha))

template <typename T1,typename T2>
void RunSoftsignKernel(vector<string> data_files,vector<DataType> data_types,
                  vector<vector<int64_t>> &shapes,float alpha) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 *input = new T1[input_size];
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 *output = new T2[output_size];
  vector<void *> datas = {(void *)input, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas,alpha);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);


  data_path = ktestcaseFilePath + data_files[1];
  T2 *output_exp = new T2[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input;
  delete [] output;
  delete [] output_exp;
}

TEST_F(TEST_SOFTSIGN_UT, INPUT_FILE_DTYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}};
  vector<string> files{"softsign/data/softsign_data_input_1.txt",
                       "softsign/data/softsign_data_output_1.txt"};
  RunSoftsignKernel<Eigen::half,Eigen::half>(files, data_types, shapes,alpha);
}

TEST_F(TEST_SOFTSIGN_UT, INPUT_FILE_DTYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}};
  vector<string> files{"softsign/data/softsign_data_input_2.txt",
                       "softsign/data/softsign_data_output_2.txt"};
  RunSoftsignKernel<float,float>(files, data_types, shapes,alpha);
}

TEST_F(TEST_SOFTSIGN_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{15, 12, 30}, {15, 12, 30}};
  vector<string> files{"softsign/data/softsign_data_input_3.txt",
                       "softsign/data/softsign_data_output_3.txt"};
  RunSoftsignKernel<double,double>(files, data_types, shapes,alpha);
}

TEST_F(TEST_SOFTSIGN_UT, INPUT_FILE_DTYPE_FLOAT_SUCC_2) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{32, 64, 64}, {32, 64, 64}};
  vector<string> files{"softsign/data/softsign_data_input_4.txt",
                       "softsign/data/softsign_data_output_4.txt"};
  RunSoftsignKernel<float,float>(files, data_types, shapes,alpha);
}

TEST_F(TEST_SOFTSIGN_UT, INPUT_FILE_DTYPE_FLOAT16_SUCC_2) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{32, 64, 64}, {32, 64, 64}};
  vector<string> files{"softsign/data/softsign_data_input_5.txt",
                       "softsign/data/softsign_data_output_5.txt"};
  RunSoftsignKernel<Eigen::half,Eigen::half>(files, data_types, shapes,alpha);
}

TEST_F(TEST_SOFTSIGN_UT, INPUT_FILE_DTYPE_DOUBLE_SUCC_2) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{64, 64, 64}, {64, 64, 64}};
  vector<string> files{"softsign/data/softsign_data_input_6.txt",
                       "softsign/data/softsign_data_output_6.txt"};
  RunSoftsignKernel<double,double>(files, data_types, shapes,alpha);
}


TEST_F(TEST_SOFTSIGN_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types,datas,alpha);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTSIGN_UT, INPUT_INT_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas,alpha);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTSIGN_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  bool output[22] = {(double)0};
  vector<void *> datas = {(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas,alpha);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SOFTSIGN_UT, OUTPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}};
  float input1[22] = {(float)1};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)nullptr};
  CREATE_NODEDEF(shapes, data_types, datas,alpha);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}