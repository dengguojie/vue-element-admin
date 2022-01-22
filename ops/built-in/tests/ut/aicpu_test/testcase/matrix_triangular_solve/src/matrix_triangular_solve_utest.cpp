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
#include "Eigen/Dense"
#include "complex"
#include <cmath>

using namespace std;
using namespace Eigen;
using namespace aicpu;

class TEST_MATRIXTRIANGULARSOLVE_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, lowerattr, adjointattr)          \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                 \
  NodeDefBuilder(node_def.get(), "MatrixTriangularSolve", "MatrixTriangularSolve") \
      .Input({"matrix", (data_types)[0], (shapes)[0], (datas)[0]})                 \
      .Input({"rhs", (data_types)[1], (shapes)[1], (datas)[1]})                    \
      .Attr("lower", lowerattr)                                                    \
      .Attr("adjoint", adjointattr)                                                \
      .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]})

template <typename T>
bool FloatCompareResult(T output[], T expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (fabs(output[i] - expect_output[i]) > 0.001) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

// read input and output data from files which generate by your python file
template <typename T1, typename T2, typename T3>
void RunMatrixTriangularSolveKernel(vector<string> data_files, 
  vector<DataType> data_types,
  vector<vector<int64_t>>& shapes, 
  bool lowerattr, 
  bool adjointattr) {
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

  //Count how many outputs there are, and create an array.
  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 output[output_size];
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  //Create an operator instance 
  CREATE_NODEDEF(shapes, data_types, datas, lowerattr, adjointattr);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T3 output_exp[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = FloatCompareResult(output, output_exp, output_size);  //check
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2, 3, 3}, {2, 2, 3, 2}, {2, 2, 3, 2}};
  vector<string> files{
    "matrix_triangular_solve/data/matrix_triangular_solve_data_input1_1.txt",
    "matrix_triangular_solve/data/matrix_triangular_solve_data_input2_1.txt",
    "matrix_triangular_solve/data/matrix_triangular_solve_data_output1_1.txt"};
  RunMatrixTriangularSolveKernel<float, float, float>(files, data_types, shapes, 
   true, true);
}

TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{5, 5}, {5, 2}, {5, 2}};
  vector<string> files{
    "matrix_triangular_solve/data/matrix_triangular_solve_data_input1_2.txt",
    "matrix_triangular_solve/data/matrix_triangular_solve_data_input2_2.txt",
    "matrix_triangular_solve/data/matrix_triangular_solve_data_output1_2.txt"};
  RunMatrixTriangularSolveKernel<double, double, double>(files, data_types, shapes,
   false, false);
}

TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{10, 10, 10}, {10, 10, 10}, {10, 10, 10}};
  vector<string> files{
    "matrix_triangular_solve/data/matrix_triangular_solve_data_input1_3.txt",
    "matrix_triangular_solve/data/matrix_triangular_solve_data_input2_3.txt",
    "matrix_triangular_solve/data/matrix_triangular_solve_data_output1_3.txt"};
  RunMatrixTriangularSolveKernel<std::complex<float>, std::complex<float>, 
   std::complex<float>>(files, data_types,shapes, false, true);
}

TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{1, 10, 10}, {1, 10, 1}, {1, 10, 1}};
  vector<string> files{
    "matrix_triangular_solve/data/matrix_triangular_solve_data_input1_4.txt",
    "matrix_triangular_solve/data/matrix_triangular_solve_data_input2_4.txt",
    "matrix_triangular_solve/data/matrix_triangular_solve_data_output1_4.txt"};
  RunMatrixTriangularSolveKernel<std::complex<double>, std::complex<double>, 
  std::complex<double>>(files, data_types, shapes, true, false);
}

// exception instance  
//not square
TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 4}, {2, 2, 4}};
  float input1[16] = {(float)1};
  float input2[16] = {(float)1};
  float output[16] = {(float)0};
  vector<void *> datas = {(void *)input1,(void *)input2,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, true, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

//unsupported type
TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32,DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2}, {2, 2}, {2, 2}};
  int32_t input1[4] = {(int32_t)1};
  int32_t input2[4] = {(int32_t)1};
  int32_t output[4] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, true, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

//input0 null
TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, INPUT0_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{7, 7}, {7, 7}, {7}};
  float input2[49] = {(float)1};
  float output[7] = {(float)0};
  vector<void *> datas = {(void *)nullptr,(void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, true, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

//input1 null
TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, INPUT1_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{8, 8}, {8, 8}, {8}};
  float input1[64] = {(float)1};
  float output[8] = {(float)0};
  vector<void *> datas = {(void *)input1,(void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, true, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

//mismatched shape
TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, MISMATCHED_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{7, 7}, {6, 7}, {3, 2}};
  float input1[49] = {(float)1};
  float input2[42] = {(float)1};
  float output[6] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, true, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

//mismatched type
TEST_F(TEST_MATRIXTRIANGULARSOLVE_UT, MISMATCHED_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 3}, {3, 3}, {3, 3}};
  int32_t input1[9] = {(int32_t)1};
  int32_t input2[9] = {(int32_t)1};
  int32_t output[9] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas, true, false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}