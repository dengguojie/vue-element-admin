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
#include "Eigen/LU"

using namespace std;
using namespace aicpu;

class TEST_Matrix_Determinant_UT : public testing::Test {};

template <typename T>
void CalcExpectOutput(const NodeDef &node_def, T expect_out[],int n,int m) {
  auto input = node_def.MutableInputs(0);
  T *input_data = (T*)input->GetData();
  for (int k = 0; k < n; k++) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eMatrix (m,m);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        eMatrix(i,j) = *(input_data+k*m*m+i*m+j);
      }
    }
    // Using eigen to calculate determinant
    T result = eMatrix.determinant();
    *(expect_out + k) = result;
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                          \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(), "MatrixDeterminant", "MatrixDeterminant") \
      .Input({"x", (data_types)[0], (shapes)[0], (datas)[0]})              \
      .Output({"y", (data_types)[1], (shapes)[1], (datas)[1]});

// T1 and T2 are two input and output types respectively
template <typename T1, typename T2>
void RunMatrixDeterminantKernel(vector<string> data_files, 
                                vector<DataType> data_types,vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  // Call CalTotalElements function to calculate the number of elements of the first input
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 input[input_size];
  // Call ReadFile function to read the first input data from the file
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);
  // Call CalTotalElements function to calculate the number of output elements
  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 output[output_size];
  vector<void*> datas = {(void*)input, (void*)output};
  // Construct operator node information
  CREATE_NODEDEF(shapes, data_types, datas);
  // Executive operator
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  data_path = ktestcaseFilePath + data_files[1];
  T2 output_exp[output_size];
  // Call ReadFile function to read generated expected data from file
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  // The actual calculation results are compared with the expected data
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

// Calculate output by youself function
template<typename T1, typename T2>
void RunMatrixDeterminantKernel2(vector<string> data_files, vector<DataType> data_types,
                                 vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  // Gen data use SetRandomValue for input
  uint64_t input_size = CalTotalElements(shapes, 0);
  T1 input[input_size];
  // Call the ReadFile function to read the first input data from the file
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);
  uint64_t output_size = CalTotalElements(shapes, 1);
  T2 output[output_size];
  vector<void *> datas = {(void *)input,(void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  T2 output_exp[output_size];
  CalcExpectOutput<T2>(*node_def.get(), output_exp,output_size,shapes[0][shapes[0].size()-1]);
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_Matrix_Determinant_UT, DATA_TYPE_FLOAT_SUCC) {
  // Define the input and output data types and shapes, as well as test data files
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3,5,5}, {3}};
  vector<string> files{"matrix_determinant/data/matrix_determinant_data_input1_1.txt",
                       "matrix_determinant/data/matrix_determinant_data_output1_1.txt"};
  // Executing MatrixDeterminant operator
  RunMatrixDeterminantKernel2<float, float>(files, data_types, shapes);
}

TEST_F(TEST_Matrix_Determinant_UT, DATA_TYPE_DOUBLE_SUCC) {
  // Define the input and output data types and shapes, as well as test data files
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 4, 5, 5}, {12}};
  vector<string> files{"matrix_determinant/data/matrix_determinant_data_input1_2.txt",
                       "matrix_determinant/data/matrix_determinant_data_output1_2.txt"};
  // Executing MatrixDeterminant operator
  RunMatrixDeterminantKernel2<double, double>(files, data_types, shapes);
}

TEST_F(TEST_Matrix_Determinant_UT, DATA_TYPE_COMPLEX64_SUCC) {
  // Define the input and output data types and shapes, as well as test data files
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 5, 5}, {3}};
  vector<string> files{"matrix_determinant/data/matrix_determinant_data_input1_3.txt",
                       "matrix_determinant/data/matrix_determinant_data_output1_3.txt"};
  // Executing MatrixDeterminant operator
  RunMatrixDeterminantKernel<complex<float>, complex<float>>(files, data_types, shapes);
}

TEST_F(TEST_Matrix_Determinant_UT, DATA_TYPE_COMPLEX128_SUCC) {
  // Define the input and output data types and shapes, as well as test data files
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 4, 5, 5}, {12}};
  vector<string> files{"matrix_determinant/data/matrix_determinant_data_input1_4.txt",
                       "matrix_determinant/data/matrix_determinant_data_output1_4.txt"};
  // Executing MatrixDeterminant operator
  RunMatrixDeterminantKernel<complex<double>, complex<double>>(files, data_types, shapes);
}

// Exception
TEST_F(TEST_Matrix_Determinant_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3,5,5}, {3}};
  bool output[3] = {(double)0};
  vector<void*> datas = {(void*)nullptr, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Matrix_Determinant_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3,5,5}, {3}};
  bool input[75] = {(bool)1};
  bool output[3] = {(double)0};
  vector<void*> datas = {(void *)input, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_Matrix_Determinant_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3,5,4}, {3}};
  double input[60] = {(double)1};
  double output[3] = {(double)0};
  vector<void*> datas = {(void*)input, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
