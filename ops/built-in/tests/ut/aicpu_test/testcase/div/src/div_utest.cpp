#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "node_def_builder.h"
#include "aicpu_read_file.h"
#include "utils/kernel_util.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_DIV_UT : public testing::Test {};

template <typename T>
uint32_t CalcExpectWithSameShape(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num == input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      if(input1_data[j] != 0)
      expect_out[j] = input0_data[j] / input1_data[j] ;
      else{
        KERNEL_LOG_ERROR("Invalid argumengt: Division by zero.");
        return KERNEL_STATUS_INNER_ERROR;
      }
    }
  }
}

template <typename T>
uint32_t CalcExpectWithDiffShape(const NodeDef &node_def, T expect_out[]) {
  auto input0 = node_def.MutableInputs(0);
  T *input0_data = (T *)input0->GetData();
  auto input1 = node_def.MutableInputs(1);
  T *input1_data = (T *)input1->GetData();
  int64_t input0_num = input0->NumElements();
  int64_t input1_num = input1->NumElements();
  if (input0_num > input1_num) {
    for (int64_t j = 0; j < input0_num; ++j) {
      int64_t i = j % input1_num;
      if(input1_data[i] != 0)
      expect_out[j] = input0_data[j] / input1_data[i] ;
      else{
       KERNEL_LOG_ERROR("Invalid argumengt: Division by zero.");
        return KERNEL_STATUS_INNER_ERROR;
      }
    }
  }
}

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Div", "Div")                     \
      .Input({"x1", (data_types)[0], (shapes)[0], (datas)[0]})     \
      .Input({"x2", (data_types)[1], (shapes)[1], (datas)[1]})     \
      .Output({"y", (data_types)[2], (shapes)[2], (datas)[2]})

bool ReadDivFileComplex64(std::string file_name, std::complex<float> output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    for (uint64_t index = 0; index < size; ++index) {
      string s, s1, s2;
      stringstream ss, sss;
      string ::size_type n1, n2, n3;
      bool flag = true;
      getline(in_file, s);
      n1 = s.find("(", 0);
      n2 = s.find("+", 0);
      if (n2 == string::npos) {
        n2 = s.find("-", n1 + 2);
        flag = false;
      }
      n3 = s.find("j", 0);
      s1 = s.substr(n1 + 1, n2 - n1 - 1);
      s2 = s.substr(n2 + 1, n3 - n2 - 1);
      float temp;
      ss << s1;
      ss >> temp;
      output[index].real(temp);
      sss << s2;
      sss >> temp;
      if (!flag)
        temp *= -1;
      output[index].imag(temp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

bool ReadDivFileComplex128(std::string file_name, std::complex<double> output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    for (uint64_t index = 0; index < size; ++index) {
      string s, s1, s2;
      stringstream ss, sss;
      string ::size_type n1, n2, n3;
      bool flag = true;
      getline(in_file, s);
      n1 = s.find("(", 0);
      n2 = s.find("+", 0);
      if (n2 == string::npos) {
        n2 = s.find("-", n1 + 2);
        flag = false;
      }
      n3 = s.find("j", 0);
      s1 = s.substr(n1 + 1, n2 - n1 - 1);
      s2 = s.substr(n2 + 1, n3 - n2 - 1);
      double temp;
      ss << s1;
      ss >> temp;
      output[index].real(temp);
      sss << s2;
      sss >> temp;
      if (!flag)
        temp *= -1;
      output[index].imag(temp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

bool CompareDivResultComplex64(std::complex<float> output[],
                    std::complex<float> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (std::abs(output[i] - expect_output[i]) > 1e-6) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool CompareDivResultComplex128(std::complex<double> output[],
                    std::complex<double> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    if (std::abs(output[i] - expect_output[i]) > 1e-6) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

void RunDivKernelComplex64(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<float> input1[input1_size];
  bool status = ReadDivFileComplex64(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

   data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  std::complex<float> input2[input2_size];
  status = ReadDivFileComplex64(data_path, input2, input1_size);
  EXPECT_EQ(status, true);
  uint64_t output_size = CalTotalElements(shapes, 2);
  std::complex<float> output[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  data_path = ktestcaseFilePath + data_files[2];
  std::complex<float> output_exp[output_size];
  status = ReadDivFileComplex64(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  bool compare = CompareDivResultComplex64(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

void RunDivKernelComplex128(vector<string> data_files,
                              vector<DataType> data_types,
                              vector<vector<int64_t>> &shapes) {
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  std::complex<double> input1[input1_size];
  bool status = ReadDivFileComplex128(data_path, input1, input1_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + data_files[1];
   uint64_t input2_size = CalTotalElements(shapes, 1);
  std::complex<double> input2[input2_size];
   status = ReadDivFileComplex128(data_path, input2, input2_size);
  EXPECT_EQ(status, true);
  uint64_t output_size = CalTotalElements(shapes, 2);
  std::complex<double> output[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  data_path = ktestcaseFilePath + data_files[2];
  std::complex<double> output_exp[output_size];
  status = ReadDivFileComplex128(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);
  bool compare = CompareDivResultComplex128(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}


// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3>
void RunDivKernel(vector<string> data_files,
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
}
// only generate input data by SetRandomValue,
// and calculate output by youself function
template<typename T1, typename T2, typename T3>
void RunDivKernel2(vector<DataType> data_types,
                    vector<vector<int64_t>> &shapes) {
  // gen data use SetRandomValue for input1
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T1 *input1 = new T1[input1_size];
  SetRandomValue<T1>(input1, input1_size);
  // gen data use SetRandomValue for input2
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T2 *input2 = new T2[input2_size];
  SetRandomValue<T2>(input2, input2_size);
  uint64_t output_size = CalTotalElements(shapes, 2);
  T3 *output = new T3[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  // calculate output_exp
  T3 *output_exp = new T3[output_size];
  if(input1_size == input2_size) {
    CalcExpectWithSameShape<T1>(*node_def.get(), output_exp);
  } else {
    CalcExpectWithDiffShape<T1>(*node_def.get(), output_exp);
  }
  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 12}, {12}, {3, 12}};
  vector<string> files{"div/data/div_data_input1_1.txt",
                       "div/data/div_data_input2_1.txt",
                       "div/data/div_data_output1_1.txt"};
  RunDivKernel<int32_t, int32_t, int32_t>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{13, 10, 4}, {13, 10, 4}, {13, 10, 4}};
  vector<string> files{"div/data/div_data_input1_2.txt",
                       "div/data/div_data_input2_2.txt",
                       "div/data/div_data_output1_2.txt"};
  RunDivKernel<int64_t, int64_t, int64_t>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1024}, {8, 1024}, {8, 1024}};
  vector<string> files{"div/data/div_data_input1_3.txt",
                       "div/data/div_data_input2_3.txt",
                       "div/data/div_data_output1_3.txt"};
  RunDivKernel<float, float, float>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1024}, {4, 1024}, {4, 1024}};
  vector<string> files{"div/data/div_data_input1_4.txt",
                       "div/data/div_data_input2_4.txt",
                       "div/data/div_data_output1_4.txt"};
  RunDivKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT8, DT_INT8};
  vector<vector<int64_t>> shapes = {{7, 12}, {12}, {7, 12}};
  vector<string> files{"div/data/div_data_input1_5.txt",
                       "div/data/div_data_input2_5.txt",
                       "div/data/div_data_output1_5.txt"};
  RunDivKernel<int8_t, int8_t, int8_t>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_UINT8, DT_UINT8};
  vector<vector<int64_t>> shapes = {{7, 12}, {12}, {7, 12}};
  vector<string> files{"div/data/div_data_input1_6.txt",
                       "div/data/div_data_input2_6.txt",
                       "div/data/div_data_output1_6.txt"};
  RunDivKernel<uint8_t, uint8_t, uint8_t>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{1024, 8}, {8}, {1024, 8}};
  vector<string> files{"div/data/div_data_input1_7.txt",
                       "div/data/div_data_input2_7.txt",
                       "div/data/div_data_output1_7.txt"};
  RunDivKernel<int16_t, int16_t, int16_t>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{1024, 8}, {1024, 8}, {1024, 8}};
  vector<string> files{"div/data/div_data_input1_8.txt",
                       "div/data/div_data_input2_8.txt",
                       "div/data/div_data_output1_8.txt"};
  RunDivKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_FLOAT16_BCAST_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{12, 130}, {130}, {12, 130}};
  vector<string> files{"div/data/div_data_input1_16.txt",
                       "div/data/div_data_input2_16.txt",
                       "div/data/div_data_output1_16.txt"};
  RunDivKernel<Eigen::half, Eigen::half, Eigen::half>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{12, 6}, {6}, {12, 6}};
  vector<string> files{"div/data/div_data_input1_9.txt",
                       "div/data/div_data_input2_9.txt",
                       "div/data/div_data_output1_9.txt"};
  RunDivKernel<uint16_t, uint16_t, uint16_t>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{13, 10, 4}, {13, 10, 4}, {13, 10, 4}};
  vector<string> files{"div/data/div_data_input1_10.txt",
                       "div/data/div_data_input2_10.txt",
                       "div/data/div_data_output1_10.txt"};
   RunDivKernelComplex64(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{6, 6, 6}, {6, 6, 6}, {6, 6, 6}};
  vector<string> files{"div/data/div_data_input1_11.txt",
                       "div/data/div_data_input2_11.txt",
                       "div/data/div_data_output1_11.txt"};
  RunDivKernelComplex128(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_DOUBLE_DIFF_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1}, {3}, {3}};
  vector<string> files{"div/data/div_data_input1_12.txt",
                       "div/data/div_data_input2_12.txt",
                       "div/data/div_data_output1_12.txt"};
  RunDivKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_UINT16_DIFF_X_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{1}, {3}, {3}};
  vector<string> files{"div/data/div_data_input1_13.txt",
                       "div/data/div_data_input2_13.txt",
                       "div/data/div_data_output1_13.txt"};
  RunDivKernel<uint16_t, uint16_t, uint16_t>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_DOUBLE_DIFF_Y_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3}, {1}, {3}};
  vector<string> files{"div/data/div_data_input1_14.txt",
                       "div/data/div_data_input2_14.txt",
                       "div/data/div_data_output1_14.txt"};
  RunDivKernel<double, double, double>(files, data_types, shapes);
}

TEST_F(TEST_DIV_UT, DATA_TYPE_UINT16_DIFF_Y_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3}, {1}, {3}};
  vector<string> files{"div/data/div_data_input1_15.txt",
                       "div/data/div_data_input2_15.txt",
                       "div/data/div_data_output1_15.txt"};
  RunDivKernel<uint16_t, uint16_t, uint16_t>(files, data_types, shapes);
}

// exception instance
TEST_F(TEST_DIV_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 2, 4}, {2, 2, 3}, {2, 2, 4}};
  int32_t input1[12] = {(int32_t)1};
  int32_t input2[16] = {(int32_t)0};
  int32_t output[16] = {(int32_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DIV_UT, INPUT_ZERO_EXCEPTION) {
  vector<DataType> data_types = {DT_INT16, DT_INT16, DT_INT16};
  vector<vector<int64_t>> shapes = {{1}, {1}, {1}};
  int16_t input1[12] = {(int16_t)0};
  int16_t input2[16] = {(int16_t)0};
  int16_t output[16] = {(int16_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}
TEST_F(TEST_DIV_UT, INPUT_ZERO1_EXCEPTION) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{1024}, {8, 1024}, {8, 1024}};
  uint16_t input1[8192] = {(uint16_t)1};
  uint16_t input2[8192] = {(uint16_t)0};
  uint16_t output[8192] = {(uint16_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_DIV_UT, INPUT_ZERO2_EXCEPTION) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{1}, {1, 2}, {1, 2}};
  uint16_t input1[12] = {(uint16_t)1};
  uint16_t input2[16] = {(uint16_t)0};
  uint16_t output[16] = {(uint16_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_DIV_UT, INPUT_ZERO3_EXCEPTION) {
  vector<DataType> data_types = {DT_UINT16, DT_UINT16, DT_UINT16};
  vector<vector<int64_t>> shapes = {{1, 2}, {1}, {1, 2}};
  uint16_t input1[12] = {(uint16_t)1};
  uint16_t input2[16] = {(uint16_t)0};
  uint16_t output[16] = {(uint16_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_INNER_ERROR);
}

TEST_F(TEST_DIV_UT, INPUT_ZERO4_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1}, {1, 2}, {1, 2}};
  float_t input1[12] = {(float_t)1};
  float_t input2[16] = {(float_t)0};
  float_t output[16] = {(float_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_DIV_UT, INPUT_ZERO5_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1, 2}, {1}, {1, 2}};
  float_t input1[16] = {(float_t)1};
  float_t input2[12] = {(float_t)0};
  float_t output[16] = {(float_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_DIV_UT, INPUT_ZERO6_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2}, {2}, {2}};
  float_t input1[12] = {(float_t)1};
  float_t input2[16] = {(float_t)0};
  float_t output[16] = {(float_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_DIV_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t input1[22] = {(int32_t)1};
  int64_t input2[22] = {(int64_t)2};
  int64_t output[22] = {(int64_t)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DIV_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  int32_t output[22] = {(int32_t)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DIV_UT, INPUT_BOOL_UNSUPPORT) {
  vector<DataType> data_types = {DT_BOOL, DT_BOOL, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 11}, {2, 11}};
  bool input1[22] = {(bool)1};
  bool input2[22] = {(bool)0};
  bool output[22] = {(bool)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
