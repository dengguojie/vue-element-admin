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
#include "Eigen/LU"

using namespace std;
using namespace aicpu;

bool LuReadFileComplexFloat(std::string file_name, std::complex<float> output[],
                          uint64_t size) {
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

bool LuReadFileComplexDouble(std::string file_name, std::complex<double> output[],
                           uint64_t size) {
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

class TEST_LU_UT : public testing::Test {};
#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Lu", "Lu")                       \
      .Input({"input", data_types[0], shapes[0], datas[0]})        \
      .Output({"lu", data_types[1], shapes[1], datas[1]})          \
      .Output({"p", data_types[2], shapes[2], datas[2]})

TEST_F(TEST_LU_UT, INPUT_SHAPE_SIZE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{6}, {6}, {6}};
  float input[6] = {(float)1};
  float lu[6] = {(float)1};
  int32_t p[6] = {1};
  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LU_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{6, 5}, {6, 6}, {6}};
  float input[6] = {(float)1};
  float lu[6] = {(float)1};
  int32_t p[6] = {1};
  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LU_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT8, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{6, 6}, {6, 6}, {6}};
  int8_t input[6 * 6] = {(int8_t)1};
  float lu[6 * 6] = {(float)1};
  int32_t p[6] = {1};
  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LU_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{6, 6}, {6, 6}, {6}};
  float lu[6 * 6] = {(float)0};
  int32_t p[6] = {0};
  vector<void *> datas = {(void *)nullptr, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_LU_UT, DATA_TYPE_FLOAT_SUCC_2D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 3}, {3, 3}, {3}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "lu/data/lu_data_input1_1.txt";
  constexpr uint64_t input_size = 3 * 3;
  float input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t lu_size = 3 * 3;
  float lu[lu_size] = {0};
  constexpr uint64_t p_size = 3;
  int32_t p[p_size] = {0};

  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "lu/data/lu_data_output1_1.txt";
  float lu_exp[lu_size] = {0};
  status = ReadFile(data_path, lu_exp, lu_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "lu/data/lu_data_output2_1.txt";
  int32_t p_exp[p_size] = {0};
  status = ReadFile(data_path, p_exp, p_size);
  EXPECT_EQ(status, true);

  bool compare_lu = CompareResult(lu, lu_exp, lu_size);
  bool compare_p = CompareResult(p, p_exp, p_size);
  EXPECT_EQ(compare_lu && compare_p, true);
}

TEST_F(TEST_LU_UT, DATA_TYPE_FLOAT_SUCC_3D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4, 4}, {2, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "lu/data/lu_data_input1_2.txt";
  constexpr uint64_t input_size = 2 * 4 * 4;
  float input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t lu_size = 2 * 4 * 4;
  float lu[lu_size] = {0};
  constexpr uint64_t p_size = 2 * 4;
  int32_t p[p_size] = {0};

  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "lu/data/lu_data_output1_2.txt";
  float lu_exp[lu_size] = {0};
  status = ReadFile(data_path, lu_exp, lu_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "lu/data/lu_data_output2_2.txt";
  int32_t p_exp[p_size] = {0};
  status = ReadFile(data_path, p_exp, p_size);
  EXPECT_EQ(status, true);

  bool compare_lu = CompareResult(lu, lu_exp, lu_size);
  bool compare_p = CompareResult(p, p_exp, p_size);
  EXPECT_EQ(compare_lu && compare_p, true);
}

TEST_F(TEST_LU_UT, DATA_TYPE_FLOAT_SUCC_4D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 4}, {2, 3, 4, 4}, {2, 3, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "lu/data/lu_data_input1_3.txt";
  constexpr uint64_t input_size = 2 * 3 * 4 * 4;
  float input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t lu_size = 2 * 3 * 4 * 4;
  float lu[lu_size] = {0};
  constexpr uint64_t p_size = 2 * 3 * 4;
  int32_t p[p_size] = {0};

  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "lu/data/lu_data_output1_3.txt";
  float lu_exp[lu_size] = {0};
  status = ReadFile(data_path, lu_exp, lu_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "lu/data/lu_data_output2_3.txt";
  int32_t p_exp[p_size] = {0};
  status = ReadFile(data_path, p_exp, p_size);
  EXPECT_EQ(status, true);

  bool compare_lu = CompareResult(lu, lu_exp, lu_size);
  bool compare_p = CompareResult(p, p_exp, p_size);
  EXPECT_EQ(compare_lu && compare_p, true);
}

TEST_F(TEST_LU_UT, DATA_TYPE_DOUBLE_3D_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT32};
  vector<vector<int64_t>> shapes = {{8, 32, 32}, {8, 32, 32}, {8, 32}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "lu/data/lu_data_input1_4.txt";
  constexpr uint64_t input_size = 8 * 32 * 32;
  double input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t lu_size = 8 * 32 * 32;
  double lu[lu_size] = {0};
  constexpr uint64_t p_size = 8 * 32;
  int32_t p[p_size] = {0};

  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "lu/data/lu_data_output1_4.txt";
  double lu_exp[lu_size] = {0};
  status = ReadFile(data_path, lu_exp, lu_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "lu/data/lu_data_output2_4.txt";
  int32_t p_exp[p_size] = {0};
  status = ReadFile(data_path, p_exp, p_size);
  EXPECT_EQ(status, true);

  bool compare_lu = CompareResult(lu, lu_exp, lu_size);
  bool compare_p = CompareResult(p, p_exp, p_size);
  EXPECT_EQ(compare_lu && compare_p, true);
}

TEST_F(TEST_LU_UT, DATA_TYPE_DOUBLE_4D_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_INT32};
  vector<vector<int64_t>> shapes = {{8, 8, 32, 32}, {8, 8, 32, 32}, {8, 8, 32}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "lu/data/lu_data_input1_5.txt";
  constexpr uint64_t input_size = 8 * 8 * 32 * 32;
  double input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t lu_size = 8 * 8 * 32 * 32;
  double lu[lu_size] = {0};
  constexpr uint64_t p_size = 8 * 8 * 32;
  int32_t p[p_size] = {0};

  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "lu/data/lu_data_output1_5.txt";
  double lu_exp[lu_size] = {0};
  status = ReadFile(data_path, lu_exp, lu_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "lu/data/lu_data_output2_5.txt";
  int32_t p_exp[p_size] = {0};
  status = ReadFile(data_path, p_exp, p_size);
  EXPECT_EQ(status, true);

  bool compare_lu = CompareResult(lu, lu_exp, lu_size);
  bool compare_p = CompareResult(p, p_exp, p_size);
  EXPECT_EQ(compare_lu && compare_p, true);
}

TEST_F(TEST_LU_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4, 4}, {2, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "lu/data/lu_data_input1_6.txt";
  constexpr uint64_t input_size = 2 * 4 * 4;
  std::complex<float> input[input_size] = {0};
  bool status = LuReadFileComplexFloat(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t lu_size = 2 * 4 * 4;
  std::complex<float> lu[lu_size] = {0};
  constexpr uint64_t p_size = 2 * 4;
  int32_t p[p_size] = {0};

  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + "lu/data/lu_data_output1_6.txt";
  std::complex<float> lu_exp[lu_size] = {0};
  status = LuReadFileComplexFloat(data_path, lu_exp, lu_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "lu/data/lu_data_output2_6.txt";
  int32_t p_exp[p_size] = {0};
  status = ReadFile(data_path, p_exp, p_size);
  EXPECT_EQ(status, true);

  bool compare_lu = CompareResult(lu, lu_exp, lu_size);
  bool compare_p = CompareResult(p, p_exp, p_size);
  EXPECT_EQ(compare_lu && compare_p, true);
}

TEST_F(TEST_LU_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_INT32};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4, 4}, {2, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "lu/data/lu_data_input1_7.txt";
  constexpr uint64_t input_size = 2 * 4 * 4;
  std::complex<double> input[input_size] = {0};
  bool status = LuReadFileComplexDouble(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t lu_size = 2 * 4 * 4;
  std::complex<double> lu[lu_size] = {0};
  constexpr uint64_t p_size = 2 * 4;
  int32_t p[p_size] = {0};

  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  data_path = ktestcaseFilePath + "lu/data/lu_data_output1_7.txt";
  std::complex<double> lu_exp[lu_size] = {0};
  status = LuReadFileComplexDouble(data_path, lu_exp, lu_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "lu/data/lu_data_output2_7.txt";
  int32_t p_exp[p_size] = {0};
  status = ReadFile(data_path, p_exp, p_size);
  EXPECT_EQ(status, true);

  bool compare_lu = CompareResult(lu, lu_exp, lu_size);
  bool compare_p = CompareResult(p, p_exp, p_size);
  EXPECT_EQ(compare_lu && compare_p, true);
}

TEST_F(TEST_LU_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_INT64};
  vector<vector<int64_t>> shapes = {{2, 4, 4}, {2, 4, 4}, {2, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "lu/data/lu_data_input1_8.txt";
  constexpr uint64_t input_size = 2 * 4 * 4;
  float input[input_size] = {0};
  bool status = ReadFile(data_path, input, input_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t lu_size = 2 * 4 * 4;
  float lu[lu_size] = {0};
  constexpr uint64_t p_size = 2 * 4;
  int64_t p[p_size] = {0};

  vector<void *> datas = {(void *)input, (void *)lu, (void *)p};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "lu/data/lu_data_output1_8.txt";
  float lu_exp[lu_size] = {0};
  status = ReadFile(data_path, lu_exp, lu_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "lu/data/lu_data_output2_8.txt";
  int64_t p_exp[p_size] = {0};
  status = ReadFile(data_path, p_exp, p_size);
  EXPECT_EQ(status, true);

  bool compare_lu = CompareResult(lu, lu_exp, lu_size);
  bool compare_p = CompareResult(p, p_exp, p_size);
  EXPECT_EQ(compare_lu && compare_p, true);
}