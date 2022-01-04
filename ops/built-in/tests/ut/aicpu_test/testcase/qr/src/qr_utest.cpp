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
#include "Eigen/QR"

using namespace std;
using namespace aicpu;

bool QrCompareResultfloat16(Eigen::half output[], Eigen::half expect_output[],
                            uint64_t num) {
  bool result = true;
  Eigen::half threshold = (Eigen::half)1;
  for (uint64_t i = 0; i < num; ++i) {
    Eigen::half value = output[i] - expect_output[i];
    if ((value > threshold) || (value < -threshold)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool QrReadFileComplexFloat(std::string file_name, std::complex<float> output[],
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

bool QrReadFileComplexDouble(std::string file_name, std::complex<double> output[],
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

class TEST_QR_UT : public testing::Test {};
#define CREATE_NODEDEF(shapes, data_types, datas, full_matrices)   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Qr", "Qr")                       \
      .Input({"x", data_types[0], shapes[0], datas[0]})            \
      .Output({"q", data_types[1], shapes[1], datas[1]})           \
      .Output({"r", data_types[2], shapes[2], datas[2]})           \
      .Attr("full_matrices", full_matrices)

TEST_F(TEST_QR_UT, INPUT_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{6}, {6}, {6}};
  float x[6] = {(float)1};
  float q[6] = {(float)1};
  float r[6] = {(float)1};
  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_QR_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_INT8, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 2}, {2, 11}};
  int8_t x[2 * 11] = {(int8_t)1};
  float q[2 * 2] = {(float)1};
  float r[2 * 11] = {(float)1};
  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_QR_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 11}, {2, 2}, {2, 11}};
  float q[2 * 2] = {(float)0};
  float r[2 * 11] = {(float)0};
  vector<void *> datas = {(void *)nullptr, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_QR_UT, DATA_TYPE_FLOAT_SUCC_2D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 4}, {3, 3}, {3, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_1.txt";
  constexpr uint64_t x_size = 3 * 4;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 3 * 3;
  float q[q_size] = {0};
  constexpr uint64_t r_size = 3 * 4;
  float r[r_size] = {0};

  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_1.txt";
  float q_exp[q_size] = {0};
  status = ReadFile(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_1.txt";
  float r_exp[r_size] = {0};
  status = ReadFile(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = CompareResult(q, q_exp, q_size);
  bool compare_r = CompareResult(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}

TEST_F(TEST_QR_UT, DATA_TYPE_FLOAT_SUCC_3D) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 3}, {2, 3, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_2.txt";
  constexpr uint64_t x_size = 2 * 3 * 4;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 2 * 3 * 3;
  float q[q_size] = {0};
  constexpr uint64_t r_size = 2 * 3 * 4;
  float r[r_size] = {0};

  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_2.txt";
  float q_exp[q_size] = {0};
  status = ReadFile(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_2.txt";
  float r_exp[r_size] = {0};
  status = ReadFile(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = CompareResult(q, q_exp, q_size);
  bool compare_r = CompareResult(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}

TEST_F(TEST_QR_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 3}, {2, 3, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_3.txt";
  constexpr uint64_t x_size = 2 * 3 * 4;
  Eigen::half x[x_size] = {(Eigen::half)0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 2 * 3 * 3;
  Eigen::half q[q_size] = {(Eigen::half)0};
  constexpr uint64_t r_size = 2 * 3 * 4;
  Eigen::half r[r_size] = {(Eigen::half)0};
  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_3.txt";
  Eigen::half q_exp[q_size] = {(Eigen::half)0};
  status = ReadFile(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_3.txt";
  Eigen::half r_exp[r_size] = {(Eigen::half)0};
  status = ReadFile(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = QrCompareResultfloat16(q, q_exp, q_size);
  bool compare_r = QrCompareResultfloat16(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}

TEST_F(TEST_QR_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 3}, {2, 3, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_4.txt";
  constexpr uint64_t x_size = 2 * 3 * 4;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 2 * 3 * 3;
  float q[q_size] = {0};
  constexpr uint64_t r_size = 2 * 3 * 4;
  float r[r_size] = {0};

  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_4.txt";
  float q_exp[q_size] = {0};
  status = ReadFile(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_4.txt";
  float r_exp[r_size] = {0};
  status = ReadFile(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = CompareResult(q, q_exp, q_size);
  bool compare_r = CompareResult(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}

TEST_F(TEST_QR_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{64, 32, 32}, {64, 32, 32}, {64, 32, 32}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_5.txt";
  constexpr uint64_t x_size = 64 * 32 * 32;
  double x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 64 * 32 * 32;
  double q[q_size] = {0};
  constexpr uint64_t r_size = 64 * 32 * 32;
  double r[r_size] = {0};

  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_5.txt";
  double q_exp[q_size] = {0};
  status = ReadFile(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_5.txt";
  double r_exp[r_size] = {0};
  status = ReadFile(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = CompareResult(q, q_exp, q_size);
  bool compare_r = CompareResult(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}

TEST_F(TEST_QR_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_COMPLEX64, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 3}, {2, 3, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_6.txt";
  constexpr uint64_t x_size = 2 * 3 * 4;
  std::complex<float> x[x_size] = {0};
  bool status = QrReadFileComplexFloat(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 2 * 3 * 3;
  std::complex<float> q[q_size] = {0};
  constexpr uint64_t r_size = 2 * 3 * 4;
  std::complex<float> r[r_size] = {0};

  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_6.txt";
  std::complex<float> q_exp[q_size] = {0};
  status = QrReadFileComplexFloat(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_6.txt";
  std::complex<float> r_exp[r_size] = {0};
  status = QrReadFileComplexFloat(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = CompareResult(q, q_exp, q_size);
  bool compare_r = CompareResult(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}

TEST_F(TEST_QR_UT, DATA_TYPE_COMPLEX128_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_COMPLEX128, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 3}, {2, 3, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_7.txt";
  constexpr uint64_t x_size = 2 * 3 * 4;
  std::complex<double> x[x_size] = {0};
  bool status = QrReadFileComplexDouble(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 2 * 3 * 3;
  std::complex<double> q[q_size] = {0};
  constexpr uint64_t r_size = 2 * 3 * 4;
  std::complex<double> r[r_size] = {0};

  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = false;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_7.txt";
  std::complex<double> q_exp[q_size] = {0};
  status = QrReadFileComplexDouble(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_7.txt";
  std::complex<double> r_exp[r_size] = {0};
  status = QrReadFileComplexDouble(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = CompareResult(q, q_exp, q_size);
  bool compare_r = CompareResult(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}

TEST_F(TEST_QR_UT, DATA_TYPE_DOUBLE_FULL_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{16, 32, 32}, {16, 32, 32}, {16, 32, 32}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_8.txt";
  constexpr uint64_t x_size = 16 * 32 * 32;
  double x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 16 * 32 * 32;
  double q[q_size] = {0};
  constexpr uint64_t r_size = 16 * 32 * 32;
  double r[r_size] = {0};

  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_8.txt";
  double q_exp[q_size] = {0};
  status = ReadFile(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_8.txt";
  double r_exp[r_size] = {0};
  status = ReadFile(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = CompareResult(q, q_exp, q_size);
  bool compare_r = CompareResult(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}

TEST_F(TEST_QR_UT, DATA_TYPE_FLOAT_FULL_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{2, 3, 4}, {2, 3, 3}, {2, 3, 4}};

  // read data from file for input
  string data_path = ktestcaseFilePath + "qr/data/qr_data_input1_9.txt";
  constexpr uint64_t x_size = 2 * 3 * 4;
  float x[x_size] = {0};
  bool status = ReadFile(data_path, x, x_size);
  EXPECT_EQ(status, true);

  constexpr uint64_t q_size = 2 * 3 * 3;
  float q[q_size] = {0};
  constexpr uint64_t r_size = 2 * 3 * 4;
  float r[r_size] = {0};

  vector<void *> datas = {(void *)x, (void *)q, (void *)r};
  bool full_matrices = true;
  CREATE_NODEDEF(shapes, data_types, datas, full_matrices);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + "qr/data/qr_data_output1_9.txt";
  float q_exp[q_size] = {0};
  status = ReadFile(data_path, q_exp, q_size);
  EXPECT_EQ(status, true);
  data_path = ktestcaseFilePath + "qr/data/qr_data_output2_9.txt";
  float r_exp[r_size] = {0};
  status = ReadFile(data_path, r_exp, r_size);
  EXPECT_EQ(status, true);

  bool compare_q = CompareResult(q, q_exp, q_size);
  bool compare_r = CompareResult(r, r_exp, r_size);
  EXPECT_EQ(compare_q && compare_r, true);
}