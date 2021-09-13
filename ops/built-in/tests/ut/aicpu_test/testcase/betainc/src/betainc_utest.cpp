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
#include <unistd.h>
#include <iostream>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

namespace {

enum CASE {NO_SCALAR, A_SCALAR, B_SCALAR, ALL_SCALAR};

template <typename T>
bool CompareRealResult(const T& output, const T& expect_output,
                       const double& rel_error, int line_num, double& err,
                       double& err_zero_point, int& err_pos) {
  bool res = true;
  double curr_err = 0;

  if (isnan(T(expect_output)) && isnan(T(output))) {
    res = false;
  } else if (!isnan(T(expect_output)) && !isnan(T(output))) {
    if (expect_output == T(0)) {
      curr_err = static_cast<double>(abs(T(output - expect_output)));
      res = false;
      err_zero_point = (curr_err > err_zero_point ? curr_err : err_zero_point);
    } else if (isinf(expect_output)) {
      res = !isinf(output) || (expect_output * output < T(0));
    } else {
      curr_err =
          static_cast<double>(abs(T(expect_output - output) / expect_output));
      res = false;
      if (abs(double(output - expect_output)) >
          abs(rel_error * static_cast<double>(expect_output))) {
        err_pos = curr_err > err ? line_num : err_pos;
      }
      err = curr_err > err ? curr_err : err;
    }
  } else {
    res = true;
  }

  return res;
}

template <typename T>
bool MyReadFile(std::string file_name, T output[], int size) {
  file_name = ktestcaseFilePath + file_name;
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    string tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }
      try {
        output[index] = T(stod(tmp));
      } catch (std::exception &e) {
        output[index] = T(stold(tmp));
      }

      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

}  // namespace

class TEST_BETAINC_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas)                  \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "Betainc", "Betainc")             \
      .Input({"a", data_types[0], shapes[0], datas[0]})            \
      .Input({"b", data_types[1], shapes[1], datas[1]})            \
      .Input({"x", data_types[2], shapes[2], datas[2]})            \
      .Output({"z", data_types[3], shapes[3], datas[3]})

template <typename T>
void RunBetainc(vector<string> &data_files, vector<DataType> &data_types,
                vector<vector<int64_t>> &shapes,
                KernelStatus kernel_status = KERNEL_STATUS_OK, CASE c = NO_SCALAR) {
  bool status;

  int input1_size, input2_size, input3_size, output_size;
  T const_data_a = 0.5;
  T const_data_b = 0.5;
  T const_data_x = 0.5;
  T *input1 = nullptr;
  T *input2 = nullptr;
  T *input3 = nullptr;

  if( c == A_SCALAR ){
    input1 = &const_data_a;
    cout << "input1 = " << *input1 << endl;
  }
  else{
    input1_size = CalTotalElements(shapes, 0);
    input1 = new T[input1_size];
    status = MyReadFile<T>(data_files[0], input1, input1_size);
    EXPECT_EQ(status, true);
  }

  if (c == B_SCALAR){
    input2 = &const_data_b;
    input3 = &const_data_x;
  }
  else{
    input2_size = CalTotalElements(shapes, 1);
    input2 = new T[input2_size];
    status = MyReadFile<T>(data_files[1], input2, input2_size);
    EXPECT_EQ(status, true);

    input3_size = CalTotalElements(shapes, 2);
    input3 = new T[input3_size];
    status = MyReadFile<T>(data_files[2], input3, input3_size);
    EXPECT_EQ(status, true);
  }

  output_size = CalTotalElements(shapes, 3);

  T *output = new T[output_size];

  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  //自研算子执行函数
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, kernel_status);

  T *output_exp = nullptr;
  // read data from file for expect ouput
  if (kernel_status == KERNEL_STATUS_OK) {
    
    output_exp = new T[output_size];
    status = MyReadFile<T>(data_files[3], output_exp, output_size);
    EXPECT_EQ(status, true);

    double err = 0;
    double err_zero_point = 0;
    int err_pos = 0;

    bool compare = true;

    cout << "running compare" << endl;
    for (int i = 0; i < output_size; i++) {
      compare = CompareRealResult<T>(*(output + i), *(output_exp + i), 0.001, i,
                                    err, err_zero_point, err_pos);
    }

    cout.precision(20);
    cout << "max relative err: " << fixed << err
         << " max err at zero point: " << fixed << err_zero_point << endl;
    cout << "intput = " << *input1 << " " << input2[int(err_pos)]
         << " " << input3[int(err_pos)] << endl;
    cout << "output = " << *(output + int(err_pos)) << " "
         << *(output_exp + int(err_pos)) << endl;
  }

  if(nullptr != input1 && c != A_SCALAR){
    delete[] input1;
  }
  if(nullptr != input2 && c != B_SCALAR){
    delete[] input2;
  }
  if(nullptr != input3 && c != B_SCALAR){
    delete[] input3;
  }
  if(nullptr != output){
    delete[] output;
  }
  if (output_exp != nullptr) {
    delete[] output_exp;
  }
}

template <typename T>
void RunBetainc_allscalar(T a, T b, T x, T expected_z,
                          vector<DataType> &data_types,
                          KernelStatus kernel_status = KERNEL_STATUS_OK) {
  bool status;
  int input1_size, input2_size, input3_size;

  T *input1 = &a;
  T *input2 = &b;
  T *input3 = &x;

  T output_data = 0;
  T *output = &output_data;

  vector<void *> datas = {(void *)input1, (void *)input2, (void *)input3,
                          (void *)output};
  vector<vector<int64_t>> shapes = {{}, {}, {}, {}};
  //自研算子执行函数
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, kernel_status);

  T *output_exp = &expected_z;
  // read data from file for expect ouput
  if (kernel_status == KERNEL_STATUS_OK) {
    double err = 0;
    double err_zero_point = 0;
    int err_pos = 0;

    bool compare = true;

    cout << "running compare" << endl;

    compare = CompareRealResult<T>(*(output), *(output_exp), 0.001, 0,
                                  err, err_zero_point, err_pos);
    
    cout << "max err: " << err << " at pos: " << err_pos << endl;
  }
}

TEST_F(TEST_BETAINC_UT, succ_float) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1600,},{1600,},{1600,},{1600,}};
  vector<string> files{
      "betainc/data/betainc_input1_1.txt", "betainc/data/betainc_input2_1.txt",
      "betainc/data/betainc_input3_1.txt", "betainc/data/betainc_output_1.txt"};
  RunBetainc<float>(files, data_types, shapes);
}

TEST_F(TEST_BETAINC_UT, succ_double) {
  vector<DataType> data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{1600,},{1600,},{1600,},{1600,}};
  vector<string> files{
      "betainc/data/betainc_input1_1.txt", "betainc/data/betainc_input2_1.txt",
      "betainc/data/betainc_input3_1.txt", "betainc/data/betainc_output_1.txt"};
  RunBetainc<double>(files, data_types, shapes);
}


TEST_F(TEST_BETAINC_UT, succ_scalarbcast_a) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{},{1600,},{1600,},{1600,}};
  vector<string> files{"betainc/data/betainc_input1_1.txt",
                       "betainc/data/betainc_input2_1.txt",
                       "betainc/data/betainc_input3_1.txt",
                       "betainc/data/betainc_output_ascalar_1.txt"};
  RunBetainc<float>(files, data_types, shapes, KERNEL_STATUS_OK, A_SCALAR);
}

TEST_F(TEST_BETAINC_UT, succ_scalarbcast_bc) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1600,},{},{},{1600,}};
  vector<string> files{"betainc/data/betainc_input1_1.txt",
                       "betainc/data/betainc_input2_1.txt",
                       "betainc/data/betainc_input3_1.txt",
                       "betainc/data/betainc_output_bcscalar_1.txt"};
  RunBetainc<float>(files, data_types, shapes, KERNEL_STATUS_OK, B_SCALAR);
}

TEST_F(TEST_BETAINC_UT, fail_shape) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1600,},{1600,},{500,},{1600,}};
  vector<string> files{
      "betainc/data/betainc_input1_1.txt", "betainc/data/betainc_input2_1.txt",
      "betainc/data/betainc_input3_1.txt", "betainc/data/betainc_output_1.txt"};
  RunBetainc<float>(files, data_types, shapes, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BETAINC_UT, fail_dtype) {
  vector<DataType> data_types = {DT_FLOAT, DT_DOUBLE, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{1600,},{1600,},{1600,},{1600,}};
  vector<string> files{
      "betainc/data/betainc_input1_1.txt", "betainc/data/betainc_input2_1.txt",
      "betainc/data/betainc_input3_1.txt", "betainc/data/betainc_output_1.txt"};
  RunBetainc<float>(files, data_types, shapes, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_BETAINC_UT, succ_allscalar) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT};
  RunBetainc_allscalar<float>(0.5, 0.014409, 0.027, 0.004685046, data_types);
}
