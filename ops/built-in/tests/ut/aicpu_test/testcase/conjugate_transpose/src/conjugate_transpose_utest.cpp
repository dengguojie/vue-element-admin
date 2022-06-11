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

class TEST_CONJUGATETRANSPOSE_UT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "conjugate_transpose test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "conjugate_transpose TearDown" << std::endl;
  }
};

#define CREATE_NODEDEF(shapes, data_types, datas)                            \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();           \
  NodeDefBuilder(node_def.get(), "ConjugateTranspose", "ConjugateTranspose") \
      .Input({"x", data_types[0], shapes[0], datas[0]})                      \
      .Input({"perm", data_types[1], shapes[1], datas[1]})                   \
      .Output({"y", data_types[2], shapes[2], datas[2]})

template <typename T1, typename T2>
void RunConjugateTransposeKernel(vector<string> data_files, vector<DataType> data_types,
                                 vector<vector<int64_t>>& shapes, T2 input1[]) {
  // read data from file for input0
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input0_size = CalTotalElements(shapes, 0);
  T1* input0 = new T1[input0_size];
  bool status = ReadFile(data_path, input0, input0_size);
  EXPECT_EQ(status, true);

  uint64_t input1_size = CalTotalElements(shapes, 1);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T1* output = new T1[output_size];
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect output
  data_path = ktestcaseFilePath + data_files[1];
  T1* output_exp = new T1[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);

  delete[] input0;
  delete[] output;
  delete[] output_exp;
}


TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_BOOL_SUCC) {
  vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
  vector<vector<int64_t>> shapes = {{2, 3}, {2}, {3, 2}};
  bool input0[2 * 3] = {false, false, false, true, true, true};
  int32_t input1[2] = {1, 0};
  uint64_t output_size = 3 * 2;
  bool output_exp[3 * 2] = {false, true, false, true, false, true};
  bool output[3 * 2];

  vector<void *> datas = {(void *)input0, (void *)input1, (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_DOUBLE_SUCC) {
  vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int32_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_0.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_0.txt",
  };
  RunConjugateTransposeKernel<double, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_UINT8_SUCC) {
  vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_UINT8};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int32_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_1.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_1.txt",
  };
  RunConjugateTransposeKernel<uint8_t, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_UINT16_SUCC) {
  vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_UINT16};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int32_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_2.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_2.txt",
  };
  RunConjugateTransposeKernel<uint16_t, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT8_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int32_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_5.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_5.txt",
  };
  RunConjugateTransposeKernel<int8_t, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT16_SUCC) {
  vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int32_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_6.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_6.txt",
  };
  RunConjugateTransposeKernel<int16_t, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT32_SUCC) {
  vector<DataType> data_types = {DT_INT32, DT_INT64, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int64_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_7.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_7.txt",
  };
  RunConjugateTransposeKernel<int32_t, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT64_SUCC) {
  vector<DataType> data_types = {DT_INT64, DT_INT64, DT_INT64};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int64_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_8.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_8.txt",
  };
  RunConjugateTransposeKernel<int64_t, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_FLOAT_SUCC) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int32_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_9.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_9.txt",
  };
  RunConjugateTransposeKernel<float, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_FLOAT16_SUCC) {
  vector<DataType> data_types = {DT_FLOAT16, DT_INT64, DT_FLOAT16};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int64_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_10.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_10.txt",
  };
  RunConjugateTransposeKernel<Eigen::half, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int32_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_11.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_11.txt",
  };
  RunConjugateTransposeKernel<std::complex<float>, int32_t>(files, data_types, shapes,
                                                            input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX128__SUCC) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int64_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_12.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_12.txt",
  };
  RunConjugateTransposeKernel<std::complex<double>, int64_t>(files, data_types, shapes,
                                                             input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT8_4D_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5}, {4}, {2, 3, 5, 4}};
  int32_t input1[4] = {0, 1, 3, 2};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_13.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_13.txt",
  };
  RunConjugateTransposeKernel<int8_t, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT8_5D_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5, 6}, {5}, {2, 3, 5, 4, 6}};
  int32_t input1[5] = {0, 1, 3, 2, 4};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_14.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_14.txt",
  };
  RunConjugateTransposeKernel<int8_t, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT8_6D_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {
      {2, 3, 4, 5, 6, 7}, {6}, {2, 3, 5, 4, 6, 7}};
  int32_t input1[6] = {0, 1, 3, 2, 4, 5};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_15.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_15.txt",
  };
  RunConjugateTransposeKernel<int8_t, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_INT8_7D_SUCC) {
  vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
  vector<vector<int64_t>> shapes = {
      {2, 3, 4, 5, 6, 7, 8}, {7}, {2, 3, 5, 4, 6, 7, 8}};
  int32_t input1[7] = {0, 1, 3, 2, 4, 5, 6};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_16.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_16.txt",
  };
  RunConjugateTransposeKernel<int8_t, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC_2D) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2, 2}};
  int32_t input1[2] = {0, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_17.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_17.txt",
  };
  RunConjugateTransposeKernel<std::complex<float>, int32_t>(files, data_types, shapes, input1);
}


TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC_3D_LARGE) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{256, 2, 1024}, {3}, {256, 2, 1024}};
  int32_t input1[3] = {0, 1, 2};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_19.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_19.txt",
  };
  RunConjugateTransposeKernel<std::complex<float>, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX128_SUCC_3D_LARGE) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{256, 2, 1024}, {3}, {1024, 2, 256}};
  int64_t input1[3] = {2, 1, 0};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_20.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_20.txt",
  };
  RunConjugateTransposeKernel<std::complex<double>, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC_3D) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int32_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_21.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_21.txt",
  };
  RunConjugateTransposeKernel<std::complex<float>, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX128_SUCC_3D) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};
  int64_t input1[3] = {0, 2, 1};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_22.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_22.txt",
  };
  RunConjugateTransposeKernel<std::complex<double>, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC_4D) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5}, {4}, {2, 3, 5, 4}};
  int32_t input1[4] = {0, 1, 3, 2};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_23.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_23.txt",
  };
  RunConjugateTransposeKernel<std::complex<float>, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX128_SUCC_4D) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5}, {4}, {2, 3, 5, 4}};
  int64_t input1[4] = {0, 1, 3, 2};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_24.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_24.txt",
  };
  RunConjugateTransposeKernel<std::complex<double>, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC_5D) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5, 6}, {5}, {2, 3, 5, 4, 6}};
  int32_t input1[5] = {0, 1, 3, 2, 4};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_25.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_25.txt",
  };
  RunConjugateTransposeKernel<std::complex<float>, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX128_SUCC_5D) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5, 6}, {5}, {2, 3, 5, 4, 6}};
  int64_t input1[5] = {0, 1, 3, 2, 4};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_26.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_26.txt",
  };
  RunConjugateTransposeKernel<std::complex<double>, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC_6D) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5, 6, 7}, {6}, {2, 3, 5, 4, 6, 7}};
  int32_t input1[6] = {0, 1, 3, 2, 4, 5};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_27.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_27.txt",
  };
  RunConjugateTransposeKernel<std::complex<float>, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX128_SUCC_6D) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5, 6, 7}, {6}, {2, 3, 5, 4, 6, 7}};
  int64_t input1[6] = {0, 1, 3, 2, 4, 5};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_28.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_28.txt",
  };
  RunConjugateTransposeKernel<std::complex<double>, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX64_SUCC_7D) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5, 6, 7, 8}, {7}, {2, 3, 5, 4, 6, 7, 8}};
  int32_t input1[7] = {0, 1, 3, 2, 4, 5, 6};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_29.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_29.txt",
  };
  RunConjugateTransposeKernel<std::complex<float>, int32_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, DATA_TYPE_COMPLEX128_SUCC_7D) {
  vector<DataType> data_types = {DT_COMPLEX128, DT_INT64, DT_COMPLEX128};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5, 6, 7, 8}, {7}, {2, 3, 5, 4, 6, 7, 8}};
  int64_t input1[7] = {0, 1, 3, 2, 4, 5, 6};
  vector<string> files{
      "conjugate_transpose/data/conjugate_transpose_data_input_30.txt",
      "conjugate_transpose/data/conjugate_transpose_data_output_30.txt",
  };
  RunConjugateTransposeKernel<std::complex<double>, int64_t>(files, data_types, shapes, input1);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, INPUT_1D_EXCEPTION) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{5}, {1}, {5}};
  int8_t input0[5] = {1};
  int8_t input1[1] = {0};
  int8_t output[5] = {1};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};
  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, INPUT_DIM_EXCEPTION) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{2, 3, 4, 5, 6, 7, 8}, {8}, {3, 2, 4, 5, 6, 7, 8, 9}};

  constexpr uint64_t input0_size = 2 * 3 * 4 * 5 * 6 * 7 * 8;
  int32_t input0[input0_size] = {0};
  constexpr uint64_t input1_size = 8;
  int32_t input1[input1_size] = {1, 0, 2, 3, 4, 5, 6, 7};
  constexpr uint64_t output_size = 3 * 2 * 4 * 5 * 6 * 7 * 8 * 9;
  int32_t output[output_size] = {0};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, INPUT_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_STRING, DT_INT32, DT_INT32};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};

  constexpr uint64_t input0_size = 3 * 2 * 3;
  int32_t input0[input0_size] = {0};
  constexpr uint64_t input1_size = 3;
  int32_t input1[input1_size] = {0, 2, 1};
  constexpr uint64_t output_size = 3 * 3 * 2;
  int32_t output[output_size] = {0};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, PERM_DTYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_FLOAT16, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};

  constexpr uint64_t input0_size = 3 * 2 * 3;
  int32_t input0[input0_size] = {0};
  constexpr uint64_t input1_size = 3;
  float_t input1[input1_size] = {0, 2, 1};
  constexpr uint64_t output_size = 3 * 3 * 2;
  int32_t output[output_size] = {0};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, PERM_SHAPE_EXCEPTION) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {4}, {3, 3, 2}};

  constexpr uint64_t input0_size = 3 * 2 * 3;
  int32_t input0[input0_size] = {0};
  constexpr uint64_t input1_size = 4;
  int32_t input1[input1_size] = {0, 2, 1, 3};
  constexpr uint64_t output_size = 3 * 3 * 2;
  int32_t output[output_size] = {0};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, PERM_RANGE_EXCEPTION) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};

  constexpr uint64_t input0_size = 3 * 2 * 3;
  int32_t input0[input0_size] = {0};
  constexpr uint64_t input1_size = 3;
  int32_t input1[input1_size] = {0, 3, 1};
  constexpr uint64_t output_size = 3 * 3 * 2;
  int32_t output[output_size] = {0};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CONJUGATETRANSPOSE_UT, PERM_NOT_UNIQUE_EXCEPTION) {
  vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
  vector<vector<int64_t>> shapes = {{3, 2, 3}, {3}, {3, 3, 2}};

  constexpr uint64_t input0_size = 3 * 2 * 3;
  int32_t input0[input0_size] = {0};
  constexpr uint64_t input1_size = 3;
  int32_t input1[input1_size] = {0, 2, 2};
  constexpr uint64_t output_size = 3 * 3 * 2;
  int32_t output[output_size] = {0};
  vector<void*> datas = {(void*)input0, (void*)input1, (void*)output};

  CREATE_NODEDEF(shapes, data_types, datas);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
