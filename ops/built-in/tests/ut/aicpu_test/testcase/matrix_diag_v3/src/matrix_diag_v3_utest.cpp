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

class TEST_MATRIX_DIAG_V3_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, align)                \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();      \
  NodeDefBuilder(node_def.get(), "MatrixDiagV3",                        \
                 "MatrixDiagV3")                                        \
      .Input({"x", data_types[0], shapes[0], datas[0]})                 \
      .Input({"k", data_types[1], shapes[1], datas[1]})                 \
      .Input({"num_rows", data_types[2], shapes[2], datas[2]})          \
      .Input({"num_cols", data_types[3], shapes[3], datas[3]})          \
      .Input({"padding_value", data_types[4], shapes[4], datas[4]})     \
      .Output({"y", data_types[5], shapes[5], datas[5]})                \
      .Attr("align", align)

// read input and output data from files which generate by your python file
template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void RunMatrixDiagV3Kernel(vector<string> data_files,
vector<DataType> data_types,
vector<vector<int64_t>> &shapes,string &alignvalue = "RIGHT_LEFT") {
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

    // read data from file for input3
    data_path = ktestcaseFilePath + data_files[2];
    uint64_t input3_size = CalTotalElements(shapes, 2);
    T3 *input3 = new T3[input3_size];
    status = ReadFile(data_path, input3, input3_size);
    EXPECT_EQ(status, true);

    // read data from file for input4
    data_path = ktestcaseFilePath + data_files[3];
    uint64_t input4_size = CalTotalElements(shapes, 3);
    T4 *input4 = new T4[input4_size];
    status = ReadFile(data_path, input4, input4_size);
    EXPECT_EQ(status, true);

    // read data from file for input5
    data_path = ktestcaseFilePath + data_files[4];
    uint64_t input5_size = CalTotalElements(shapes, 4);
    T5 *input5 = new T5[input5_size];
    status = ReadFile(data_path, input5, input5_size);
    EXPECT_EQ(status, true);


    uint64_t output_size = CalTotalElements(shapes, 5);
    T6 *output = new T6[output_size];
    vector<void *> datas = {(void *)input1,
    (void *)input2,
    (void *)input3,
    (void *)input4,
    (void *)input5,
    (void *)output};

    string align = alignvalue;
    CREATE_NODEDEF(shapes, data_types, datas, align);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // read data from file for expect ouput
    data_path = ktestcaseFilePath + data_files[5];
    T6 *output_exp = new T6[output_size];
    //status = ReadFile(data_path, output_exp, output_size);
    //EXPECT_EQ(status, true);

    //bool compare = CompareResult(output, output_exp, output_size);
    //std::cout<<"MatrixDiagV3 output = "<<output<<std::endl;
    //std::cout<<"MatrixDiagV3 output_exp = "<<output_exp<<std::endl;
    //EXPECT_EQ(compare, true);
    delete [] input1;
    delete [] input2;
    delete [] input3;
    delete [] input4;
    delete [] input5;
    delete [] output;
    delete [] output_exp;
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_INT32_SUCCESS) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2,3,3}, {2}, {1}, {1}, {1}, {2, 3, 3}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_1.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_1.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_1.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_1.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_1.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_1.txt"};
    string align = "RIGHT_LEFT";
    RunMatrixDiagV3Kernel<int32_t, int32_t,  int32_t, int32_t, int32_t, int32_t>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_FLOAT_SUCCESS) {
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2,3,3}, {2}, {1}, {1}, {1}, {2, 3, 3}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_2.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_2.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_2.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_2.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_2.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_2.txt"};
    string align = "RIGHT_LEFT";
    RunMatrixDiagV3Kernel<float, int32_t,  int32_t, int32_t, float, float>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_DOUBLE_SUCCESS) {
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_INT32, DT_INT32, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2,3,3}, {2}, {1}, {1}, {1}, {2, 3, 3}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_3.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_3.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_3.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_3.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_3.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_3.txt"};
    string align = "RIGHT_LEFT";
    RunMatrixDiagV3Kernel<double, int32_t,  int32_t, int32_t, double, double>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_DOUBLE_32K_SUCCESS) {
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_INT32, DT_INT32, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{8,8,8,8}, {2}, {1}, {1}, {1}, {8,8,8,8}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_13.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_13.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_13.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_13.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_13.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_13.txt"};
    string align = "RIGHT_LEFT";
    RunMatrixDiagV3Kernel<double, int32_t,  int32_t, int32_t, double, double>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_INT64_SUCCESS) {
    vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT32, DT_INT32, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2,3,3}, {2}, {1}, {1}, {1}, {2, 3, 3}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_4.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_4.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_4.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_4.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_4.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_4.txt"};
    string align = "RIGHT_LEFT";
    RunMatrixDiagV3Kernel<int64_t, int32_t,  int32_t, int32_t, int64_t, int64_t>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_COMPLEX64_SUCCESS) {
    vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_INT32, DT_INT32, DT_COMPLEX64, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{2,3,3}, {2}, {1}, {1}, {1}, {2, 3, 3}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_5.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_5.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_5.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_5.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_5.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_5.txt"};
    string align = "RIGHT_LEFT";
    RunMatrixDiagV3Kernel<std::complex<std::float_t>, int32_t,  int32_t, int32_t, std::complex<std::float_t>, std::complex<std::float_t>>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_FLOAT16_SUCCESS) {
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{2,3,3}, {2}, {1}, {1}, {1}, {2, 3, 3}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_6.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_6.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_6.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_6.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_6.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_6.txt"};
    string align = "LEFT_RIGHT";
    RunMatrixDiagV3Kernel<Eigen::half, int32_t,  int32_t, int32_t, Eigen::half, Eigen::half>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_UINT8_SUCCESS) {
    vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_INT32, DT_INT32, DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{2,3,4}, {1}, {1}, {1}, {1}, {2, 3, 5, 5}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_7.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_7.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_7.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_7.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_7.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_7.txt"};
    string align = "LEFT_RIGHT";
    RunMatrixDiagV3Kernel<uint8_t, int32_t,  int32_t, int32_t, uint8_t, uint8_t>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_UINT16_SUCCESS) {
    vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_INT32, DT_INT32, DT_UINT16, DT_UINT16};
    vector<vector<int64_t>> shapes = {{2,3,4}, {1}, {1}, {1}, {1}, {2, 3, 5, 5}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_8.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_8.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_8.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_8.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_8.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_8.txt"};
    string align = "LEFT_RIGHT";
    RunMatrixDiagV3Kernel<uint16_t, int32_t,  int32_t, int32_t, uint16_t, uint16_t>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_UINT32_SUCCESS) {
    vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_INT32, DT_INT32, DT_UINT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{2,3,4}, {1}, {1}, {1}, {1}, {2, 3, 5, 5}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_9.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_9.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_9.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_9.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_9.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_9.txt"};
    string align = "RIGHT_RIGHT";
    RunMatrixDiagV3Kernel<uint32_t, int32_t,  int32_t, int32_t, uint32_t, uint32_t>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_UINT64_SUCCESS) {
    vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_INT32, DT_INT32, DT_UINT64, DT_UINT64};
    vector<vector<int64_t>> shapes = {{2,3,4}, {1}, {1}, {1}, {1}, {2, 3, 5, 5}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_10.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_10.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_10.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_10.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_10.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_10.txt"};
    string align = "LEFT_LEFT";
    RunMatrixDiagV3Kernel<uint64_t, int32_t,  int32_t, int32_t, uint64_t, uint64_t>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_COMPLEX128_SUCCESS) {
    vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_INT32, DT_INT32, DT_COMPLEX128, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{2,3,3}, {2}, {1}, {1}, {1}, {2, 3, 3}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_11.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_11.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_11.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_11.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_11.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_11.txt"};
    string align = "RIGHT_LEFT";
    RunMatrixDiagV3Kernel<std::complex<std::double_t>, int32_t,  int32_t, int32_t, std::complex<std::double_t>, std::complex<std::double_t>>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DT_INT8_SUCCESS) {
    vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT32, DT_INT32, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{2,3,4}, {1}, {1}, {1}, {1}, {2, 3, 5, 5}};
    vector<string> files{"matrix_diag_v3/data/matrix_diag_v3_data_input1_12.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input2_12.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input3_12.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input4_12.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_input5_12.txt",
    "matrix_diag_v3/data/matrix_diag_v3_data_output1_12.txt"};
    string align = "LEFT_LEFT";
    RunMatrixDiagV3Kernel<int8_t, int32_t,  int32_t, int32_t, int8_t, int8_t>(files, data_types, shapes, align);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Int32_Success) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {1}, {1},{1},{1},{3,2}};
    int32_t input0[2] = {1,2};
    int32_t input1[1] = {-1};
    int32_t input2[1] = {3};
    int32_t input3[1] = {2};
    int32_t input4[1] = {9};

    int32_t output_exp[6] = {9,9,1,9,9,2};
    uint64_t output_size = 6;
    int32_t *output = new int32_t[output_size];
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool compare = CompareResult(output, output_exp, output_size);
    std::cout<<"MatrixDiagV3UT output = "<<output<<std::endl;
    std::cout<<"MatrixDiagV3UT output_exp = "<<output_exp<<std::endl;
    EXPECT_EQ(compare, true);

    delete [] output;
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, FLOAT_Success) {
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2}, {1}, {1},{1},{1},{3,2}};
    float input0[2] = {1.1,2.2};
    int32_t input1[1] = {-1};
    int32_t input2[1] = {3};
    int32_t input3[1] = {2};
    float input4[1] = {9.9};

    float output_exp[6] = {9.9,9.9,1.1,9.9,9.9,2.2};
    uint64_t output_size = 6;
    float *output = new float[output_size];
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool compare = CompareResult(output, output_exp, output_size);
    std::cout<<"MatrixDiagV3UT output = "<<output<<std::endl;
    std::cout<<"MatrixDiagV3UT output_exp = "<<output_exp<<std::endl;
    EXPECT_EQ(compare, true);

    delete [] output;
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, DOUBLE_Success) {
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_INT32, DT_INT32, DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2}, {1}, {1},{1},{1},{3,2}};
    double input0[2] = {1.1,2.2};
    int32_t input1[1] = {-1};
    int32_t input2[1] = {3};
    int32_t input3[1] = {2};
    double input4[1] = {9.9};

    double output_exp[6] = {9.9,9.9,1.1,9.9,9.9,2.2};
    uint64_t output_size = 6;
    double *output = new double[output_size];
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    bool compare = CompareResult(output, output_exp, output_size);
    std::cout<<"MatrixDiagV3UT output = "<<output<<std::endl;
    std::cout<<"MatrixDiagV3UT output_exp = "<<output_exp<<std::endl;
    EXPECT_EQ(compare, true);

    delete [] output;
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Input_Bool) {
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_INT32, DT_INT32, DT_BOOL, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {1}, {1},{1},{1},{3,2}};
    bool input0[2] = {(bool)1,(bool)2};
    int32_t input1[1] = {-1};
    int32_t input2[1] = {3};
    int32_t input3[1] = {2};
    bool input4[1] = {(bool)9};

    int32_t output[6] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Dtype) {
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT64};
    vector<vector<int64_t>> shapes = {{2}, {1}, {1},{1},{1},{3,3}};
    int16_t input0[2] = {1,2};
    int32_t input1[3] = {-1};
    int32_t input2[1] = {3};
    int32_t input3[1] = {3};
    int16_t input4[1] = {9};
    
    int16_t output[6] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}


TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Shape_K) {
    vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT32, DT_INT32, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2}, {3}, {1},{1},{1},{3,2}};
    int16_t input0[2] = {1,2};
    int32_t input1[3] = {-1,1,2};
    int32_t input2[1] = {3};
    int32_t input3[1] = {2};
    int16_t input4[1] = {9};
    
    int16_t output[6] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Shape_Num_Rows) {
    vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT32, DT_INT32, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2}, {2}, {2},{1},{1},{3,2}};
    int16_t input0[2] = {1,2};
    int32_t input1[2] = {-1,1};
    int32_t input2[2] = {3,4};
    int32_t input3[1] = {2};
    int16_t input4[1] = {9};
    
    int16_t output[6] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Shape_Num_Cols) {
    vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT32, DT_INT32, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2}, {2}, {1},{2},{1},{3,2}};
    int16_t input0[2] = {1,2};
    int32_t input1[2] = {-1,1};
    int32_t input2[1] = {3};
    int32_t input3[2] = {2,3};
    int16_t input4[1] = {9};
    
    int16_t output[6] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Shape_Padding_Value) {
    vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT32, DT_INT32, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2}, {2}, {1},{1},{2},{3,2}};
    int16_t input0[2] = {1,2};
    int32_t input1[2] = {-1,1};
    int32_t input2[1] = {3};
    int32_t input3[1] = {2};
    int16_t input4[2] = {8,9};
    
    int16_t output[6] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Value_K) {
    vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT32, DT_INT32, DT_INT16, DT_INT16};
    vector<vector<int64_t>> shapes = {{2}, {2}, {1},{1},{2},{3,2}};
    int16_t input0[2] = {1,2};
    int32_t input1[2] = {3,2};
    int32_t input2[1] = {3};
    int32_t input3[1] = {2};
    int16_t input4[2] = {8,9};
    
    int16_t output[6] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Value_Num_Rows) {
    vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT32, DT_INT32, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{2,3}, {1}, {1},{1},{1},{2,2,4}};
    int8_t input0[6] = {1,2,3,4,5,6};
    int32_t input1[1] = {-1};
    int32_t input2[1] = {2};
    int32_t input3[1] = {4};
    int8_t input4[1] = {0};
    
    int8_t output[8] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Value_Num_Cols) {
    vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT32, DT_INT32, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{2,3}, {1}, {1},{1},{1},{2,4,2}};
    int8_t input0[6] = {1,2,3,4,5,6};
    int32_t input1[1] = {-1};
    int32_t input2[1] = {4};
    int32_t input3[1] = {2};
    int8_t input4[1] = {0};
    
    int8_t output[8] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}


TEST_F(TEST_MATRIX_DIAG_V3_UT, Exception_Value_Rows_and_Cols) {
    vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT32, DT_INT32, DT_INT8, DT_INT8};
    vector<vector<int64_t>> shapes = {{2,3}, {1}, {1},{1},{1},{2,3,3}};
    int8_t input0[6] = {1,2,3,4,5,6};
    int32_t input1[1] = {-1};
    int32_t input2[1] = {3};
    int32_t input3[1] = {3};
    int8_t input4[1] = {0};
    
    int8_t output[8] = {0};
    vector<void *> datas = {(void *)input0, (void *)input1,(void *)input2, (void *)input3, (void *)input4,(void *)output};
    CREATE_NODEDEF(shapes, data_types, datas, "RIGHT_LEFT");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}