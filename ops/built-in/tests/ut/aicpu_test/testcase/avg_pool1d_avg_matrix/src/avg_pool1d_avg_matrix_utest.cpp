#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"
#include <string>
using namespace std;
using namespace aicpu;

class TEST_AVGPOOLl1DAVGMATRIX_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas,format,ksize, strides,pads,   \
   ceil_mode,count_include_pad)                                                \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();             \
  NodeDefBuilder(node_def.get(), "AvgPool1DAvgMatrix", "AvgPool1DAvgMatrix")   \
      .Input({"x", data_types[0], shapes[0], datas[0], FORMAT_NC1HWC0})        \
      .Output({"y", data_types[1], shapes[1], datas[1], FORMAT_NC1HWC0})       \
      .Attr("ksize", ksize)                                                    \
      .Attr("strides", strides)                                                \
      .Attr("pads", pads)                                                      \
      .Attr("ceil_mode", ceil_mode)                                            \
      .Attr("count_include_pad", count_include_pad);


#define ADD_CASE(base_type, aicpu_type)                                        \
  TEST_F(TEST_AVGPOOLl1DAVGMATRIX_UT, AVGPOOLl1DAVGMATRIX##aicpu_type) {       \
    vector<DataType> data_types = {aicpu_type, aicpu_type};                    \
    vector<vector<int64_t>> shapes = {{1,1,1,1,16}, {1,1,1,1,16}};             \
    base_type input[16];                                                       \
    int64_t ksize = 1;                                                         \
    int64_t strides = 2;                                                       \
    bool ceil_mode = false;                                                    \
    bool count_include_pad = true;                                             \
    vector<int64_t>pads ={1,2};                                                \
    SetRandomValue<base_type>(input, 16);                                      \
    base_type output[16] = {(base_type)0};                                     \
    vector<void *> datas = {(void *)input, (void *)output};                    \
    CREATE_NODEDEF(shapes, data_types, datas,format,ksize,strides,pads,        \
    ceil_mode,count_include_pad);                                              \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                              \
    base_type expect_out[16] = {(base_type)1};                                  \
    for(int i = 0 ; i < 16 ; i++){                                             \
        expect_out[i] = (base_type)1;                                          \
    }                                                                          \
    bool res = CompareResult<base_type>(output, expect_out, 16);               \
    EXPECT_EQ(res, true);                                                      \
  }

  TEST_F(TEST_AVGPOOLl1DAVGMATRIX_UT,ceil_mode_true ) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1,1,1,1,16}, {1,1,1,1,16}};
    vector<vector<aicpu::Format>> format = {{FORMAT_NCHW}, {FORMAT_NC1HWC0}};
    float input[64];
    int64_t ksize = 2;
    int64_t strides = 2;
    bool ceil_mode = true;
    bool count_include_pad = true;
    vector<int64_t>pads ={1,2};
    SetRandomValue<float>(input, 64);
    float output[16] = {(float)0};
    vector<void *> datas = {(void *)input, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas,format,ksize,strides,pads,ceil_mode,
    count_include_pad);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    float expect_out[16] = {(float)0};
    for(int i = 0 ; i < 16 ; i++){
        expect_out[i] = (float)0.5;
    }
    bool res = CompareResult<float>(output, expect_out, 16);
    EXPECT_EQ(res, true);
  }
  TEST_F(TEST_AVGPOOLl1DAVGMATRIX_UT,ksize_zero) {
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1,1,1,1,16}, {1,1,1,1,16}};
    vector<vector<aicpu::Format>> format = {{FORMAT_NCHW}, {FORMAT_NC1HWC0}};
    float input[16];
    int64_t ksize = 0;
    int64_t strides = 2;
    bool ceil_mode = true;
    bool count_include_pad = false;
    vector<int64_t>pads ={1,2};
    SetRandomValue<float>(input, 16);
    float output[100] = {(float)0};
    vector<void *> datas = {(void *)input, (void *)output};
    CREATE_NODEDEF(shapes, data_types, datas,format,ksize,strides,pads,ceil_mode,
    count_include_pad);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
  }

ADD_CASE(float, DT_FLOAT)
ADD_CASE(int8_t, DT_INT8)
ADD_CASE(int16_t, DT_INT16)
ADD_CASE(int32_t, DT_INT32)
ADD_CASE(int64_t, DT_INT64)
ADD_CASE(uint8_t, DT_UINT8)
ADD_CASE(Eigen::half, DT_FLOAT16)
ADD_CASE(double, DT_DOUBLE)
