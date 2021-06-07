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
#include "securec.h"

using namespace std;
using namespace aicpu;

class TRANSDATA_UT : public testing::Test {};
TEST_F(TRANSDATA_UT, FLOAT_FORMAT_DHWCN) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16] = {59.2, 62.1, 22.5,  40.5, -97.8, 68.1, 52.3, -50.9,
                          22.4, 7.2,  -51.3, 63.5, 75.2,  82.3, 37.1, 10.4};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];
  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input(
          {"src", DT_FLOAT, {1, 2, 2, 2, 2}, (void*)input_data, FORMAT_DHWCN})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src",
               DT_FLOAT,
               {4, 1, 16, 16},
               (void*)output_data,
               FORMAT_FRACTAL_Z_3D})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, FLOAT_FORMAT_HWCN_TO_FZC04) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16];
  for (int i = 0; i < 16; i++) {
    input_data[i] = i + 1;
  }
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[256];
  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_FLOAT, {2, 2, 2, 2}, (void*)input_data, FORMAT_HWCN})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"dst",
               DT_FLOAT,
               {1, 16, 1, 16},
               (void*)output_data,
               FORMAT_FRACTAL_Z_C04})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, FLOAT_FORMAT_MD_FAILED) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16];
  for (int i = 0; i < 16; i++) {
    input_data[i] = i + 1;
  }
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[256];
  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_FLOAT, {2, 2, 2, 2}, (void*)input_data, FORMAT_MD})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"dst",
               DT_FLOAT,
               {1, 16, 1, 16},
               (void*)output_data,
               FORMAT_FRACTAL_Z_C04})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, FLOAT_FORMAT_HWCN_TO_FZC04_C_FAILED) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16];
  for (int i = 0; i < 16; i++) {
    input_data[i] = i + 1;
  }
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[256];
  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_FLOAT, {2, 2, 5, 2}, (void*)input_data, FORMAT_HWCN})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"dst",
               DT_FLOAT,
               {1, 16, 1, 16},
               (void*)output_data,
               FORMAT_FRACTAL_Z_C04})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, FLOAT_FORMAT_ND) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16];
  for (int i = 0; i < 16; i++) {
    input_data[i] = i + 1;
  }
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];
  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_FLOAT, {2, 1}, (void*)input_data, FORMAT_ND})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"dst",
               DT_FLOAT,
               {1, 1, 1, 2, 2, 16, 16},
               (void*)output_data,
               FORMAT_FRACTAL_Z_3D})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, FLOAT_FORMAT_NDHWC) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16] = {59.2, 62.1, 22.5,  40.5, -97.8, 68.1, 52.3, -50.9,
                          22.4, 7.2,  -51.3, 63.5, 75.2,  82.3, 37.1, 10.4};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];
  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input(
          {"src", DT_FLOAT, {2, 1, 2, 2, 2}, (void*)input_data, FORMAT_NDHWC})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src",
               DT_FLOAT,
               {4, 1 , 16, 16},
               (void*)output_data,
               FORMAT_FRACTAL_Z_3D})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, FLOAT_FORMAT_NCDHW) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16] = {59.2, 62.1, 22.5,  40.5, -97.8, 68.1, 52.3, -50.9,
                          22.4, 7.2,  -51.3, 63.5, 75.2,  82.3, 37.1, 10.4};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];
  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input(
          {"src", DT_FLOAT, {2, 2, 1, 2, 2}, (void*)input_data, FORMAT_NCDHW})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src",
               DT_FLOAT,
               {4, 1, 16, 16},
               (void*)output_data,
               FORMAT_FRACTAL_Z_3D})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}


TEST_F(TRANSDATA_UT, FLOAT_FORMAT_NCHW) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16] = {59.2, 62.1, 22.5,  40.5, -97.8, 68.1, 52.3, -50.9,
                          22.4, 7.2,  -51.3, 63.5, 75.2,  82.3, 37.1, 10.4};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];

  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_FLOAT, {2, 2, 2, 2}, (void*)input_data, FORMAT_NCHW})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src",
               DT_FLOAT,
               {4, 1,16, 16},
               (void*)output_data,
               FORMAT_FRACTAL_Z})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, INT8_FORMAT_NCHW) {
  std::cout << "Test TransData begin" << std::endl;
  int8_t input_data[16] = {59, 62, 22, 40, 97, 68, 52, -50,
                           22, 7,  -5, 63, 75, 82, 37, 10};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];

  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_INT8, {2, 2, 2, 2}, (void*)input_data, FORMAT_NCHW})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src",
               DT_INT8,
               {4,1, 16, 32},
               (void*)output_data,
               FORMAT_FRACTAL_Z})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, INT8_FORMAT_ND_TO_FORMAT_FRACTAL_NZ) {
  std::cout << "Test TransData begin" << std::endl;
  int8_t input_data[16] = {59, 62, 22, 40, 97, 68, 52, -50,
                           22, 7,  -5, 63, 75, 82, 37, 10};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];

  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_INT8, {2, 2, 2, 2}, (void*)input_data, FORMAT_ND})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src",
               DT_INT8,
               {2, 2, 1, 1, 16, 32},
               (void*)output_data,
               FORMAT_FRACTAL_NZ})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, INT8_FORMAT_NHWC_TO_FORMAT_FRACTAL_NZ) {
  std::cout << "Test TransData begin" << std::endl;
  int8_t input_data[16] = {59, 62, 22, 40, 97, 68, 52, -50,
                           22, 7,  -5, 63, 75, 82, 37, 10};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];

  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_INT8, {2, 2, 2, 2}, (void*)input_data, FORMAT_NHWC})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src",
               DT_INT8,
               {2, 2, 1, 1, 16, 32},
               (void*)output_data,
               FORMAT_FRACTAL_NZ})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, INT8_FORMAT_NHWC_TO_FORMAT_FRACTAL_NZ_FAILED) {
  std::cout << "Test TransData begin" << std::endl;
  int8_t input_data[16] = {59, 62, 22, 40, 97, 68, 52, -50,
                           22, 7,  -5, 63, 75, 82, 37, 10};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];

  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src", DT_INT8, {2, 2, 2, 2}, (void*)input_data, FORMAT_NHWC})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src",
               DT_INT8,
               {2, 1, 1, 16, 32},
               (void*)output_data,
               FORMAT_FRACTAL_NZ})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, INT8_FORMAT_FRACTAL_NZ_TO_FORMAT_ND) {
  std::cout << "Test TransData begin" << std::endl;
  int8_t input_data[2048] = {59, 62, 22, 40, 97, 68, 52, -50,
                             22, 7,  -5, 63, 75, 82, 37, 10};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];

  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src",
              DT_INT8,
              {2, 2, 1, 1, 16, 32},
              (void*)input_data,
              FORMAT_FRACTAL_NZ})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src", DT_INT8, {2, 2, 2, 2}, (void*)output_data, FORMAT_NHWC})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, INT8_FORMAT_FRACTAL_NZ_TO_FORMAT_ND_FAILED) {
  std::cout << "Test TransData begin" << std::endl;
  int8_t input_data[2048] = {59, 62, 22, 40, 97, 68, 52, -50,
                             22, 7,  -5, 63, 75, 82, 37, 10};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  float output_data[1024];

  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src",
              DT_INT8,
              {2, 1, 1, 16, 32},
              (void*)input_data,
              FORMAT_FRACTAL_NZ})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src", DT_INT8, {2, 2, 2, 2}, (void*)output_data, FORMAT_NHWC})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATA_UT, FORMAT_NCHWToFORMAT_NHWC) {
  std::cout << "Test TransData begin" << std::endl;
  int8_t input_data[16] = {59, 62, 22, 40, 97, 68, 52, 50,
                             22, 7,  5, 63, 75, 82, 37, 10};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  int64_t groups = 1;
  int8_t output_data[32];

  NodeDefBuilder(node_def.get(), "TransData", "TransData")
      .Input({"src",
              DT_INT8,
              {2, 2, 2, 2},
              (void*)input_data,
              FORMAT_NCHW})
      .Attr("src_format", DT_STRING)
      .Attr("dst_format", DT_STRING)
      .Output({"src", DT_INT8, {2, 2, 2, 2}, (void*)output_data, FORMAT_NHWC})
      .Attr("groups", groups);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}