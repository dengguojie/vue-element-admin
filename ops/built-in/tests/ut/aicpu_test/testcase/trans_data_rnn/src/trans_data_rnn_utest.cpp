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

class TRANSDATARNN_UT : public testing::Test {};

TEST_F(TRANSDATARNN_UT, TransDataRNN_TEST_1) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[16] = {59.2, 62.1, 22.5,  40.5, -97.8, 68.1, 52.3, -50.9,
                          22.4, 7.2,  -51.3, 63.5, 75.2,  82.3, 37.1, 10.4};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  float output_data[64];
  string format_src("ND");
  string format_dst("ND_RNN_BIAS");
  NodeDefBuilder(node_def.get(), "TransDataRNN", "TransDataRNN")
      .Input(
          {"src", DT_FLOAT16, {16}, (void*)input_data, FORMAT_ND})
      .Attr("src_format", format_src)
      .Attr("dst_format", format_dst)
      .Attr("input_size", 16)
      .Attr("hidden_size", 4)
      .Output({"src",
               DT_FLOAT16,
               {64},
               (void*)output_data,
               FORMAT_ND});
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATARNN_UT, TransDataRNN_TEST_2) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[640] = {59.2, 62.1, 22.5,  40.5, -97.8, 68.1, 52.3, -50.9,
                           22.4, 7.2,  -51.3, 63.5, 75.2,  82.3, 37.1, 10.4};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  float output_data[1024];
  string format_src("ND");
  string format_dst("FRACTAL_ZN_RNN");
  NodeDefBuilder(node_def.get(), "TransDataRNN", "TransDataRNN")
      .Input(
          {"src", DT_FLOAT16, {10, 64}, (void*)input_data, FORMAT_ND})
      .Attr("src_format", format_src)
      .Attr("dst_format", format_dst)
      .Attr("input_size", 10)
      .Attr("hidden_size", 16)
      .Output({"src",
               DT_FLOAT16,
               {1,4,16,16},
               (void*)output_data,
               FORMAT_ND});
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

TEST_F(TRANSDATARNN_UT, TransDataRNN_TEST_3) {
  std::cout << "Test TransData begin" << std::endl;
  float input_data[2048] = {59.2, 62.1, 22.5,  40.5, -97.8, 68.1, 52.3, -50.9,
                          22.4, 7.2,  -51.3, 63.5, 75.2,  82.3, 37.1, 10.4};
  auto node_def = CpuKernelUtils::CreateNodeDef();
  float output_data[2048];
  string format_src("ND");
  string format_dst("FRACTAL_ZN_RNN");
  NodeDefBuilder(node_def.get(), "TransDataRNN", "TransDataRNN")
      .Input(
          {"src", DT_FLOAT16, {32, 64}, (void*)input_data, FORMAT_ND})
      .Attr("src_format", format_src)
      .Attr("dst_format", format_dst)
      .Attr("input_size", 16)
      .Attr("hidden_size", 16)
      .Output({"src",
               DT_FLOAT16,
               {2,4,16,16},
               (void*)output_data,
               FORMAT_ND});
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  std::cout << "Test TransData end" << std::endl;
}

