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

using namespace std;
using namespace aicpu;

class TEST_PRODENVMATA_CALCRIJ_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, sel_a)   \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "ProdEnvMatACalcRij", "ProdEnvMatACalcRij")                       \
      .Input({"coord", data_types[0], shapes[0], datas[0]})            \
      .Input({"type", data_types[1], shapes[1], datas[1]})            \
      .Input({"natoms", data_types[1], shapes[2], datas[2]})            \
      .Input({"box", data_types[0], shapes[3], datas[3]})            \
      .Input({"mesh", data_types[1], shapes[4], datas[4]})            \
      .Output({"rij", data_types[0], shapes[5], datas[5]})           \
      .Output({"nlist", data_types[1], shapes[6], datas[6]})           \
      .Output({"distance", data_types[0], shapes[6], datas[7]})           \
      .Output({"rij_x", data_types[0], shapes[6], datas[7]})           \
      .Output({"rij_y", data_types[0], shapes[6], datas[7]})           \
      .Output({"rij_z", data_types[0], shapes[6], datas[7]})           \
      .Attr("rcut_r", 10000.0)           \
      .Attr("sel_a", sel_a);

TEST_F(TEST_PRODENVMATA_CALCRIJ_UT, TestPRODENVMATA_CALCRIJ_Small) {
  vector<DataType> data_types = {DT_FLOAT, DT_INT32};
  vector<vector<int64_t>> shapes = {{1,9}, {1,3}, {4}, {1}, {1027}, {1,6}, {1,2}};
  float input0[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t input1[3] = {0, 1, 0};
  int32_t input2[4] = {1, 3, 0, 0};
  float input3[1] = {0.0};
  int32_t input4[1027] = {1, 0, 2, 1, 2};
  for (int i = 5; i < 1027; i++) {
    input4[i] = -1;
  }
  float output0[6] = {0};
  int32_t output1[2] = {0};
  float output2[2] = {0};
  vector<void *> datas = {(void *)input0, (void *)input1, (void *)input2, (void *)input3, (void *)input4,
  (void *)output0, (void *)output1, (void *)output2};
  vector<int64_t> sel_a = {1, 1};
  CREATE_NODEDEF(shapes, data_types, datas, sel_a);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
  int32_t expect_out1[2] = {2,1};
  float expect_out0[6] = {0.6, 0.6, 0.6, 0.3, 0.3, 0.3};
  EXPECT_EQ(CompareResult<int32_t>(output1, expect_out1, 2), true);
  EXPECT_EQ(CompareResult<float>(output0, expect_out0, 6), true);
}
