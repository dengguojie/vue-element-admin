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

using namespace std;
using namespace aicpu;

class TEST_CacheSwapTable_UTest : public testing::Test {};

TEST_F(TEST_CacheSwapTable_UTest, CacheSwapTable_Success) {
  // raw data
  float cache_table[40] = {0, 0, 0, 0, 10, 5, 0, 1, 2, 1, 0, 1, 15, 7, -5, 2, 0,  0, 0,  0,
                           0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 3,  3, 0,  1, 21, 9, -5, 1};
  int64_t swap_cache_idx[5] = {-1, -1, 2, 3, -1};
  float miss_value[20] = {0, 0, 0, 0, 0, 0, 0, 0, 22, 22, 22, 22, 33, 33, 33, 33, 11, 11, 11, 11};
  float old_value[20] = {0};

  float expect_old_value[20] = {0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 15, 7, -5, 2, 0, 0, 0, 0};
  float expect_cache_table[40] = {0, 0, 0, 0, 10, 5, 0, 1, 22, 22, 22, 22, 33, 33, 33, 33, 0,  0, 0,  0,
                                  0, 0, 0, 0, 0,  0, 0, 0, 0,  0,  0,  0,  3,  3,  0,  1,  21, 9, -5, 1};

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("CacheSwapTable");

  // set input
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {10, 4};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_FLOAT);
  inputTensor0->SetData(cache_table);
  inputTensor0->SetDataSize(10 * 4 * sizeof(float));

  auto inputTensor1 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor1, nullptr);
  auto aicpuShape1 = inputTensor1->GetTensorShape();
  std::vector<int64_t> shapes1 = {5};
  aicpuShape1->SetDimSizes(shapes1);
  inputTensor1->SetDataType(DT_INT64);
  inputTensor1->SetData(swap_cache_idx);
  inputTensor1->SetDataSize(5 * sizeof(uint64_t));

  auto inputTensor2 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor2, nullptr);
  auto aicpuShape2 = inputTensor2->GetTensorShape();
  std::vector<int64_t> shapes2 = {5, 4};
  aicpuShape2->SetDimSizes(shapes2);
  inputTensor2->SetDataType(DT_FLOAT);
  inputTensor2->SetData(miss_value);
  inputTensor2->SetDataSize(5 * 4 * sizeof(float));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  outputTensor1->SetDataType(DT_FLOAT);
  outputTensor1->SetData(old_value);
  outputTensor1->SetDataSize(5 * 4 * sizeof(float));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);
  EXPECT_EQ(0, std::memcmp(old_value, expect_old_value, 20 * sizeof(float)));
  EXPECT_EQ(0, std::memcmp(cache_table, expect_cache_table, 40 * sizeof(float)));
}
