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

class TEST_CandidateSampler_UTest : public testing::Test {};

TEST_F(TEST_CandidateSampler_UTest, LogUniformCandidateSamplerKernels_Unique_Success) {
  // raw data
  int64_t x[3][2] = {{1, 7}, {0, 4}, {3, 3}};
  int64_t sampled[5] = {0};
  float true_count[3 * 2] = {0};
  float sampled_count[5] = {0};

  int sampled_expect[5] = {3, 1, 0, 2, 4};
  float true_count_expect[3 * 2] = {0.994091, 0.9698137, 0.9999436, 0.8830796, 0.930058, 0.930058};
  float sampled_count_expect[5] = {0.9999654, 0.9746604, 0.99542814, 0.89497685, 0.93876845};

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("LogUniformCandidateSampler");

  // set attr
  auto num_true = CpuKernelUtils::CreateAttrValue();
  num_true->SetInt(2);
  nodeDef->AddAttrs("num_true", num_true.get());

  auto num_sampled = CpuKernelUtils::CreateAttrValue();
  num_sampled->SetInt(5);
  nodeDef->AddAttrs("num_sampled", num_sampled.get());

  auto unique = CpuKernelUtils::CreateAttrValue();
  unique->SetBool(true);
  nodeDef->AddAttrs("unique", unique.get());

  auto range_max = CpuKernelUtils::CreateAttrValue();
  range_max->SetInt(10);
  nodeDef->AddAttrs("range_max", range_max.get());

  auto seed = CpuKernelUtils::CreateAttrValue();
  seed->SetInt(1200);
  nodeDef->AddAttrs("seed", seed.get());

  // set input
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {3, 2};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_INT64);
  inputTensor0->SetData(x);
  inputTensor0->SetDataSize(3 * 2 * sizeof(int64_t));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  outputTensor1->SetDataType(DT_INT64);
  outputTensor1->SetData(sampled);
  outputTensor1->SetDataSize(5 * sizeof(int64_t));

  auto outputTensor2 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor2, nullptr);
  outputTensor2->SetDataType(DT_FLOAT);
  outputTensor2->SetData(true_count);
  outputTensor2->SetDataSize(3 * 2 * sizeof(float));

  auto outputTensor3 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor3, nullptr);
  outputTensor3->SetDataType(DT_FLOAT);
  outputTensor3->SetData(sampled_count);
  outputTensor3->SetDataSize(5 * sizeof(float));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);

  std::cout << "************output*************" << std::endl;
  std::set<int64_t> sample_set;
  for (int i = 0; i < 5; ++i) {
    std::cout << sampled[i] << ", " << sampled_count[i] << std::endl;
    bool ret1 = sample_set.find(sampled[i]) == sample_set.end();
    bool ret2 = sampled[i] >= 0 && sampled[i] < 10;
    // bool ret3 = sampled_count[i]  > 0 && sampled_count[i] < 1;
    EXPECT_EQ(ret1, true);
    EXPECT_EQ(ret2, true);
    // EXPECT_EQ(ret3, true);
    sample_set.insert(sampled[i]);
  }
  for (int i = 0; i < 6; ++i) {
    std::cout << true_count[i] << ", ";
    // bool ret = true_count[i]  > 0 && true_count[i] < 1;
    // EXPECT_EQ(ret, true);
  }
  std::cout << std::endl;
}

TEST_F(TEST_CandidateSampler_UTest, LogUniformCandidateSamplerKernels_Success) {
  // raw data
  int64_t x[3][2] = {{1, 7}, {0, 4}, {3, 3}};
  int64_t sampled[5] = {0};
  float true_count[3 * 2] = {0};
  float sampled_count[5] = {0};

  int sampled_expect[5] = {3, 1, 0, 2, 4};
  float true_count_expect[3 * 2] = {0.994091, 0.9698137, 0.9999436, 0.8830796, 0.930058, 0.930058};
  float sampled_count_expect[5] = {0.9999654, 0.9746604, 0.99542814, 0.89497685, 0.93876845};

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("LogUniformCandidateSampler");

  // set attr
  auto num_true = CpuKernelUtils::CreateAttrValue();
  num_true->SetInt(2);
  nodeDef->AddAttrs("num_true", num_true.get());

  auto num_sampled = CpuKernelUtils::CreateAttrValue();
  num_sampled->SetInt(5);
  nodeDef->AddAttrs("num_sampled", num_sampled.get());

  auto unique = CpuKernelUtils::CreateAttrValue();
  unique->SetBool(false);
  nodeDef->AddAttrs("unique", unique.get());

  auto range_max = CpuKernelUtils::CreateAttrValue();
  range_max->SetInt(10);
  nodeDef->AddAttrs("range_max", range_max.get());

  auto seed = CpuKernelUtils::CreateAttrValue();
  seed->SetInt(0);
  nodeDef->AddAttrs("seed", seed.get());

  // set input
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {3, 2};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_INT64);
  inputTensor0->SetData(x);
  inputTensor0->SetDataSize(3 * 2 * sizeof(int64_t));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  outputTensor1->SetDataType(DT_INT64);
  outputTensor1->SetData(sampled);
  outputTensor1->SetDataSize(5 * sizeof(int64_t));

  auto outputTensor2 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor2, nullptr);
  outputTensor2->SetDataType(DT_FLOAT);
  outputTensor2->SetData(true_count);
  outputTensor2->SetDataSize(3 * 2 * sizeof(float));

  auto outputTensor3 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor3, nullptr);
  outputTensor3->SetDataType(DT_FLOAT);
  outputTensor3->SetData(sampled_count);
  outputTensor3->SetDataSize(5 * sizeof(float));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);

  std::cout << "************output*************" << std::endl;
  for (int i = 0; i < 5; ++i) {
    std::cout << sampled[i] << ", " << sampled_count[i] << std::endl;
    bool ret2 = sampled[i] >= 0 && sampled[i] < 10;
    EXPECT_EQ(ret2, true);
  }
  for (int i = 0; i < 6; ++i) {
    std::cout << true_count[i] << ", ";
  }
  std::cout << std::endl;
}

TEST_F(TEST_CandidateSampler_UTest, UniformCandidateSamplerKernels_Unique_Success) {
  // raw data
  int64_t x[3][2] = {{1, 7}, {0, 4}, {3, 3}};
  int64_t sampled[5] = {0};
  float true_count[3 * 2] = {0};
  float sampled_count[5] = {0};

  int sampled_expect[5] = {1, 5, 7, 4, 6};
  float true_count_expect[3 * 2] = {0.468559, 0.468559, 0.468559, 0.468559, 0.468559, 0.468559};
  float sampled_count_expect[5] = {0.5, 0.5, 0.5, 0.5, 0.5};

  auto nodeDef = CpuKernelUtils::CreateNodeDef();
  nodeDef->SetOpType("UniformCandidateSampler");

  // set attr
  auto num_true = CpuKernelUtils::CreateAttrValue();
  num_true->SetInt(2);
  nodeDef->AddAttrs("num_true", num_true.get());

  auto num_sampled = CpuKernelUtils::CreateAttrValue();
  num_sampled->SetInt(5);
  nodeDef->AddAttrs("num_sampled", num_sampled.get());

  auto unique = CpuKernelUtils::CreateAttrValue();
  unique->SetBool(true);
  nodeDef->AddAttrs("unique", unique.get());

  auto range_max = CpuKernelUtils::CreateAttrValue();
  range_max->SetInt(10);
  nodeDef->AddAttrs("range_max", range_max.get());

  auto seed = CpuKernelUtils::CreateAttrValue();
  seed->SetInt(120);
  nodeDef->AddAttrs("seed", seed.get());

  // set input
  auto inputTensor0 = nodeDef->AddInputs();
  EXPECT_NE(inputTensor0, nullptr);
  auto aicpuShape0 = inputTensor0->GetTensorShape();
  std::vector<int64_t> shapes0 = {3, 2};
  aicpuShape0->SetDimSizes(shapes0);
  inputTensor0->SetDataType(DT_INT64);
  inputTensor0->SetData(x);
  inputTensor0->SetDataSize(3 * 2 * sizeof(int64_t));

  // set output
  auto outputTensor1 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor1, nullptr);
  outputTensor1->SetDataType(DT_INT64);
  outputTensor1->SetData(sampled);
  outputTensor1->SetDataSize(5 * sizeof(int64_t));

  auto outputTensor2 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor2, nullptr);
  outputTensor2->SetDataType(DT_FLOAT);
  outputTensor2->SetData(true_count);
  outputTensor2->SetDataSize(3 * 2 * sizeof(float));

  auto outputTensor3 = nodeDef->AddOutputs();
  EXPECT_NE(outputTensor3, nullptr);
  outputTensor3->SetDataType(DT_FLOAT);
  outputTensor3->SetData(sampled_count);
  outputTensor3->SetDataSize(5 * sizeof(float));

  CpuKernelContext ctx(DEVICE);
  EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
  uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
  EXPECT_EQ(ret, KERNEL_STATUS_OK);

  std::cout << "************output*************" << std::endl;
  std::set<int64_t> sample_set;
  for (int i = 0; i < 5; ++i) {
    std::cout << sampled[i] << ", " << sampled_count[i] << std::endl;
    bool ret1 = sample_set.find(sampled[i]) == sample_set.end();
    bool ret2 = sampled[i] >= 0 && sampled[i] < 10;
    // bool ret3 = sampled_count[i]  > 0 && sampled_count[i] < 1;
    EXPECT_EQ(ret1, true);
    EXPECT_EQ(ret2, true);
    // EXPECT_EQ(ret3, true);
    sample_set.insert(sampled[i]);
  }
  for (int i = 0; i < 6; ++i) {
    std::cout << true_count[i] << ", ";
    // bool ret = true_count[i]  > 0 && true_count[i] < 1;
    // EXPECT_EQ(ret, true);
  }
  std::cout << std::endl;
}