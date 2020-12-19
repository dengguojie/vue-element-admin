#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif

#include "cpu_node_def.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;

class NODE_DEF_UTest : public testing::Test {};

TEST_F(NODE_DEF_UTest, GetInputAndOutputTensor)
{
    auto node = CpuKernelUtils::CreateNodeDef();
    auto ret = node->AddInputs();
    EXPECT_NE(ret, nullptr);
    ret = node->AddOutputs();
    EXPECT_NE(ret, nullptr);

    ret = node->MutableInputs(0);
    EXPECT_NE(ret, nullptr);
    int32_t inSize = node->InputsSize();
    EXPECT_EQ(inSize, 1);
    ret = node->MutableInputs(1);
    EXPECT_EQ(ret, nullptr);

    ret = node->MutableOutputs(0);
    EXPECT_NE(ret, nullptr);
    int32_t outSize = node->OutputsSize();
    EXPECT_EQ(outSize, 1);
    EXPECT_EQ(node->MutableOutputs(1), nullptr);
}

