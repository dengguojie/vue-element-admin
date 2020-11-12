#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include "cpu_node_def.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;

class NODE_DEF_UTest : public testing::Test {
protected:
    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

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

TEST_F(NODE_DEF_UTest, AddInputOutputFailed)
{
    auto node = CpuKernelUtils::CreateNodeDef();
    MOCKER_CPP(CpuKernelUtils::CreateTensor, std::shared_ptr<Tensor>(TensorImpl *))
    .stubs()
    .will(returnValue(std::shared_ptr<Tensor>(nullptr)));
    auto ret = node->AddInputs();
    EXPECT_EQ(ret, nullptr);
    ret = node->AddOutputs();
    EXPECT_EQ(ret, nullptr);
}

TEST_F(NODE_DEF_UTest, GetInputOutputFailed)
{
    auto node = CpuKernelUtils::CreateNodeDef();
    auto ret = node->AddInputs();
    EXPECT_NE(ret, nullptr);
    ret = node->AddOutputs();
    EXPECT_NE(ret, nullptr);

    auto attr1 = CpuKernelUtils::CreateAttrValue();
    node->AddAttrs("aaa", attr1.get());

    MOCKER_CPP(CpuKernelUtils::CreateTensor, std::shared_ptr<Tensor>(TensorImpl *))
    .stubs()
    .will(returnValue(std::shared_ptr<Tensor>(nullptr)));
    MOCKER_CPP(CpuKernelUtils::CreateAttrValue, std::shared_ptr<AttrValue>(AttrValueImpl *))
    .stubs()
    .will(returnValue(std::shared_ptr<AttrValue>(nullptr)));

    ret = node->MutableInputs(0);
    EXPECT_EQ(ret, nullptr);
    ret = node->MutableOutputs(0);
    EXPECT_EQ(ret, nullptr);
    auto attrRet = node->Attrs();
}
