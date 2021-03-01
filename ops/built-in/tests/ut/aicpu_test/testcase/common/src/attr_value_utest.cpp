#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "cpu_attr_value.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;

class ATTR_VALUE_UTest : public testing::Test {};

TEST_F(ATTR_VALUE_UTest, String)
{
    auto attr1 = CpuKernelUtils::CreateAttrValue();
    attr1->SetString("aaa");
    string ret = attr1->GetString();
    EXPECT_EQ(ret, "aaa");

    auto attr2 = CpuKernelUtils::CreateAttrValue();
    vector<string> list = {"aaa", "bbb"};
    attr2->SetListString(list);
    attr2->AddListString("ccc");
    EXPECT_EQ(attr2->ListStringSize(), 3);
    auto listRet = attr2->GetListString();
    EXPECT_EQ(listRet[0], "aaa");
    EXPECT_EQ(listRet[1], "bbb");
    EXPECT_EQ(listRet[2], "ccc");
}

TEST_F(ATTR_VALUE_UTest, Int)
{
    auto attr1 = CpuKernelUtils::CreateAttrValue();
    attr1->SetInt((int64_t)0);
    auto ret = attr1->GetInt();
    EXPECT_EQ(ret, 0);

    auto attr2 = CpuKernelUtils::CreateAttrValue();
    vector<int64_t> list = {0, 1};
    attr2->SetListInt(list);
    attr2->AddListInt(2);
    EXPECT_EQ(attr2->ListIntSize(), 3);
    auto listRet = attr2->GetListInt();
    EXPECT_EQ(listRet[0], 0);
    EXPECT_EQ(listRet[1], 1);
    EXPECT_EQ(listRet[2], 2);
}

TEST_F(ATTR_VALUE_UTest, Float)
{
    auto attr1 = CpuKernelUtils::CreateAttrValue();
    attr1->SetFloat(1);
    auto ret = attr1->GetFloat();
    EXPECT_EQ(ret, 1);

    auto attr2 = CpuKernelUtils::CreateAttrValue();
    vector<float> list = {1, 2};
    attr2->SetListFloat(list);
    attr2->AddListFloat(3);
    EXPECT_EQ(attr2->ListFloatSize(), 3);
    auto listRet = attr2->GetListFloat();
    EXPECT_EQ(listRet[0], 1);
    EXPECT_EQ(listRet[1], 2);
    EXPECT_EQ(listRet[2], 3);
}

TEST_F(ATTR_VALUE_UTest, Bool)
{
    auto attr1 = CpuKernelUtils::CreateAttrValue();
    attr1->SetBool(true);
    auto ret = attr1->GetBool();
    EXPECT_EQ(ret, true);

    auto attr2 = CpuKernelUtils::CreateAttrValue();
    vector<bool> list = {true, false};
    attr2->SetListBool(list);
    attr2->AddListBool(false);
    EXPECT_EQ(attr2->ListBoolSize(), 3);
    auto listRet = attr2->GetListBool();
    EXPECT_EQ(listRet[0], true);
    EXPECT_EQ(listRet[1], false);
    EXPECT_EQ(listRet[2], false);
}

TEST_F(ATTR_VALUE_UTest, DataType)
{
    auto attr1 = CpuKernelUtils::CreateAttrValue();
    attr1->SetDataType(DT_INT8);
    auto ret = attr1->GetDataType();
    EXPECT_EQ(ret, 2);

    auto attr2 = CpuKernelUtils::CreateAttrValue();
    vector<DataType> list = {DT_INT8, DT_INT32};
    attr2->SetListDataType(list);
    attr2->AddListDataType(DT_INT16);
    EXPECT_EQ(attr2->ListDataTypeSize(), DT_INT32);
    auto listRet = attr2->GetListDataType();
    EXPECT_EQ(listRet[0], DT_INT8);
    EXPECT_EQ(listRet[1], DT_INT32);
    EXPECT_EQ(listRet[2], DT_INT16);
}

TEST_F(ATTR_VALUE_UTest, TensorShape)
{
    auto attr1 = CpuKernelUtils::CreateAttrValue();
    auto shape = CpuKernelUtils::CreateTensorShape();
    vector<int64_t> dims = {3,1,3,5};
    shape->SetDimSizes(dims);
    attr1->SetTensorShape(shape.get());
    auto ret = attr1->GetTensorShape();
    EXPECT_EQ(ret->GetDims(), 4);

    auto attr2 = CpuKernelUtils::CreateAttrValue();
    vector<TensorShape *> list;
    auto shape2 = CpuKernelUtils::CreateTensorShape();
    vector<int64_t> dims2 = {3,1,3};
    shape2->SetDimSizes(dims2);
    list.push_back(shape2.get());
    attr2->SetListTensorShape(list);

    auto shape3 = attr2->AddListTensorShape();
    vector<int64_t> dims3 = {3,1};
    shape3->SetDimSizes(dims3);
    EXPECT_EQ(attr2->ListTensorShapeSize(), 2);
    auto listRet = attr2->GetListTensorShape();
    EXPECT_EQ(listRet[0].GetDims(), 3);
    EXPECT_EQ(listRet[1].GetDims(), 2);
}


TEST_F(ATTR_VALUE_UTest, Tensor)
{
    auto tensor = CpuKernelUtils::CreateTensor();
    auto attr1 = CpuKernelUtils::CreateAttrValue();
    tensor->SetDataType(DT_INT8);
    attr1->SetTensor(tensor.get());
    auto ret = attr1->GetTensor();
    EXPECT_EQ(ret->GetDataType(), DT_INT8);

    auto attr2 = CpuKernelUtils::CreateAttrValue();
    vector<Tensor *> list;
    auto tensor2 = CpuKernelUtils::CreateTensor();
    tensor2->SetDataType(DT_INT32);
    list.push_back(tensor2.get());
    attr2->SetListTensor(list);

    auto tensor3 = attr2->AddListTensor();
    tensor3->SetDataType(DT_INT16);
    EXPECT_EQ(attr2->ListTensorSize(), 2);
    auto listRet = attr2->GetListTensor();
    EXPECT_EQ(listRet[0].GetDataType(), DT_INT32);
    EXPECT_EQ(listRet[1].GetDataType(), DT_INT16);
}

