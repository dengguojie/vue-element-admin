#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>
#include "cpu_kernel.h"
#include "status.h"
#include "cpu_kernel_register.h"
#include "aicpu_task_struct.h"
#include "device_cpu_kernel.h"
#include "cpu_types.h"
#include "cpu_kernel_utils.h"
#include "Eigen/Core"


using namespace std;
using namespace aicpu;

class TEST_DEFORMABLEOFFSETS_STest : public testing::Test {
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

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host1)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host1..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,1,3,3};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_FLOAT16);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(9 * sizeof(Eigen::half));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<Eigen::half> inputOffset;
    for (int i = 0;i < 9;i++) {
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
    }
    for (int i = 0;i < 9;i++) {
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
    }
    for(int j = 162;j < 243;j++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,27,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT16);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(Eigen::half));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[81];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,1,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(81 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{1, 1, 1, 1};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NCHW");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(1);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 81;s++) {
        cout << outputY[s];
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host2)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host2..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,1,3,3};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_FLOAT16);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(9 * sizeof(Eigen::half));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<float> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<float>(0.0));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<float>(1.0));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,27,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(float));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[81];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,1,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(81 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{1, 1, 1, 1};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NCHW");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(1);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 81;s++) {
        cout << outputY[s];
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host3)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host3..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<int8_t> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<int8_t>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,3,3,1};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_INT8);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(9 * sizeof(int8_t));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<Eigen::half> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<Eigen::half>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,27,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT16);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(Eigen::half));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[81];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,1,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(81 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{0, 0, 0, 0};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NHWC");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(1);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 81;s++) {
        cout << outputY[s] << ' ';
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host4)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host4..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<int8_t> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<int8_t>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,3,3,1};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_INT8);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(9 * sizeof(int8_t));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<float> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<float>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<float>(1.0));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,27,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(float));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[81];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,1,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(81 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{1, 1, 1, 1};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NHWC");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(1);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 81;s++) {
        cout << outputY[s];
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host5)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host5..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,1,3,3};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_FLOAT16);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(9 * sizeof(Eigen::half));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<float> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<float>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<float>(1));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,27,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT16);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(float));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[81];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,1,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(81 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{2, 3, 4, 5};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("HWCN");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(1);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 81;s++) {
        cout << outputY[s];
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host6)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host6..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,1,3,3};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_FLOAT16);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(9 * sizeof(Eigen::half));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<Eigen::half> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<Eigen::half>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,27,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT16);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(Eigen::half));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[81];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,1,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(81 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{1, 1, 1, 1};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NCHW");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(2);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 81;s++) {
        cout << outputY[s];
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host7)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host7..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,1,3,3};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_FLOAT16);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(9 * sizeof(Eigen::half));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<Eigen::half> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<Eigen::half>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,18,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT16);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(Eigen::half));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[81];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,1,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(81 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{1, 1, 1, 1};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NCHW");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(1);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 81;s++) {
        cout << outputY[s];
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host8)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host8..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 27;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,3,3,3};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_FLOAT16);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(27 * sizeof(Eigen::half));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<Eigen::half> inputOffset;

    for (int i = 0;i < 9;i++) {
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
    }
    for (int i = 0;i < 9;i++) {
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0.5));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(0));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        inputOffset.push_back(static_cast<Eigen::half>(-0.5));
    }
    for(int j = 162;j < 243;j++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,27,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT16);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(Eigen::half));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[243];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,3,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(243 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{1, 1, 1, 1};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NCHW");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(1);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 243;s++) {
        cout << outputY[s];
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host9)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host9..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 243;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,9,3,3};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_FLOAT16);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(243 * sizeof(Eigen::half));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<Eigen::half> inputOffset;
    for (int l = 0;l < 3;l++) {
        for (int i = 0;i < 9;i++) {
            inputOffset.push_back(static_cast<Eigen::half>(-0.5));
            inputOffset.push_back(static_cast<Eigen::half>(0));
            inputOffset.push_back(static_cast<Eigen::half>(0.5));
            inputOffset.push_back(static_cast<Eigen::half>(-0.5));
            inputOffset.push_back(static_cast<Eigen::half>(0));
            inputOffset.push_back(static_cast<Eigen::half>(0.5));
            inputOffset.push_back(static_cast<Eigen::half>(-0.5));
            inputOffset.push_back(static_cast<Eigen::half>(0));
            inputOffset.push_back(static_cast<Eigen::half>(0.5));
        }
        for (int i = 0;i < 9;i++) {
            inputOffset.push_back(static_cast<Eigen::half>(0.5));
            inputOffset.push_back(static_cast<Eigen::half>(0.5));
            inputOffset.push_back(static_cast<Eigen::half>(0.5));
            inputOffset.push_back(static_cast<Eigen::half>(0));
            inputOffset.push_back(static_cast<Eigen::half>(0));
            inputOffset.push_back(static_cast<Eigen::half>(0));
            inputOffset.push_back(static_cast<Eigen::half>(-0.5));
            inputOffset.push_back(static_cast<Eigen::half>(-0.5));
            inputOffset.push_back(static_cast<Eigen::half>(-0.5));
        }
        for(int j = 162;j < 243;j++){
            inputOffset.push_back(static_cast<Eigen::half>(1));
        }
    }

    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,81,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT16);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(729 * sizeof(Eigen::half));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[729];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,9,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(729 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{1, 1, 1, 1};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NCHW");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(3);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 729;s++) {
        cout << outputY[s];
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_DEFORMABLEOFFSETS_STest, Host10)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_STest Host10..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("DeformableOffsets");

    //set x
    auto xTensor = nodeDef->AddInputs();
    EXPECT_NE(xTensor, nullptr);
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++) {
         inputX.push_back(static_cast<Eigen::half>(i));
    }
    auto xShape = xTensor->GetTensorShape();
    std::vector<int64_t> xShapes = {1,3,3,1};
    xShape->SetDimSizes(xShapes);
    xTensor->SetDataType(DT_FLOAT16);
    xTensor->SetData(inputX.data());
    xTensor->SetDataSize(9 * sizeof(Eigen::half));

    //set offset
    auto offsetTensor = nodeDef->AddInputs();
    EXPECT_NE(offsetTensor, nullptr);
    std::vector<Eigen::half> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<Eigen::half>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }
    auto offsetShape = offsetTensor->GetTensorShape();
    std::vector<int64_t> offsetShapes = {1,27,3,3};
    offsetShape->SetDimSizes(offsetShapes);
    offsetTensor->SetDataType(DT_FLOAT16);
    offsetTensor->SetData(inputOffset.data());
    offsetTensor->SetDataSize(243 * sizeof(Eigen::half));

    //set output
    auto yTensor = nodeDef->AddOutputs();
    EXPECT_NE(yTensor, nullptr);
    Eigen::half outputY[81];
    auto yShape = yTensor->GetTensorShape();
    std::vector<int64_t> yShapes = {1,1,9,9};
    yShape->SetDimSizes(yShapes);
    yTensor->SetDataType(DT_FLOAT16);
    yTensor->SetData(outputY);
    yTensor->SetDataSize(81 * sizeof(Eigen::half));

    vector<int64_t> stridesListInt{1, 1};
    auto strides = CpuKernelUtils::CreateAttrValue();
    strides->SetListInt(stridesListInt);
    nodeDef->AddAttrs("strides", strides.get());

    vector<int64_t> padsListInt{0, 0, 0, 0};
    auto pads = CpuKernelUtils::CreateAttrValue();
    pads->SetListInt(padsListInt);
    nodeDef->AddAttrs("pads", pads.get());

    vector<int64_t> ksizeListInt{3, 3};
    auto ksize = CpuKernelUtils::CreateAttrValue();
    ksize->SetListInt(ksizeListInt);
    nodeDef->AddAttrs("ksize", ksize.get());

    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    auto dilations = CpuKernelUtils::CreateAttrValue();
    dilations->SetListInt(dilationsListInt);
    nodeDef->AddAttrs("dilations", dilations.get());

    auto dataFormat = CpuKernelUtils::CreateAttrValue();
    dataFormat->SetString("NHWC");
    nodeDef->AddAttrs("data_format", dataFormat.get());

    auto deformableGroups = CpuKernelUtils::CreateAttrValue();
    deformableGroups->SetInt(1);
    nodeDef->AddAttrs("deformable_groups", deformableGroups.get());

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    for(int s = 0;s < 81;s++) {
        cout << outputY[s] << ' ';
        if(0 == (s + 1) % 9) {
            cout << " " << endl;
        }
    }
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}