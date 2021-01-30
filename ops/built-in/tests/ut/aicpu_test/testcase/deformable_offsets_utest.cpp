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
#include "node_def_builder.h"


using namespace std;
using namespace aicpu;

class TEST_DEFORMABLEOFFSETS_UTest : public testing::Test {
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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host1)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host1..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }

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

    Eigen::half outputY[81];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{1, 1, 1, 1};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("NCHW");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_FLOAT16, {1,1,3,3}, inputX.data()})
        .Input({"offset", DT_FLOAT16, {1,27,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,1,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 1);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host2)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host2..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }

    std::vector<float> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<float>(0.0));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<float>(1.0));
    }

    Eigen::half outputY[81];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> padsListInt{1, 1, 1, 1};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("NCHW");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_FLOAT16, {1,1,3,3}, inputX.data()})
        .Input({"offset", DT_FLOAT, {1,27,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,1,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 1);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host3)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host3..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();

    std::vector<int8_t> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<int8_t>(i));
    }

    std::vector<Eigen::half> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<Eigen::half>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }

    Eigen::half outputY[81];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{0, 0, 0, 0};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("NHWC");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_INT8, {1,1,3,3}, inputX.data()})
        .Input({"offset", DT_FLOAT, {1,27,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,1,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 1);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host4)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host4..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<int8_t> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<int8_t>(i));
    }

    std::vector<float> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<float>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<float>(1.0));
    }

    Eigen::half outputY[81];

    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{1, 1, 1, 1};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("NHWC");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_INT8, {1,3,3,1}, inputX.data()})
        .Input({"offset", DT_FLOAT, {1,27,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,1,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 1);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host5)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host5..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }

    std::vector<float> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<float>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<float>(1));
    }

    Eigen::half outputY[81];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{2, 3, 4, 5};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("HWCN");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_FLOAT16, {1,1,3,3}, inputX.data()})
        .Input({"offset", DT_FLOAT16, {1,27,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,1,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 1);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host6)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host6..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }

    std::vector<Eigen::half> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<Eigen::half>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }

    Eigen::half outputY[81];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{1, 1, 1, 1};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    vector<int64_t> ksizeListInt{3, 3};
    string format("NCHW");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_FLOAT16, {1,1,3,3}, inputX.data()})
        .Input({"offset", DT_FLOAT16, {1,27,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,1,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 2);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host7)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host7..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }

    std::vector<Eigen::half> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<Eigen::half>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }

    Eigen::half outputY[81];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{1, 1, 1, 1};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("NCHW");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_FLOAT16, {1,1,3,3}, inputX.data()})
        .Input({"offset", DT_FLOAT16, {1,18,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,1,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 1);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host8)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host8..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 27;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }

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

    Eigen::half outputY[243];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{1, 1, 1, 1};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("NCHW");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_FLOAT16, {1,3,3,3}, inputX.data()})
        .Input({"offset", DT_FLOAT16, {1,27,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,3,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 1);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host9)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host9..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 243;i++){
         inputX.push_back(static_cast<Eigen::half>(i));
    }

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

    Eigen::half outputY[729];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{1, 1, 1, 1};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("NCHW");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_FLOAT16, {1,9,3,3}, inputX.data()})
        .Input({"offset", DT_FLOAT16, {1,81,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,9,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 3);

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

TEST_F(TEST_DEFORMABLEOFFSETS_UTest, Host10)
{
    cout << "stsart run TEST_DEFORMABLEOFFSETS_UTest Host10..." << endl;
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    std::vector<Eigen::half> inputX;
    for(int i = 0;i < 9;i++) {
         inputX.push_back(static_cast<Eigen::half>(i));
    }

    std::vector<Eigen::half> inputOffset;
    for(int i = 0;i < 162;i++){
         inputOffset.push_back(static_cast<Eigen::half>(0.7));
    }
    for(int i = 162;i < 243;i++){
         inputOffset.push_back(static_cast<Eigen::half>(1));
    }

    Eigen::half outputY[81];
    vector<int64_t> stridesListInt{1, 1};
    vector<int64_t> padsListInt{0, 0, 0, 0};
    vector<int64_t> ksizeListInt{3, 3};
    vector<int64_t> dilationsListInt{0, 0, 0, 0};
    string format("NHWC");

    NodeDefBuilder(nodeDef.get(), "DeformableOffsets", "DeformableOffsets")
        .Input({"x", DT_FLOAT16, {1,3,3,1}, inputX.data()})
        .Input({"offset", DT_FLOAT16, {1,27,3,3}, inputOffset.data()})
        .Output({"y", DT_FLOAT16, {1,1,9,9}, outputY})
        .Attr("strides", stridesListInt)
        .Attr("pads", padsListInt)
        .Attr("ksize", ksizeListInt)
        .Attr("dilations", dilationsListInt)
        .Attr("data_format", format)
        .Attr("deformable_groups", 1);

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