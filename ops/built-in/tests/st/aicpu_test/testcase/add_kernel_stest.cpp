#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

using namespace std;
using namespace aicpu;

class TEST_ADD_KERNEL_STest : public testing::Test {
  protected:
    virtual void SetUp() {}

    virtual void TearDown() {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ADD_KERNEL_STest, scalar_add_scalar) {
    cout<<"Test Kernel Begin."<<endl;
    int8_t input_0[1] = {1};
    int8_t input_1[1] = {1};
    int8_t output[1] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_INT8, {}, (void *)input_0})
        .Input({"x2", DT_INT8, {}, (void *)input_1})
        .Output({"y", DT_INT8, {}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 2);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, scalar_add_vector) {
    cout<<"Test Kernel Begin."<<endl;
    int16_t input_0[1] = {1};
    int16_t input_1[2] = {1, 1};
    int16_t output[2] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_INT16, {}, (void *)input_0})
        .Input({"x2", DT_INT16, {2}, (void *)input_1})
        .Output({"y", DT_INT16, {2}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 2);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, vector_add_scalar) {
    cout<<"Test Kernel Begin."<<endl;
    int32_t input_0[2] = {1, 1};
    int32_t input_1[1] = {1};
    int32_t output[2] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_INT32, {2}, (void *)input_0})
        .Input({"x2", DT_INT32, {}, (void *)input_1})
        .Output({"y", DT_INT32, {2}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 2);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, vector_add_vector_match) {
    cout<<"Test Kernel Begin."<<endl;
    int64_t input_0[2] = {1, 1};
    int64_t input_1[2] = {1, 1};
    int64_t output[2] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_INT64, {2, 1, 1}, (void *)input_0})
        .Input({"x2", DT_INT64, {2, 1, 1}, (void *)input_1})
        .Output({"y", DT_INT64, {2, 1, 1}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 2);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, vector_add_vector_broatcast_0) {
    cout<<"Test Kernel Begin."<<endl;
    uint8_t input_0[1] = {1};
    uint8_t input_1[2] = {1, 1};
    uint8_t output[2] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_UINT8, {1}, (void *)input_0})
        .Input({"x2", DT_UINT8, {2, 1, 1, 1}, (void *)input_1})
        .Output({"y", DT_UINT8, {2, 1, 1, 1}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 2);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, vector_add_vector_broatcast_1) {
    cout<<"Test Kernel Begin."<<endl;
    uint16_t input_0[2] = {1, 1};
    uint16_t input_1[1] = {1};
    uint16_t output[2] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_UINT16, {2, 1, 1, 1, 1}, (void *)input_0})
        .Input({"x2", DT_UINT16, {1}, (void *)input_1})
        .Output({"y", DT_UINT16, {2, 1, 1, 1, 1}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 2);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, vector_add_vector_broatcast_both) {
    cout<<"Test Kernel Begin."<<endl;
    uint32_t input_0[2] = {1, 1};
    uint32_t input_1[2] = {1, 1};
    uint32_t output[4] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_UINT32, {1, 2}, (void *)input_0})
        .Input({"x2", DT_UINT32, {2, 1, 1, 1, 1, 1}, (void *)input_1})
        .Output({"y", DT_UINT32, {2, 1, 1, 1, 1, 2}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 2);
    EXPECT_EQ(output[1], 2);
    EXPECT_EQ(output[2], 2);
    EXPECT_EQ(output[3], 2);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, data_type_not_match) {
    cout<<"Test Kernel Begin."<<endl;
    float input_0[1] = {0};
    double input_1[1] = {0};
    float output[1] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_FLOAT, {}, (void *)input_0})
        .Input({"x2", DT_DOUBLE, {}, (void *)input_1})
        .Output({"y", DT_FLOAT, {}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_NE(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, data_type_not_support) {
    cout<<"Test Kernel Begin."<<endl;
    bool input_0[1] = {true};
    bool input_1[1] = {true};
    bool output[1] = {false};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_BOOL, {}, (void *)input_0})
        .Input({"x2", DT_BOOL, {}, (void *)input_1})
        .Output({"y", DT_BOOL, {}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_NE(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, bcast_max_size_not_match) {
    cout<<"Test Kernel Begin."<<endl;
    float input_0[1] = {1};
    float input_1[1] = {1};
    float output[1] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_BOOL, {1}, (void *)input_0})
        .Input({"x2", DT_BOOL, {1}, (void *)input_1})
        .Output({"y", DT_BOOL, {1, 1}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_NE(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, bcast_shape_not_match) {
    cout<<"Test Kernel Begin."<<endl;
    float input_0[1] = {1};
    float input_1[1] = {1};
    float output[1] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_BOOL, {1, 2}, (void *)input_0})
        .Input({"x2", DT_BOOL, {1, 3}, (void *)input_1})
        .Output({"y", DT_BOOL, {1, 1}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_NE(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}

TEST_F(TEST_ADD_KERNEL_STest, bcast_not_support) {
    cout<<"Test Kernel Begin."<<endl;
    float input_0[1] = {1};
    float input_1[1] = {1};
    float output[1] = {0};

    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "Add", "Add")
        .Input({"x1", DT_BOOL, {1, 2}, (void *)input_0})
        .Input({"x2", DT_BOOL, {1, 3}, (void *)input_1})
        .Output({"y", DT_BOOL, {1, 3}, (void *)output});

    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_NE(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    cout<<"Test Kernel "<<nodeDef->GetOpType()<<" Finish. "<<endl;
}
