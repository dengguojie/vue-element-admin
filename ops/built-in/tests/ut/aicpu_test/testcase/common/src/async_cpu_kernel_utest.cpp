#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "async_cpu_kernel.h"
#include "cpu_context.h"

using namespace std;
using namespace aicpu;

class ASYNC_CPU_KERNEL_UTest : public testing::Test {};

class TestAsyncCpuKernel : public AsyncCpuKernel {
public:
    ~TestAsyncCpuKernel() = default;
    typedef std::function<void(uint32_t status)> DoneCallback;
    virtual uint32_t ComputeAsync(CpuKernelContext &ctx, DoneCallback done) {
        std::cout << "TestAsyncCpuKernel ComputeAsync." << std::endl;
        done(1);
        return 1;
    }
};

TEST_F(ASYNC_CPU_KERNEL_UTest, AsyncCpuKernelCompute)
{
    TestAsyncCpuKernel *kernel = new TestAsyncCpuKernel();
    CpuKernelContext ctx(HOST);
    auto ret = kernel->Compute(ctx);
    EXPECT_EQ(ret, 1);
}

TEST_F(ASYNC_CPU_KERNEL_UTest, AsyncCpuKernelAsyncCompute)
{
    TestAsyncCpuKernel *kernel = new TestAsyncCpuKernel();
    CpuKernelContext ctx(HOST);
    auto done = [](uint32_t status){
        std::cout << "call done!" << std::endl;
    };
    auto ret = kernel->ComputeAsync(ctx, done);

    EXPECT_EQ(ret, 1);
}