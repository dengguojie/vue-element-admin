#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include <thread>
#include <atomic>
#include <sys/time.h>
#include "node_def_builder.h"
#include "cpu_kernel_utils.h"

using namespace std;
using namespace aicpu;

namespace {
const char *TEST_THREAD_POOL = "TestThreadPool";
std::atomic<uint32_t> g_threadIndex(0);

double GetCurrentCpuTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return double((double)tp.tv_sec * 1.e3 + (double)tp.tv_usec * 1.e-3);
}
}

namespace aicpu {
class TestThreadPoolKernel : public CpuKernel {
public:
    ~TestThreadPoolKernel() = default;

protected:
    uint32_t Compute(CpuKernelContext &ctx) override
    {
        string op = ctx.GetOpType();
        cout << "TestThreadPoolKernel:" << op << " begin." << endl;
        uint32_t totalSize = 128;
        uint32_t perUnitSize = totalSize;

        std::unique_ptr<int32_t[]> input { new int32_t[totalSize] };
        for (uint32_t i = 0; i < totalSize; ++i) {
            input[i] = 2;
        }
        std::unique_ptr<int32_t[]> output { new int32_t[totalSize] };
        auto shardCopy = [&](int64_t start, int64_t end) {
            auto start_ms = GetCurrentCpuTime();
            g_threadIndex++;
            uint32_t index = g_threadIndex;
            for (int64_t i = start; i < end; ++i) {
                output[i] = input[i];
            }

            auto end_ms = GetCurrentCpuTime();
            auto elapsed = end_ms - start_ms;
            cout << "shardCopy begin, thread Index:" << index << ", perunit start:" << start << ", end:" << end <<
                ", size:" << end - start << ", start_ms:" << (uint64_t)start_ms << ", end_ms:" << (uint64_t)end_ms <<
                ", elapsed:" << elapsed << " ms\n";
        };
        auto start = GetCurrentCpuTime();
        CpuKernelUtils::ParallelFor(ctx, totalSize, perUnitSize, shardCopy);
        auto end = GetCurrentCpuTime();
        auto elapsed = end - start;
        std::cout << "thread num:" << g_threadIndex << ", duration:" << elapsed << " ms\n";
        for (uint32_t i = 0; i < totalSize; ++i) {
            if (output[i] != 2) {
                return -1;
            }
        }
        return KERNEL_STATUS_OK;
    }
};
REGISTER_CPU_KERNEL(TEST_THREAD_POOL, TestThreadPoolKernel);
} // namespace aicpu

class TEST_THREAD_POOL_ST : public testing::Test {
protected:
    virtual void SetUp() {}

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
};

TEST_F(TEST_THREAD_POOL_ST, TestThreadPool)
{
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(nodeDef.get(), "TestThreadPool", "TestThreadPool");
    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);

    g_threadIndex.exchange(0);
    ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}
