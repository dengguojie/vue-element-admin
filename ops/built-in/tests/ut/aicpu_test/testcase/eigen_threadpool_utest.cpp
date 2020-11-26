#ifndef private
#define private public
#define protected public
#endif
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include <mockcpp/ChainingMockHelper.h>

#include <unistd.h>
#include <sys/sysinfo.h>

#include "eigen_threadpool.h"

using namespace std;
using namespace aicpu;

class EIGEN_THREAD_POOL_UTest : public testing::Test {
protected:
    virtual void SetUp()
    {
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
    EigenThreadPool instance;
};

TEST_F(EIGEN_THREAD_POOL_UTest, ParallelFor)
{
    uint32_t input[10] = { 0 };
    uint32_t output[10] = { 0 };
    for (size_t i = 0; i < 10; i++) {
        input[i] = 2;
    }

    auto shardCopy = [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            output[i] = input[i];
        }
    };
    instance.ParallelFor(10, 5, shardCopy);
    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(output[i], 2);
    }

    for (size_t i = 0; i < 10; i++) {
        output[i] = 0;
    }
    instance.ParallelFor(10, 1, shardCopy);
    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(output[i], 2);
    }

    for (size_t i = 0; i < 10; i++) {
        output[i] = 0;
    }
    instance.ParallelFor(10, 10, shardCopy);
    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(output[i], 2);
    }
}

TEST_F(EIGEN_THREAD_POOL_UTest, ParallelForFailed)
{
    instance.ParallelFor(10, 5, nullptr);
}

TEST_F(EIGEN_THREAD_POOL_UTest, GetInstanceFailed)
{
    MOCKER(get_nprocs).stubs().will(returnValue(-1));
    EigenThreadPool poolInstance;
    poolInstance.initFlag_ = false;
    EigenThreadPool *pool = poolInstance.GetInstance();
    EXPECT_EQ(pool, nullptr);
}
