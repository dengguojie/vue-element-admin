#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "host_sharder.h"
#include "device_sharder.h"
#include "eigen_threadpool.h"

using namespace std;
using namespace aicpu;


namespace {
void EigenParallelForFake(EigenThreadPool *pool, int64_t total, int64_t perUnitSize, const SharderWork &work) {}
}

class SHARDER_UTest : public testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
private:
};

TEST_F(SHARDER_UTest, ParallelFor)
{
    DeviceSharder deviceSharder(DEVICE);
    deviceSharder.ParallelFor(10, 1, [&](int64_t start, int64_t end){});
}
