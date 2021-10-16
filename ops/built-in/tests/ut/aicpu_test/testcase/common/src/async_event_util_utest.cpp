#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "async_event_util.h"

using namespace std;
using namespace aicpu;

class ASYNC_EVENT_UTIL_UTest : public testing::Test {};

TEST_F(ASYNC_EVENT_UTIL_UTest, NotifyWait)
{
    int param_value = 1;
    void *param = reinterpret_cast<void *>(&param_value);
    AsyncEventUtil::GetInstance().NotifyWait(param, sizeof(int));
}

TEST_F(ASYNC_EVENT_UTIL_UTest, RegEventCb)
{
    auto cb = [](void *param){
        return;
    };
    auto ret = AsyncEventUtil::GetInstance().RegEventCb(1, 1, cb);
    EXPECT_FALSE(ret);
}

TEST_F(ASYNC_EVENT_UTIL_UTest, RegEventCbWithTimes)
{
    auto cb = [](void *param){
        return;
    };
    auto ret = AsyncEventUtil::GetInstance().RegEventCb(1, 1, cb, 0);
    EXPECT_FALSE(ret);
}

TEST_F(ASYNC_EVENT_UTIL_UTest, UnregEventCb)
{
    auto cb = [](void *param){
        return;
    };
    AsyncEventUtil::GetInstance().UnregEventCb(1, 1);
}

TEST_F(ASYNC_EVENT_UTIL_UTest, InitEventUtil)
{
    AsyncEventUtil::GetInstance().InitEventUtil();
}