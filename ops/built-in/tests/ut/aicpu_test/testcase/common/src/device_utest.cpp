#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "device.h"

using namespace std;
using namespace aicpu;

class DEVICE_UTest : public testing::Test {};

TEST_F(DEVICE_UTest, InitHost)
{
    Device device(HOST);

    auto type = device.GetDeviceType();
    EXPECT_EQ(type, HOST);

    auto *sharder = device.GetSharder();
    EXPECT_NE(sharder, nullptr);
}

TEST_F(DEVICE_UTest, InitDevice)
{
    Device device(DEVICE);

    auto type = device.GetDeviceType();
    EXPECT_EQ(type, DEVICE);

    auto *sharder = device.GetSharder();
    EXPECT_NE(sharder, nullptr);
}

