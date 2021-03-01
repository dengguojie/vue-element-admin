#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "kernel_util.h"
#undef private
#undef protected

using namespace std;
using namespace aicpu;

class KERNEL_UTIL_UT : public testing::Test {};

TEST_F(KERNEL_UTIL_UT, DType)
{
  DataType dtype = DType("DT_INT32");
  EXPECT_EQ(dtype, DT_INT32);
}

TEST_F(KERNEL_UTIL_UT, DTypeStr)
{
  std::string dtype_str = DTypeStr(DT_INT32);
  EXPECT_EQ(dtype_str, "DT_INT32");
}
