#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_NO_OP_UT : public testing::Test {};

#define CREATE_NODEDEF()                                           \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef(); \
  NodeDefBuilder(node_def.get(), "NoOp", "NoOp")

#define NO_OP_CASE_WITH_SHAPE(case_name)                            \
  TEST_F(TEST_NO_OP_UT, TestNoOp_##case_name) {                     \
    CREATE_NODEDEF();                                               \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                   \
  }

NO_OP_CASE_WITH_SHAPE(no_op_test)