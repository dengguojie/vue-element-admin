#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

// ----------------Muls-------------------
class dynamic_muls : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_muls SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_muls TearDown" << std::endl;
    }
};