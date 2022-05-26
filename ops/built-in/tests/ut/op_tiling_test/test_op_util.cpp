#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "op_util.h"

using namespace std;

class OpUtilTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "OpUtilTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "OpUtilTest TearDown" << std::endl;
  }
};

TEST_F(OpUtilTest, op_util_test_func_to_vec) {
  gert::Shape rt_shape({2, 3, 4});
  std::vector<int64_t> expect_vec = {2, 3, 4};
  EXPECT_EQ(ops::ToVector(rt_shape), expect_vec);
}

TEST_F(OpUtilTest, op_util_test_func_shape_to_str) {
  gert::Shape rt_shape({2, 3, 4});
  std::string expect_str = "[2, 3, 4]";
  EXPECT_EQ(ops::ToString(rt_shape), expect_str);
}

TEST_F(OpUtilTest, op_util_test_func_dtype_to_str) {
  ge::DataType input_dtype = ge::DT_FLOAT;
  std::string expect_str = "DT_FLOAT";
  EXPECT_EQ(ops::ToString(input_dtype), expect_str);
}

TEST_F(OpUtilTest, op_util_test_func_format_to_str) {
  ge::Format input_format = ge::FORMAT_NC1HWC0;
  std::string expect_str = "NC1HWC0";
  EXPECT_EQ(ops::ToString(input_format), expect_str);
}
