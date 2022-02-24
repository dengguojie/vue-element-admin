#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "nn_calculation_ops.h"
#include "array_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------Deconv test proto-------------------
class DilationProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Dilation Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Dilation Proto Test TearDown" << std::endl;
  }
};


// base ut
TEST_F(DilationProtoTest, dilationBaseCase) {
    ge::op::Dilation dilation;
    dilation.UpdateInputDesc("x", create_desc_with_ori({1, 1, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 3, 3}, ge::FORMAT_NCHW));
    dilation.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 6, 6}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 6, 6}, ge::FORMAT_NCHW));
    dilation.SetAttr("dilations", {1, 1, 2, 2});
    dilation.SetAttr("pads", {0, 1, 0, 1});
    auto status = dilation.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = dilation.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// error ut1, dilation size is not same with x
TEST_F(DilationProtoTest, dilationErrorCase1) {
    ge::op::Dilation dilation;
    dilation.UpdateInputDesc("x", create_desc_with_ori({1, 1, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 3, 3}, ge::FORMAT_NCHW));
    dilation.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 6, 6}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 6, 6}, ge::FORMAT_NCHW));
    dilation.SetAttr("dilations", {1, 1, 2, 2, 1});
    dilation.SetAttr("pads", {0, 1, 0, 1});
    auto status = dilation.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// error ut2, dilation is not positive
TEST_F(DilationProtoTest, dilationErrorCase2) {
    ge::op::Dilation dilation;
    dilation.UpdateInputDesc("x", create_desc_with_ori({1, 1, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 3, 3}, ge::FORMAT_NCHW));
    dilation.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 6, 6}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 6, 6}, ge::FORMAT_NCHW));
    dilation.SetAttr("dilations", {1, 1, 2, -2});
    dilation.SetAttr("pads", {0, 1, 0, 1});
    auto status = dilation.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// error ut3, pad size is 4
TEST_F(DilationProtoTest, dilationErrorCase3) {
    ge::op::Dilation dilation;
    dilation.UpdateInputDesc("x", create_desc_with_ori({1, 1, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 3, 3}, ge::FORMAT_NCHW));
    dilation.UpdateOutputDesc("y", create_desc_with_ori({1, 1, 6, 6}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1, 6, 6}, ge::FORMAT_NCHW));
    dilation.SetAttr("dilations", {1, 1, 2, -2});
    dilation.SetAttr("pads", {0, 1, 0, 1, 1});
    auto status = dilation.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
