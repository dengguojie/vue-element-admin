#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "transformation_ops.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

// ---------------TransDataRNN-------------------
class TransDataRNNProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TransDataRNN Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TransDataRNN Proto Test TearDown" << std::endl;
  }
};


// base ut
/*
TEST_F(TransDataRnnProtoTest, TransDataRnnBaseTest) {
    ge::op::TransDataRnn transDataRnn;
    transDataRnn.UpdateInputDesc("src", create_desc_with_ori({17, 4000}, ge::DT_FLOAT16, ge::ND,{17, 4000},ge::ND));
    transDataRnn.UpdateOutputDesc("dst", create_desc_with_ori({2, 250, 16, 16}, ge::DT_FLOAT16, ge::FRACTAL_ZN_RNN,{2, 250, 16, 16},ge::FRACTAL_ZN_RNN));
    transDataRnn.SetAttr("src_format", "ND");
    transDataRnn.SetAttr("dst_format", "FRACTAL_ZN_RNN");
    transDataRnn.SetAttr("input_size", 17);
    transDataRnn.SetAttr("hidden_size", 4000);
    auto status = transDataRnn.VerifyAllAttr(true);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);

    auto ret = transDataRnn.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}*/
