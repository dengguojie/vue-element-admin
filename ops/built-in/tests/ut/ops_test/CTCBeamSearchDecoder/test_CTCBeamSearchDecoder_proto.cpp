#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "ctc_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class CTCBeamSearchDecoder_infer_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "CTCBeamSearchDecoder_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "CTCBeamSearchDecoder_infer_test TearDown" << std::endl;
  }
};

TEST_F(CTCBeamSearchDecoder_infer_test, CTCBeamSearchDecoder_infer_test_1) {


  // new op and do infershape
  ge::op::CTCBeamSearchDecoder op;
  op.UpdateInputDesc("inputs", create_desc({2,3}, ge::DT_FLOAT));
  op.UpdateInputDesc("sequence_length", create_desc({2}, ge::DT_FLOAT));


  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CTCBeamSearchDecoder_infer_test, CTCBeamSearchDecoder_infer_test_2) {


  // new op and do infershape
  ge::op::CTCBeamSearchDecoder op;
  op.UpdateInputDesc("inputs", create_desc({2,3,3}, ge::DT_FLOAT));
  op.UpdateInputDesc("sequence_length", create_desc({2,3}, ge::DT_FLOAT));
  

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CTCBeamSearchDecoder_infer_test, CTCBeamSearchDecoder_infer_test_3) {


  // new op and do infershape
  ge::op::CTCBeamSearchDecoder op;
  op.UpdateInputDesc("inputs", create_desc({2,3,3}, ge::DT_FLOAT));
  op.UpdateInputDesc("sequence_length", create_desc({2}, ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(CTCBeamSearchDecoder_infer_test, CTCBeamSearchDecoder_infer_test_4) {
  // new op and do infershape
  ge::op::CTCBeamSearchDecoder op;
  op.UpdateInputDesc("inputs", create_desc({2,3,3}, ge::DT_FLOAT));
  op.UpdateInputDesc("sequence_length", create_desc({3}, ge::DT_FLOAT));
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
