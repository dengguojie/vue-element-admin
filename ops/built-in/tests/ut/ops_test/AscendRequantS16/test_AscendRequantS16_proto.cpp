#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "quantize_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class ascend_requant_s16_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "ascend_requant_s16_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ascend_requant_s16_infer_test TearDown" << std::endl;
  }
};

//REG_OP(AscendRequantS16)
//  .INPUT(x0, TensorType({DT_INT16}))
//  .INPUT(req_scale, TensorType({DT_UINT64}))
//  .OPTIONAL_INPUT(x1, TensorType({DT_INT16}))
//  .OUTPUT(y0, TensorType({DT_INT8}))
//  .OUTPUT(y1, TensorType({DT_INT16}))
//  .ATTR(dual_output, Bool, false)
//  .ATTR(relu_flag, Bool, false)
//  .OP_END_FACTORY_REG(AscendRequantS16)

TEST_F(ascend_requant_s16_infer_test, ascend_requant_s16_infer_test_1) {
  bool relu_flag = false;
  bool dual_output = false;

  // expect result
  std::vector<int64_t> expected_shape = {3, 16, 16, 64};

  // new op and do infershape
  ge::op::AscendRequantS16 op;
  op.UpdateInputDesc("x0", create_desc_with_ori({3, 16, 16, 64}, ge::DT_INT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("req_scale", create_desc_with_ori({1, 1, 1, 64}, ge::DT_UINT64, ge::FORMAT_NHWC, {3, 3, 64}, ge::FORMAT_NHWC));
  op.set_attr_dual_output(dual_output);
  op.set_attr_relu_flag(relu_flag);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y0");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}