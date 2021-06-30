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

class AscendQuantProto : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "AscendQuantProto SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AscendQuantProto TearDown" << std::endl;
  }
};

TEST_F(AscendQuantProto, AscendQuant_proto_0) {
  // set input info
  auto shape_x1 = vector<int64_t>({3, -1, -1, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{3, 3}, {12, 51}, {17, 60}, {5, 5}};
  
  auto scale = 1.0;
  auto offset = -3.0;
  auto round_mode = "Round";
  auto sqrt_mode = false;
  auto test_format = ge::FORMAT_NHWC;

  // expect result
  std::vector<int64_t> expected_shape = {3, -1, -1, 5};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{3, 3}, {12, 51}, {17, 60}, {5, 5}};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, test_format, shape_x1, test_format, range_x1);

  // new op and do infershape
  ge::op::AscendQuant op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.set_attr_scale(scale);
  op.set_attr_offset(offset);
  op.set_attr_sqrt_mode(sqrt_mode);
  op.set_attr_round_mode(round_mode);
  op.set_attr_dst_type(2);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}