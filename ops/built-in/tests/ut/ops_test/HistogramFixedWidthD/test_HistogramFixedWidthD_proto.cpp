#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "math_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class histogram_fixed_with_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "histogram_fixed_with_d Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "histogram_fixed_with_d Proto Test TearDown" << std::endl;
  }
};

TEST_F(histogram_fixed_with_d, histogram_fixed_with_d_infershape_1){
  // set input info
  auto input_shape = vector<int64_t>({2});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{2, 2}};
  auto test_format = ge::FORMAT_ND;
  int dtype = 1;
  int nbins=5;

  // expect result
  std::vector<int64_t> expected_shape = {5};

  // create desc
  auto input_desc = create_desc_shape_range(input_shape, ge::DT_INT32, test_format,
  input_shape, test_format, input_range);
  
  auto input_desc1 = create_desc_shape_range(input_shape, ge::DT_INT32, test_format,
  input_shape, test_format, input_range);

  // new op and do infershape
  ge::op::HistogramFixedWidthD op;
  op.UpdateInputDesc("x", input_desc);
  op.UpdateInputDesc("range", input_desc);
  op.SetAttr("nbins", nbins);
  op.SetAttr("dtype", dtype);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);

  //auto output_desc = op.GetOutputDesc("y");
  //EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  //EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}