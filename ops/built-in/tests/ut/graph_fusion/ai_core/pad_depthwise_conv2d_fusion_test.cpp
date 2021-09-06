#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class pad_dwconv2d_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "pad_dwconv2d_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "pad_dwconv2d_fusion_test TearDown" << std::endl;
  }
};

TEST_F(pad_dwconv2d_fusion_test, one_pad_one_dwconv2d) {
  std::cout << "enter pad_dwconv2d_fusion_test.one_pad_one_dwconv2d" << std::endl;
  /*
   * data    padding_const
   *   \         /
   *    \       /
   *     \     /
   *      \   /
   *       pad    dwconv2d_filter_const
   *        |     /
   *        |    /
   *        dwconv2d
   * */

  ge::Graph graph("one_pad_one_dwconv2d");

  // data
  auto data = op::Data("data");
  ge::Shape data_shape({1, 224, 224, 3});
  ge::TensorDesc data_tensor_desc(data_shape, FORMAT_NHWC, DT_FLOAT);
  data_tensor_desc.SetOriginFormat(FORMAT_NHWC);
  data.update_input_desc_x(data_tensor_desc);
  data.update_output_desc_y(data_tensor_desc);

  // padding_const
  std::vector<vector<int32_t>> paddings = {{0,0}, {3,3}, {3,3}, {0,0}};
  vector<int64_t> paddings_dims = {4, 2};
  TensorDesc padding_const_tensor_desc(ge::Shape(paddings_dims), FORMAT_NHWC, DT_INT32);
  Tensor padding_const_tensor(padding_const_tensor_desc);
  uint32_t* padding_const_tensor_value = new uint32_t[8];
  for (size_t dim = 0; dim < 8; dim++) {
    *(padding_const_tensor_value + dim) = paddings[dim / 2][dim % 2];
  }
  padding_const_tensor.SetData((uint8_t*)padding_const_tensor_value, 8 * sizeof(uint32_t));
  auto paddings_const = op::Const("paddings").set_attr_value(padding_const_tensor);

  // pad
  auto pad = op::Pad("pad").set_input_x(data).set_input_paddings(paddings_const);
  pad.update_input_desc_x(data_tensor_desc);
  pad.update_input_desc_paddings(padding_const_tensor_desc);
  ge::Shape pad_output_shape({1, 230, 230, 3});
  ge::TensorDesc pad_output_tensor_desc(pad_output_shape, FORMAT_NHWC, DT_FLOAT);
  pad.update_output_desc_y(pad_output_tensor_desc);

  // dwconv2d_filter_const
  TensorDesc filterDesc(ge::Shape({7,7,3,64}), FORMAT_HWCN, DT_FLOAT);
  filterDesc.SetOriginFormat(FORMAT_HWCN);
  auto dwconv2d_filter_const = op::Const("dwconv2d_filter_const");
  Tensor filter;
  float * filterValue = new float[7*7*3*64];
  filter.SetTensorDesc(filterDesc);
  filter.SetData((uint8_t*)filterValue, 4*7*7*3*64);
  dwconv2d_filter_const.set_attr_value(filter);

  // dwconv2d
  auto dwconv2d = op::DepthwiseConv2D("dwconv2d");
  dwconv2d.set_input_x(pad)
  .set_input_filter(dwconv2d_filter_const)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  auto dwconv2d_op_dsc = ge::OpDescUtils::GetOpDescFromOperator(dwconv2d);
  (void)ge::AttrUtils::SetStr(dwconv2d_op_dsc, "padding", "VALID");
  dwconv2d.update_input_desc_x(pad_output_tensor_desc);
  dwconv2d.update_input_desc_filter(filterDesc);
  ge::Shape dwconv2d_output_shape({1, 112, 112, 64});
  ge::TensorDesc dwconv2d_output_tensor_desc(dwconv2d_output_shape, FORMAT_NHWC, DT_FLOAT);
  dwconv2d.update_output_desc_y(dwconv2d_output_tensor_desc);


  std::vector<Operator> inputs = {data,paddings_const,dwconv2d_filter_const};
  std::vector<Operator> outputs = {dwconv2d};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  // check before fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 5);

  // do fusion
  fe::FusionPassTestUtils::RunGraphFusionPass("PadDepthwiseConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  // check after fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 3);

  bool has_pad_node = false;
  vector<int64_t> pads;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Pad") {
      has_pad_node = true;
    }
    if (node->GetType() == "DepthwiseConv2D") {
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), "pads", pads);
    }
  }
  EXPECT_EQ(has_pad_node, false);
  for (auto pad : pads) {
    EXPECT_EQ(pad, 3);
  }
}