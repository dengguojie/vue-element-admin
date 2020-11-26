#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_calculation_ops.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"
#include "fp16_t.hpp"

using namespace ge;

namespace fe {

class split_conv2d_concat_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "split_conv2d_concat_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "split_conv2d_concat_test TearDown" << std::endl;
  }

  /************************************
   *
   *               split
   *       filter    /\     filter
   *           \  /       \ /
   *         conv2d ... conv2d
   *              \       /
   *                 \/
   *              concat(v2)
   *                 |
   *
   *************************************/
  void BuildGraph(ComputeGraphPtr &compute_graph) {
    ge::Graph graph("test");
    auto input0 = op::Data("input0");
    ge::Shape shape_x({1, 7, 7, 32});
    ge::TensorDesc tensorDescX(shape_x, FORMAT_NHWC, DT_FLOAT16);
    input0.update_input_desc_x(tensorDescX);
    input0.update_output_desc_y(tensorDescX);

    auto split_dim = op::Const("split_dim");
    Tensor axis;
    int32_t * dataValue = new int32_t[1];
    * dataValue = 3;
    axis.SetTensorDesc(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));
    axis.SetData((uint8_t*)dataValue, 4);
    split_dim.set_attr_value(axis);
    split_dim.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto split_layer = op::Split("split");
    split_layer.set_input_split_dim(split_dim)
               .set_input_x(input0)
               .set_attr_num_split(2)
               .create_dynamic_output_y(2);

    TensorDesc filterDesc(ge::Shape({3,3,16,16}), FORMAT_HWCN, DT_FLOAT16);
    filterDesc.SetOriginFormat(FORMAT_HWCN);

    auto filter_0 = op::Const("filter_0");
    Tensor filter;
    fp16_t * filterValue = new fp16_t[3*3*16*16];
    filter.SetTensorDesc(filterDesc);
    filter.SetData((uint8_t*)filterValue, 2*3*3*16*16);
    filter_0.set_attr_value(filter);
    filter_0.update_output_desc_y(filterDesc);

    auto conv2d_0_layer = op::Conv2D("conv2d_0");
    conv2d_0_layer.set_input_x(split_layer, "y0")
                  .set_input_filter(filter_0)
                  .set_attr_strides({0,1,1,0})
                  .set_attr_pads({0,0,0,0});

    auto filter_1 = op::Const("filter_1");
    filter_1.set_attr_value(filter);
    filter_1.update_output_desc_y(filterDesc);

    auto conv2d_1_layer = op::Conv2D("conv2d_1");
    conv2d_1_layer.set_input_x(split_layer, "y1")
                  .set_input_filter(filter_1)
                  .set_attr_strides({0,1,1,0})
                  .set_attr_pads({0,0,0,0});

    auto concat_dim = op::Const("concat_dim");
    concat_dim.set_attr_value(axis);
    concat_dim.update_output_desc_y(TensorDesc(ge::Shape({1}), FORMAT_NHWC, DT_INT32));

    auto concat_layer = op::Concat("concat");
    concat_layer.set_input_concat_dim(concat_dim)
                .create_dynamic_input_x(2)
                .set_dynamic_input_x(0, conv2d_0_layer)
                .set_dynamic_input_x(1, conv2d_1_layer)
                .set_attr_N(2);

    delete[] dataValue;
    delete[] filterValue;

    std::vector<Operator> inputs{input0, split_dim, filter_0, filter_1, concat_dim};
    std::vector<Operator> outputs{concat_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    compute_graph = ge::GraphUtils::GetComputeGraph(graph);
  }
};

// TEST_F(split_conv2d_concat_test, split_conv2d_concat_test_1) {
//   ge::ComputeGraphPtr compute_graph;
//   BuildGraph(compute_graph);
//   FusionPassTestUtils::InferShapeAndType(compute_graph);
//   FusionPassTestUtils::RunGraphFusionPass("ASplitConv2dConcatPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph);
//   bool findSplit = false;
//   bool findConcat = false;
//   for (auto node: compute_graph->GetAllNodes()) {
//     if (node->GetType() == "Split") {
//         findSplit = true;
//     }
//     if (node->GetType() == "Concat") {
//         findConcat = true;
//     }
//   }
//   EXPECT_EQ(findSplit, false);
//   EXPECT_EQ(findConcat, false);
// }
} // namespace fe
