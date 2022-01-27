#include "array_ops.h"
#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "nn_calculation_ops.h"
#include "transformation_ops.h"
#define private public
#define protected public
#include "buffer_fusion/ub_fusion/ai_core/conv2d_backprop_input/tbe_dw_transdata_fusion_pass.h"
#include "fusion_pass_test_utils.h"
#include "fusion_pass_test_slice_utils.h"


using namespace ge;
using namespace op;

class dw_transdata_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dw_transdata_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dw_transdata_fusion_test TearDown" << std::endl;
    }
};

namespace fe {
  static Status RunBufferFusionPass(string fusionPassName, BufferFusionPassType passType,
    ge::ComputeGraphPtr& compute_graph_ptr, vector <ge::NodePtr> &fusion_nodes) {
      std::map<string, BufferFusionPassRegistry::CreateFn> createFns =
          BufferFusionPassRegistry::GetInstance().GetCreateFnByType(passType);
      const auto &iter = createFns.find(fusionPassName);
      if (iter != createFns.end()) {
          if (passType == fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) {
              auto BufferFusionPassBasePtr = std::unique_ptr<TbeDwTransDataFusionPass>(
                      dynamic_cast<TbeDwTransDataFusionPass *>(iter->second()));
              if (BufferFusionPassBasePtr == nullptr) {
                  return FAILED;
              }
              ge::ComputeGraph::Vistor<ge::NodePtr> NodePtrs = compute_graph_ptr->GetAllNodes();
              std::vector<ge::NodePtr> Node_v(NodePtrs.begin(), NodePtrs.end());

              BufferFusionPassBasePtr->SetName(fusionPassName);
              vector<BufferFusionPattern*> patterns = BufferFusionPassBasePtr->DefinePatterns();
              for (auto pattern : patterns) {
                std::vector<BufferFusionOpDesc *> desc = pattern->GetOpDescs();
                vector<ge::NodePtr> transdataNode;
                vector<ge::NodePtr> dwNodes;
                for (auto i : NodePtrs) {
                  auto opDesc = i->GetOpDesc();
                  if (opDesc->GetType() == "Conv2DBackpropFilter") {
                    dwNodes.push_back(i);
                  }
                  if (opDesc->GetType() == "TransData") {
                    transdataNode.push_back(i);
                  }
                }

                BufferFusionMapping mapping;
                for (auto i : desc) {
                  if (i->desc_name == "conv2d_backprop_filter") {
                    mapping[i] = dwNodes;
                  }
                  if (i->desc_name == "transdata1") {
                    mapping[i] = transdataNode;
                  }
                }
                BufferFusionPassBasePtr->GetFusionNodes(mapping, fusion_nodes);
              }
              return SUCCESS;
          }
      }

      return FAILED;

  }
}
TEST_F(dw_transdata_fusion_test, dw_transdata_fusion_test_0) {
    ge::Graph graph("dw_transdata_fusion_test_0");

    auto fmap_shape = vector<int64_t>({-1, -1, -1, -1});
    std:vector<std::pair<int64_t, int64_t>> unlimited_range = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
    ge::TensorDesc fmap_desc(ge::Shape(fmap_shape), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    fmap_desc.SetShapeRange(unlimited_range);
    auto data_fmap = op::Data("data_fmap");
    auto data_dedy = op::Data("data_dedy");
    data_fmap.update_input_desc_x(fmap_desc);
    data_fmap.update_output_desc_y(fmap_desc);
    data_dedy.update_input_desc_x(fmap_desc);
    data_dedy.update_output_desc_y(fmap_desc);

    auto filter_data = op::Const("filter_size");
    auto filter_size_shape = vector<int64_t>({4});
    ge::TensorDesc filter_size_desc(ge::Shape(filter_size_shape), ge::FORMAT_NCHW, ge::DT_INT32);
    Tensor filter_size;
    int32_t *filter_size_value = new int32_t[4];
    filter_size.SetTensorDesc(filter_size_desc);
    filter_size.SetData((uint8_t *)filter_size_value, 4 * sizeof(int32_t));
    filter_data.set_attr_value(filter_size);
    filter_data.update_output_desc_y(filter_size_desc);

    auto trans_data_shape = vector<int64_t>({-1, -1, -1, -1, 16});
    auto transdata1 = op::TransData("transdata1").set_input_src(data_fmap);
    auto transdata2 = op::TransData("transdata2").set_input_src(data_dedy);
    ge::TensorDesc trans_data_output_desc(ge::Shape(trans_data_shape), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    transdata1.update_input_desc_src(fmap_desc);
    transdata1.update_output_desc_dst(trans_data_output_desc);
    transdata2.update_input_desc_src(fmap_desc);
    transdata2.update_output_desc_dst(trans_data_output_desc);

    auto dw = op::Conv2DBackpropFilter("conv2d_backprop_filter")
                  .set_input_x(transdata1)
                  .set_input_filter_size(filter_data)
                  .set_input_out_backprop(transdata2)
                  .set_attr_strides({-1, -1, -1, -1})
                  .set_attr_pads({-1, -1, -1, -1})
                  .set_attr_dilations({1, 1, 1, 1})
                  .set_attr_groups({1})
                  .set_attr_data_format("NCHW");

    auto dw_out_shape = vector<int64_t>({-1, -1, -1, -1});
    TensorDesc dw_output_desc_y(ge::Shape(dw_out_shape), FORMAT_FRACTAL_Z, DT_FLOAT);
    dw.update_input_desc_x(trans_data_output_desc);
    dw.update_input_desc_filter_size(filter_size_desc);
    dw.update_input_desc_out_backprop(trans_data_output_desc);
    dw.update_output_desc_y(dw_output_desc_y);

    auto transdata3 = op::TransData("transdata3").set_input_src(dw);
    auto transdata3_shape = vector<int64_t>({-1, -1, -1, -1});
    ge::TensorDesc transdata3_output_desc(ge::Shape(transdata3_shape), ge::FORMAT_NCHW, ge::DT_FLOAT);
    transdata3.update_input_desc_src(dw_output_desc_y);
    transdata3.update_output_desc_dst(transdata3_output_desc);

    std::vector<Operator> inputs{data_fmap, data_dedy, filter_data};
    std::vector<Operator> outputs{transdata3};

    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    vector <ge::NodePtr> fusion_nodes;
    Status res = fe::RunBufferFusionPass("TbeDwTransDataFusionPass",
                                         fe::BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                                         compute_graph_ptr, fusion_nodes);
    bool find_dw = false;
    bool find_td = false;
    for (auto fusion_node: fusion_nodes) {
      if (fusion_node->GetType() == "Conv2DBackpropFilter") {
        find_dw = true;
      }
      if (fusion_node->GetType() == "TransData") {
        find_td = true;
      }
    }
    EXPECT_EQ(find_dw, true);
    EXPECT_EQ(find_td, true);
    EXPECT_EQ(res, SUCCESS);
}
