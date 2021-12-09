#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "nn_norm_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace testing;
using namespace ge;
using namespace fe;

namespace fe {
using namespace testing;
using namespace fe;
using namespace ge;

class fusion_mul_fusion_optimizer_pass_unittest : public testing::Test {
 public:
 protected:
  static void SetUpTestCase() {
    std::cout << "fusion_mul_fusion_optimizer_pass_unittest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fusion_mul_fusion_optimizer_pass_unittest TearDown" << std::endl;
  }

  static void CreateMulGraph(ComputeGraphPtr graph) {
    /*
     *        input 0 (NCHW)     input 1 (NC1HWC0)
     *               \           /
     *                 Mul(NCHW)
     *                    |
     *                   output(NCHW)
     *
     *  */
    OpDescPtr l2_loss_op = std::make_shared<OpDesc>("l2loss", "L2Loss");
    GeTensorDesc l2_loss_tensor_desc(GeShape({3, 1, 5, 6, 16}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    l2_loss_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    l2_loss_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    GeTensorDesc l2_loss_out_tensor_desc(GeShape({1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
    l2_loss_out_tensor_desc.SetOriginShape(GeShape({1}));
    l2_loss_out_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    l2_loss_op->AddInputDesc(l2_loss_tensor_desc);
    l2_loss_op->AddOutputDesc(l2_loss_out_tensor_desc);
    auto l2loss_node = graph->AddNode(l2_loss_op);


    OpDescPtr apply_momentum_op = std::make_shared<OpDesc>("am", "ApplyMomentum");
    GeTensorDesc am_tensor_desc(GeShape({3, 4, 5, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16);
    am_tensor_desc.SetOriginShape(GeShape({3, 4, 5, 6}));
    am_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    apply_momentum_op->AddOutputDesc(am_tensor_desc);
    for (uint32_t i = 0;  i < 5; i++) {
      apply_momentum_op->AddInputDesc(am_tensor_desc);
    }
    auto am_node = graph->AddNode(apply_momentum_op);

    OpDescPtr mul_o_p = std::make_shared<OpDesc>("mul", "Mul");
    mul_o_p->AddInputDesc(l2_loss_out_tensor_desc);
    mul_o_p->AddInputDesc(l2_loss_out_tensor_desc);
    mul_o_p->AddOutputDesc(l2_loss_out_tensor_desc);
    auto mul_Node = graph->AddNode(mul_o_p);

    OpDescPtr bias_add_o_p = std::make_shared<OpDesc>("bias_add", "BiasAddGrad");
    bias_add_o_p->AddInputDesc(am_tensor_desc);
    bias_add_o_p->AddOutputDesc(am_tensor_desc);
    auto bias_node = graph->AddNode(bias_add_o_p);

    GraphUtils::AddEdge(l2loss_node->GetOutDataAnchor(0), mul_Node->GetInDataAnchor(1));
    GraphUtils::AddEdge(am_node->GetOutDataAnchor(0), mul_Node->GetInDataAnchor(0));
    GraphUtils::AddEdge(mul_Node->GetOutDataAnchor(0), bias_node->GetInDataAnchor(0));
  }
};
}