#include "nn_batch_norm_ops.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "gtest/gtest.h"

using namespace ge;
using namespace op;

class n_p_u_get_float_status_v2_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "n_p_u_get_float_status_v2 SetUp" << std::endl; }

  static void TearDownTestCase() {
    std::cout << "n_p_u_get_float_status_v2 TearDown" << std::endl;
  }
};

TEST_F(n_p_u_get_float_status_v2_infer_test, n_p_u_get_float_status_v2_infer_test_1) {
  auto graph = std::make_shared<ge::ComputeGraph>("n_p_u_get_float_status_v2_infer_test_1");

  ge::GeShape output_shape({8});
  ge::GeTensorDesc output_desc(output_shape, ge::FORMAT_ND, ge::DT_FLOAT);
  output_desc.SetOriginFormat(ge::FORMAT_ND);
  output_desc.SetOriginDataType(ge::DT_FLOAT);
  output_desc.SetOriginShape(output_shape);

  ge::OpDescPtr n_p_u_get_float_status_v2 = std::make_shared<ge::OpDesc>("NPUGetFloatStatusV2", "NPUGetFloatStatusV2");
  n_p_u_get_float_status_v2->AddOutputDesc(output_desc);
  n_p_u_get_float_status_v2->AddInputDesc(output_desc);
  ge::NodePtr n_p_u_get_float_status_v2_node = graph->AddNode(n_p_u_get_float_status_v2);

  fe::FusionPassTestUtils::InferShapeAndType(graph);
}