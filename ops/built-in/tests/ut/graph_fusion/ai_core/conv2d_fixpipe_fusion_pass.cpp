#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#define protected public
#define private public
#include "buffer_fusion/ub_fusion/ai_core/conv/tbe_conv2d_fixpipe_fusion_pass.h"
#include "common/util/platform_info.h"
#undef protected
#undef private

using namespace ge;

class ConvFixpipe : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Conv fixpipe pass SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Conv fixpipe pass TearDown" << std::endl;
  }
};

TEST_F(ConvFixpipe, ConvFixpipe_1) {
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_info;
  opti_info.soc_version = "Ascend920A";
  platform_info.ai_core_spec.cube_vector_split = 1;
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend920A"] = platform_info;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_info);
  OpDescPtr cast = std::make_shared<OpDesc>("cast", "Cast");

  vector<int64_t> dim({40, 25, 7, 7});
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetOriginFormat(FORMAT_NCHW);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetOriginShape(shape);
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetShape(shape);

  cast->AddInputDesc("x", tensor_desc);
  cast->AddOutputDesc("y", tensor_desc);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  NodePtr cast_node = graph->AddNode(cast);

  fe::TbeConv2DFixpipeFusionPass fusion_pass;
  vector<fe::BufferFusionPattern*> patterns = fusion_pass.DefinePatterns();

  fe::BufferFusionOpDesc fusion_desc;
  fusion_desc.desc_name = "elemwise";
  const fe::BufferFusionOpDesc *fusion_desc_ptr = &fusion_desc;
  fe::BufferFusionMapping mapping;
  vector<NodePtr> elemwise_nodes = {cast_node};
  mapping.emplace(fusion_desc_ptr, elemwise_nodes);
  vector<NodePtr> fusion_nodes;

  fusion_pass.GetFusionNodes(mapping, fusion_nodes);
  fusion_desc_ptr = nullptr;
}

