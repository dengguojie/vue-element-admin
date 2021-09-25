#include "gtest/gtest.h"
#include "onnx_parser.h"
#include "parser_common.h"

using namespace ge;

class yolov5_onnx_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "yolov5_onnx_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "yolov5_onnx_plugin_test TearDown" << std::endl;
  }
};

TEST_F(yolov5_onnx_plugin_test, yolov5_onnx_plugin_test_case_1) {
  CleanGlobal();
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_yolov5_V11.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}
