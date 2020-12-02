#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class add_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "add_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "add_plugin_test TearDown" << std::endl;
  }
};

TEST_F(add_plugin_test, add_plugin_test_case_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/add_case_1.txt";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);
}