#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class depthwiseconv2dbackpropfilter_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "depthwiseconv2dbackpropfilter_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "depthwiseconv2dbackpropfilter_plugin_test TearDown" << std::endl;
  }
};

TEST_F(depthwiseconv2dbackpropfilter_plugin_test, depthwiseconv2dbackpropfilter_plugin_test_1) {
  ge::Graph graph;

  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/depthwiseconv2dbackpropfilter_case_1.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
