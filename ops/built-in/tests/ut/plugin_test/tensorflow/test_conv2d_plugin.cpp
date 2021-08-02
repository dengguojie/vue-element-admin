#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tensorflow_parser.h"


class conv2d_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "conv2d_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "conv2d_plugin_test TearDown" << std::endl;
  }
};

TEST_F(conv2d_plugin_test, conv2d_plugin_test_explicit_pad) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/conv2d_explicit_pad.pb";

  auto status = ge::aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
