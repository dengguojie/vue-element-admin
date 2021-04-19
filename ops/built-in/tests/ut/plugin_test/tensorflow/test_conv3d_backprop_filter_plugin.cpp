#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "tensorflow_parser.h"

using namespace ge;

class conv3d_bp_filter_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "conv3d_bp_filter_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "conv3d_bp_filter_plugin_test TearDown" << std::endl;
  }
};

TEST_F(conv3d_bp_filter_plugin_test, conv3d_bp_filter_plugin_test_1) {
  ge::Graph graph;

  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/conv3d_bp_filter_plugin_test_1.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}
