#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tensorflow_parser.h"

using namespace ge;

class max_pool3d_grad_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "max_pool3d_grad_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "max_pool3d_grad_plugin_test TearDown" << std::endl;
  }
};

TEST_F(max_pool3d_grad_plugin_test, max_pool3d_grad_plugin_test_1) {
  ge::Graph graph;

  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/maxPool3DGrad_case_1.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}
