#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class non_zero_with_value_shape_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "non_zero_with_value_shape_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "non_zero_with_value_shape_plugin_test TearDown" << std::endl;
  }
};

TEST_F(non_zero_with_value_shape_test, non_zero_with_value_shape_plugin_test_1) {
  ge::Graph graph;

  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/non_zero_with_value_shape_plugin_1.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}
