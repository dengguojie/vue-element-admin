#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class histogram_fixed_width_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "xxxxxxhistogram_fixed_width_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "xxxxhistogram_fixed_width_plugin_test TearDown" << std::endl;
  }
};

TEST_F(histogram_fixed_width_test, histogram_fixed_width_test_case_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/histogram_fixed_width.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  // check op count, some op need check op attr, op input count.
}
