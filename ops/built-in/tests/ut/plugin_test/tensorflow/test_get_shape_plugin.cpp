#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tensorflow_parser.h"

using namespace ge;

class get_shape_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "get_shape_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "get_shape_plugin_test TearDown" << std::endl;
  }
};

TEST_F(get_shape_test, get_shape_testcase_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/get_shape_plugin_1.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
