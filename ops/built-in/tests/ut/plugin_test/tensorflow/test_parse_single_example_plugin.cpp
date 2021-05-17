#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class sparse_single_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensor_array_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensor_array_plugin_test TearDown" << std::endl;
  }
};

TEST_F(sparse_single_test, sparse_single_test_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/parse_single_case_1.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}