#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"
#include "register/register_error_codes.h"

class npu_onnx_graph_op_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
  }

  static void TearDownTestCase() {
  }
};
namespace domi {
Status AutoMappingNpuOnnxGraphOpPartitionedCall(const ge::Operator &op_src, ge::Operator &op_dest);
}
TEST_F(npu_onnx_graph_op_test, npu_onnx_graph_op_test_case_1) {
  ge::Operator op_src;
  ge::Operator op_dest;
  domi::AutoMappingNpuOnnxGraphOpPartitionedCall(op_src, op_dest);
}

