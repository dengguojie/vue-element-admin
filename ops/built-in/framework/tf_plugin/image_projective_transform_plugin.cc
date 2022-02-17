#include "register/register.h"
#include "tensor.h"

#include "op_log.h"

#include <iostream>
namespace domi {
Status ParseImageProjectiveTransform(const Message* op_src, ge::Operator& op) {
  AutoMappingFn(op_src, op);
  ge::TensorDesc input_tensor = op.GetInputDesc("images");
  input_tensor.SetOriginFormat(ge::FORMAT_NHWC);
  input_tensor.SetFormat(ge::FORMAT_NHWC);
  if (op.UpdateInputDesc("images", input_tensor) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update input format failed.");
    return FAILED;
  }
  ge::TensorDesc output_tensor = op.GetOutputDesc("transformed_images");
  output_tensor.SetOriginFormat(ge::FORMAT_NHWC);
  output_tensor.SetFormat(ge::FORMAT_NHWC);
  if (op.UpdateOutputDesc("transformed_images", output_tensor) != ge::GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "update output format failed.");
    return FAILED;
  }
  return SUCCESS;
}
// register ImageProjectiveTransform op to GE
REGISTER_CUSTOM_OP("ImageProjectiveTransform")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ImageProjectiveTransformV2")
    .ParseParamsFn(ParseImageProjectiveTransform)
    .ImplyType(ImplyType::AI_CPU);
}  // namespace domi
