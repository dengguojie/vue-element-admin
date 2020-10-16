#include "inc/reduce_ops.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "./util/error_util.h"
#include "register/register.h"
#include "common/util/error_manager/error_manager.h"


namespace ge {

static bool InferReductionShape(const ge::Operator& operation, const string& input_name, ge::TensorDesc& result_desc)
{
    result_desc = operation.GetInputDesc(input_name);
    auto shape = result_desc.GetShape();
    int64_t dimNum = shape.GetDimNum();
    int64_t axis = 0;
    int64_t idx = 0;

    if(ge::GRAPH_SUCCESS != operation.GetAttr("axis", axis))
    {
        OP_LOGE("Reduction", "Get axis failed!");
        OpsGetAttrErrReport(operation.GetName().c_str(),"axis");
        return false;
    }

    if(axis < -dimNum || axis >= dimNum)
    {
        OP_LOGE("Reduction", "The range of the axis must be between %ld and %ld !", -dimNum, dimNum - 1);
        string minvalue = Strcat(-dimNum);
        string maxvalue = Strcat(dimNum - 1);
        map<string, string> err_map;
        err_map["op_name"] = "Reduction";
        err_map["param_name"] = "axis";
        err_map["excepted_value"] = Strcat("in the range of[",minvalue,",",maxvalue,"]");
        err_map["input_value"] = Strcat(axis);
        std::string report_error_code = "E70007";
        ErrorManager::GetInstance().ReportErrMessage(report_error_code, err_map);
        return false;
    }

    if(axis < 0)
    {
        axis += dimNum;
    }

    for(idx = axis; idx < dimNum; idx++)
    {
        shape.SetDim(idx, 1);
    }
    result_desc.SetShape(shape);

    return true;
}

IMPLEMT_COMMON_INFERFUNC(ReductionInferShape) {
  OP_LOGI(op.GetName().c_str(), "Enter Reduction proto inferfunction!");
  ge::TensorDesc result_desc;
  if (!InferReductionShape(op, "x", result_desc)) {
    return GRAPH_FAILED;
  }

  auto shape = result_desc.GetShape();
  auto dtype = result_desc.GetDataType();

  // update output desc
  ge::TensorDesc output_desc = op.GetOutputDesc("y");
  output_desc.SetShape(shape);
  output_desc.SetDataType(dtype);
  (void)op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Reduction, ReductionVerify) {
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Reduction, ReductionInferShape);

VERIFY_FUNC_REG(Reduction, ReductionVerify);
} // namespace ge
