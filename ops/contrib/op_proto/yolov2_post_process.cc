/**
Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
Description: op_proto for interp caffe operator
Author:
Create: 2020-6-11
*/
#include "yolov2_post_process.h"
#include <vector>
#include <string>

namespace ge {
IMPLEMT_VERIFIER(Yolov2PostProcess, YoloV2PostProcessVerify)
{
    DataType input_type_0 = op.GetInputDesc(0).GetDataType();
    DataType input_type_1 = op.GetInputDesc(1).GetDataType();
    DataType input_type_2 = op.GetInputDesc(2).GetDataType();

    printf("[Plugin][INFO] Input type:{%d,%d,%d}, DT_FLOAT16=%d, DT_FLOAT=%d\n",
            input_type_0, input_type_1, input_type_2, DT_FLOAT16, DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(YoloV2PostProcessInferShape)
{
    TensorDesc tensordesc_output = op.GetOutputDesc("boxout_gm");

    vector<int64_t> dim_vec;
    dim_vec.push_back(100);
    dim_vec.push_back(8);
    ge::Shape output_shape = ge::Shape(dim_vec);

    tensordesc_output.SetShape(output_shape);
    tensordesc_output.SetDataType(op.GetInputDesc("scores_gm").GetDataType());
    tensordesc_output.SetFormat(op.GetInputDesc("scores_gm").GetFormat());

    printf("[Plugin][INFO] Set %s op output shape (%lu %lu).\n",
            TbeGetName(op).c_str(), dim_vec[0], dim_vec[1]);
    (void)op.UpdateOutputDesc("boxout_gm", tensordesc_output);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Yolov2PostProcess, YoloV2PostProcessInferShape);
VERIFY_FUNC_REG(Yolov2PostProcess, YoloV2PostProcessVerify);
}
