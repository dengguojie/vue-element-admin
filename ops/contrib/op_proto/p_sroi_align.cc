/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: PsRoiAlign op proto cpp file
 * Author: Huawei
 * Create: 2020-6-11
 */

#include "p_sroi_align.h"
#include <vector>

namespace ge {

    IMPLEMT_INFERFUNC(PSROIAlign, PSROIAlignInferShape) {
        auto x_shape = op.GetInputDesc("feature_map").GetShape().GetDims();
        auto rois_shape = op.GetInputDesc("rois").GetShape().GetDims();
        auto group_size = op.get_attr_group_size();
        auto spatial_scale = op.get_attr_spatial_scale();
        
        printf("[INFO[PSROIAlign][spatical_scale]:%f\n", spatial_scale);
        
        DataType x_dtype = op.GetInputDesc("feature_map").GetDataType();
        TensorDesc y_desc = op.GetOutputDesc("output_map");
     
        int64_t N  = rois_shape[0];
        int64_t C = x_shape[1] / (group_size * group_size);
        int64_t H  = group_size;
        int64_t W  = group_size;
        
        x_shape[0] = N;
        x_shape[1] = C;
        x_shape[2] = H;
        x_shape[3] = W;
 
        y_desc.SetShape(ge::Shape(x_shape));
        y_desc.SetDataType(x_dtype);
        (void)op.UpdateOutputDesc("output_map", y_desc);
        
        return GRAPH_SUCCESS;
    }
    
    IMPLEMT_VERIFIER(PSROIAlign, PSROIAlignVerify) {
        return GRAPH_SUCCESS;
    }

    INFER_FUNC_REG(PSROIAlign, PSROIAlignInferShape);
    VERIFY_FUNC_REG(PSROIAlign, PSROIAlignVerify);
}  // namespace ge
