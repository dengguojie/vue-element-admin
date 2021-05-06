/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: RoiAlign caffe opp header file
 * Author:
 * Create: 2020-6-11
 */

#include "roi_align.h"
#include <vector>

namespace ge {
IMPLEMT_INFERFUNC(ROIAlignTIK, ROIAlignInferShape) {
    auto pooledH = op.get_attr_pooled_h();
    auto pooledW = op.get_attr_pooled_w();
    auto roiShape = op.get_input_desc_rois().GetShape().GetDims();
    auto featureMapShape = op.get_input_desc_feature_map().GetShape().GetDims();
    int64_t inputN, inputC, poolH, poolW;

    inputN = roiShape[0];
    inputC = featureMapShape[1];
    poolH = pooledH;
    poolW = pooledW;
    vector<int64_t> roiAlignShape({inputN, inputC, poolH, poolW});

    auto outDesc = op.GetOutputDesc("y");
    outDesc.SetShape(Shape(roiAlignShape));
    (void)op.update_output_desc_y(outDesc);

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(ROIAlignTIK, ROIAlignVerify) {
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ROIAlignTIK, ROIAlignInferShape);

VERIFY_FUNC_REG(ROIAlignTIK, ROIAlignVerify);
}
