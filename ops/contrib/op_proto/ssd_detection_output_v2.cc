/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: ssd detection output op proto cpp file
 * Author:
 * Create: 2020-6-11
 * Note:
 */

#include <vector>
#include "ssd_detection_output_v2.h"

// namespace ge
namespace ge {
    IMPLEMT_INFERFUNC(SSDDetectionOutputV2, SSDDetectionOutputV2InferShape) {
        auto keepTopK = op.get_attr_keep_top_k();
        if (keepTopK == -1) {
            keepTopK = 1024;
        }

        auto locShape = op.get_input_desc_bbox_delta().GetShape().GetDims();
        if (locShape.empty()) {
            printf("get mbox loc failed.");
            return GRAPH_FAILED;
        }
        auto batchSize = locShape[0];
        auto boxType = op.get_input_desc_bbox_delta().GetDataType();

        vector<int64_t> actualNumShape({batchSize, 8});
        auto outDescBoxNum = op.GetOutputDesc("out_boxnum");
        outDescBoxNum.SetShape(Shape(actualNumShape));
        outDescBoxNum.SetDataType(ge::DT_INT32);
        (void)op.update_output_desc_out_boxnum(outDescBoxNum);
		
        const int alignLength = 128;
        keepTopK = ((keepTopK + alignLength - 1) / alignLength) * alignLength;
        vector<int64_t> boxShape({batchSize, keepTopK, 8});
        auto outDesc = op.GetOutputDesc("y");
        outDesc.SetShape(Shape(boxShape));
        outDesc.SetDataType(ge::DataType(boxType));
        (void)op.update_output_desc_y(outDesc);

        return GRAPH_SUCCESS;
    }
    IMPLEMT_VERIFIER(SSDDetectionOutputV2, SSDDetectionOutputV2Verify) {

        return GRAPH_SUCCESS;
    }
    INFER_FUNC_REG(SSDDetectionOutputV2, SSDDetectionOutputV2InferShape);
    VERIFY_FUNC_REG(SSDDetectionOutputV2, SSDDetectionOutputV2Verify);
}
