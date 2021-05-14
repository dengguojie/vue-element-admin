/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: faster rcnn cropandresize op proto cpp file
 * Author:
 * Create: 2020-6-17
 */

#include "fasterrcnn_cropandresize.h"
#include <string>
#include <vector>

namespace ge {
    enum DimNum {
        DIM_N = 0,
        DIM_C = 1,
        DIM_H = 2,
        DIM_W = 3
    };
    
    IMPLEMT_VERIFIER(FasterrcnnCropandresizeTik, FasterrcnnCropResizeVerity)
    {
        printf("[Plugin][Info] here to verify FasterrcnnCropandresizeTik OP\n");
        if (op.GetInputsSize() != 2) {
            printf("[ERROR][Plugin] number of input must be 2\n");
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }

    IMPLEMT_COMMON_INFERFUNC(FasterrcnnCropResizeInferShape)
    {
        TensorDesc tensorDesc = op.GetOutputDesc("cropandresize_gm");
        ge::Shape featMapShape = op.GetInputDesc("featuremap_gm").GetShape();
        vector<int64_t> vecMatchOne {1, 38, 64, 1024};
        vector<int64_t> vecMatchTwo {1, 40, 128, 1088};
        vector<int64_t> vecMapBoxOne {100, 4};
        vector<int64_t> vecMapBoxTwo {300, 4};
        if (featMapShape.GetDims() == vecMatchOne) {
            if (op.GetInputDesc(1).GetShape().GetDims() != vecMapBoxOne) {
                printf("[ERROR][PLUGIN]input shape is not valid, please check hearder file\n");
                return GRAPH_FAILED;
            }
            vector<int64_t> ouputShape(4, 0);
            ouputShape[DIM_N] = 100;
            ouputShape[DIM_C] = 14;
            ouputShape[DIM_H] = 14;
            ouputShape[DIM_W] = 1024;
            tensorDesc.SetShape((Shape)ouputShape);
            tensorDesc.SetDataType(op.GetInputDesc("featuremap_gm").GetDataType());
            tensorDesc.SetFormat(FORMAT_NHWC);
            (void)op.UpdateOutputDesc("cropandresize_gm", tensorDesc);
            return GRAPH_SUCCESS;
        }

        if (featMapShape.GetDims() == vecMatchTwo) {
            if (op.GetInputDesc(1).GetShape().GetDims() != vecMapBoxTwo) {
                printf("[ERROR][PLUGIN]input shape is not valid, please check hearder file\n");
                return GRAPH_FAILED;
            }
            vector<int64_t> ouputShape(4, 0);
            ouputShape[DIM_N] = 300;
            ouputShape[DIM_C] = 17;
            ouputShape[DIM_H] = 17;
            ouputShape[DIM_W] = 1088;
            tensorDesc.SetShape((Shape)ouputShape);
            tensorDesc.SetDataType(op.GetInputDesc("featuremap_gm").GetDataType());
            tensorDesc.SetFormat(FORMAT_NHWC);
            (void)op.UpdateOutputDesc("cropandresize_gm", tensorDesc);
            return GRAPH_SUCCESS;
        }
        printf("[ERROR][PLUGIN]input shape is not valid, please check hearder file\n");
        return GRAPH_FAILED;
    }

    COMMON_INFER_FUNC_REG(FasterrcnnCropandresizeTik, FasterrcnnCropResizeInferShape);
    VERIFY_FUNC_REG(FasterrcnnCropandresizeTik, FasterrcnnCropResizeVerity);
}
