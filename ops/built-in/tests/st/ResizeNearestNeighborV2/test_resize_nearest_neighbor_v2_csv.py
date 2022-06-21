import math
import numpy as np
from tbetoolkits import UniversalTestcaseStructure
from tbetoolkits.plugins import Plugins

@Plugins.golden.register(["resize_nearest_neighbor_v2", "resize_nearest_neighbor_v2_d"])
def resize_nearest_neighbor_golden(context: UniversalTestcaseStructure):
    input_x = context.input_arrays[0]
    srcN, srcC1, srcH, srcW, srcC0 = input_x.shape
    dstH, dstW = context.other_runtime_params["size"]
    align_corners = context.other_compilation_params["align_corners"]
    half_pixel_centers = context.other_compilation_params["half_pixel_centers"]

    scaleH, scaleW = srcH / dstH, srcW / dstW
    if align_corners and dstH > 1:
        scaleH = (srcH - 1) / (dstH - 1)
    if align_corners and dstW > 1:
        scaleW = (srcW - 1) / (dstW - 1)

    dtype = input_x.dtype
    output = np.zeros(shape=(srcN, srcC1, dstH, dstW, srcC0), dtype=dtype)
    for n in range(srcN):
        for c1 in range(srcC1):
            for h in range(dstH):
                for w in range(dstW):
                    srcX, srcY = h * scaleH, w * scaleW
                    if half_pixel_centers:
                        srcX, srcY = (0.5 + h) * scaleH, (0.5 + w) * scaleW

                    if align_corners:
                        output[n, c1, h, w] = input_x[n, c1, round(srcX), round(srcY)]
                    else:
                        output[n, c1, h, w] = input_x[n, c1, math.floor(srcX), math.floor(srcY)]
    return output