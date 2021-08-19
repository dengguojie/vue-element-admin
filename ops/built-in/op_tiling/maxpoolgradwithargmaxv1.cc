#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

    struct MaxPoolGradWithArgmaxV1TilingParams
    {
        int32_t tiling_mode;
        int32_t real_block;
        int32_t block_cycle;
        int32_t block_index;
        int32_t dxh;
        int32_t dxw;
        int32_t dyh;
        int32_t dyw;
        int32_t stride_h;
        int32_t stride_w;
        int32_t pad_bottom;
        int32_t pad_right;
        int32_t offset_h;
        int32_t offset_w;
        int32_t hoverlap;
        int32_t woverlap;
        int32_t col2img_h;
        int32_t col2img_w;
        int32_t col2img_h_every;
        int32_t col2img_h_last;
        int32_t ho_max;
        int32_t wo_max;
        int32_t ho_max_every;
        int32_t ho_max_last;
        int32_t ho_every;
        int32_t ho_last;
        int32_t ho_count;
    };

    struct CompileInfoParams {
        int32_t core_num;
        int32_t ub_size;
        int32_t l1_size;
        int32_t kernel_h;
        int32_t kernel_w;
        int32_t ori_stride_h;
        int32_t ori_stride_w;
        int32_t pad_h;
        int32_t pad_w;
        int32_t dilation_h;
        int32_t dilation_w;
        int32_t ceil_mode;
        int32_t dtype_size;
    };

    void InitTilingParams(MaxPoolGradWithArgmaxV1TilingParams& params)
    {
        params.tiling_mode = 0;
        params.real_block = 0;
        params.block_cycle = 0;
        params.block_index = 0;
        params.dxh = 0;
        params.dxw = 0;
        params.dyh = 0;
        params.dyw = 0;
        params.stride_h = 0;
        params.stride_w = 0;
        params.pad_bottom = 0;
        params.pad_right = 0;
        params.offset_h = 0;
        params.offset_w = 0;
        params.hoverlap = 0;
        params.woverlap = 0;
        params.col2img_h = 0;
        params.col2img_w = 0;
        params.col2img_h_every = 0;
        params.col2img_h_last = 0;
        params.ho_max = 0;
        params.wo_max = 0;
        params.ho_max_every = 0;
        params.ho_max_last = 0;
        params.ho_every = 0;
        params.ho_last = 0;
        params.ho_count = 0;
    }

    bool GetCompileInfos(const std::string& op_type, const nlohmann::json& op_compile_info,
                         CompileInfoParams& compile_params) {
        using namespace nlohmann;
        auto all_vars = op_compile_info["vars"];
        if (all_vars.count("core_num") == 0) {
            VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolGradWithArgmaxV1Tiling", "GetCompileInfos, get core_num error");
            return false;
        }
        compile_params.core_num = all_vars["core_num"].get<std::int32_t>();
        compile_params.ub_size = all_vars["ub_size"].get<std::int32_t>();
        compile_params.l1_size = all_vars["l1_size"].get<std::int32_t>();
        compile_params.kernel_h = all_vars["kernel_h"].get<std::int32_t>();
        compile_params.kernel_w = all_vars["kernel_w"].get<std::int32_t>();
        compile_params.ori_stride_h = all_vars["stride_h"].get<std::int32_t>();
        compile_params.ori_stride_w = all_vars["stride_w"].get<std::int32_t>();
        compile_params.pad_h = all_vars["pad_h"].get<std::int32_t>();
        compile_params.pad_w = all_vars["pad_w"].get<std::int32_t>();
        compile_params.dilation_h = all_vars["dilation_h"].get<std::int32_t>();
        compile_params.dilation_w = all_vars["dilation_w"].get<std::int32_t>();
        compile_params.ceil_mode = all_vars["ceil_mode"].get<std::int32_t>();
        compile_params.dtype_size = all_vars["dtype_size"].get<std::int32_t>();
        return true;
    }

    int32_t NumDiv(int32_t div_a, int32_t div_b)
    {
        int32_t res = 0;
        res = (div_a + div_b - 1) / div_b;
        return res;
    }

    int32_t CalUbLimit(CompileInfoParams& compile_info) {
        int32_t ori_stride_h = compile_info.ori_stride_h;
        int32_t ori_stride_w = compile_info.ori_stride_w;
        int32_t stride_hw = ori_stride_h * ori_stride_w;
        int32_t ub_size = compile_info.ub_size;
        int32_t ub_limit = 1;

        if (stride_hw < 4) {
            ub_limit = ub_size / 6;
        } else {
            ub_limit = ub_size / 4;
        }

        if (stride_hw == 1) {
            ub_limit = ub_size / 8;
        }

        return ub_limit;
    }

    int32_t CalTilingMode(CompileInfoParams& compile_info, int32_t hoverlap, int32_t woverlap, int32_t ub_limit,
                          std::vector<int64_t> grad_shape) {
        int32_t stride_h = compile_info.ori_stride_h;
        int32_t stride_w = compile_info.ori_stride_w;
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t dtype_size = compile_info.dtype_size;
        int32_t batch = grad_shape[0];
        int32_t c1 = grad_shape[1];
        int32_t dyh = grad_shape[2];
        int32_t dyw = grad_shape[3];
        int32_t channel = grad_shape[4];
        int32_t ho_max = 2;
        int32_t wo_max = NumDiv(dyw, 16) * 16;
        int32_t col2img_one_h = kernel_h;
        int32_t col2img_w = (wo_max - 1) * stride_w + kernel_w;
        int32_t col2img_h = (ho_max - 1) * stride_h + kernel_h;
        int32_t core_num = compile_info.core_num;

        if (hoverlap == 0) {
            ho_max = 1;
            col2img_one_h = stride_h;
            col2img_h = ho_max * stride_h;
        }

        if (woverlap == 0) {
            col2img_w = wo_max * stride_w;
        }

        if (kernel_h > 2 * stride_h || kernel_w > 2 * stride_w) {
            return 0;
        }

        if (batch * c1 >= core_num or dyh <= core_num) {
            if (col2img_w * col2img_h * channel * dtype_size > ub_limit) {
                if (col2img_w * col2img_one_h * channel * dtype_size < ub_limit) {
                    return 0;
                }
                return 1;
            }
            return 2;
        }
        
        if (col2img_w * col2img_h * channel * dtype_size > ub_limit) {
            if (col2img_w * col2img_one_h * channel * dtype_size < ub_limit) {
                return 3;
            }
            return 4;
        }
        return 5;
    }

    void CalCol2Img0(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info) {
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t stride_h = tiling_params.stride_h;
        int32_t stride_w = tiling_params.stride_w;
        int32_t dyw = tiling_params.dyw;
        int32_t woverlap = tiling_params.woverlap;
        int32_t col2img_w = 0;

        int32_t col2img_dyw = NumDiv(dyw, 16) * 16;
        int32_t col2img_h = kernel_h;

        if (col2img_h < stride_h) {
            col2img_h = stride_h;
        }

        if (woverlap == 0) {
            col2img_w = col2img_dyw * stride_w;
        } else {
            col2img_w = (col2img_dyw - 1) * stride_w + kernel_w;
        }

        tiling_params.col2img_h = col2img_h;
        tiling_params.col2img_w = col2img_w;
    }

    void CalCol2Img1(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                     int32_t ub_limit) {

        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t dtype_size = compile_info.dtype_size;
        int32_t stride_h = tiling_params.stride_h;
        int32_t stride_w = tiling_params.stride_w;
        int32_t dyw = tiling_params.dyw;
        int32_t woverlap = tiling_params.woverlap;
        int32_t hoverlap = tiling_params.hoverlap;

        int32_t wo_max = NumDiv(dyw, 16) * 16;
        int32_t ho_max = 2;
        int32_t col2img_h = (ho_max - 1) * stride_h + kernel_h;
        int32_t col2img_w = (wo_max - 1) * stride_w + kernel_w;

        if (hoverlap == 0) {
            ho_max = 1;
            col2img_h = ho_max * stride_h;
        }

        if (woverlap == 0) {
            col2img_w = wo_max * stride_w;
        }

        while (col2img_w * col2img_h * dtype_size * 16 > ub_limit) {
            wo_max -= 16;
            col2img_w = (wo_max - 1) * stride_w + kernel_w;
            if (woverlap == 0) {
                col2img_w = wo_max * stride_w;
            }
        }

        tiling_params.wo_max = wo_max;
        tiling_params.ho_max = ho_max;
        tiling_params.col2img_h = col2img_h;
        tiling_params.col2img_w = col2img_w;
    }

    void CalCol2Img2(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                     int32_t ub_limit) {
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t dtype_size = compile_info.dtype_size;
        int32_t dilation_h = compile_info.dilation_h;
        int32_t dilation_w = compile_info.dilation_w;
        int32_t stride_h = tiling_params.stride_h;
        int32_t stride_w = tiling_params.stride_w;
        int32_t dyw = tiling_params.dyw;
        int32_t hoverlap = tiling_params.hoverlap;
        int32_t dxw = tiling_params.dxw;
        int32_t dyh = tiling_params.dyh;
        int32_t flag = 0;

        int32_t wo_max = NumDiv(dyw, 16) * 16;
        int32_t ho_max = (dilation_h * (kernel_h - 1) + stride_h) / stride_h;
        int32_t col2img_w = (wo_max - 1) * stride_w + dilation_w * kernel_w;
        int32_t col2img_h = (ho_max - 1) * stride_h + dilation_h * kernel_h;

        if (col2img_w < dxw) {
            col2img_w = dxw;
        }
        if (hoverlap == 0) {
            ho_max = 1;
            col2img_h = ho_max * stride_h;
        }

        while (col2img_w * col2img_h * dtype_size * 16 < ub_limit && ho_max <= dyh) {
            ho_max += 1;
            flag = 1;
            col2img_h = (ho_max - 1) * stride_h + dilation_h * kernel_h;
            if (hoverlap == 0) {
                col2img_h = ho_max * stride_h;
            }
        }
        if (flag == 1) {
            ho_max -= 1;
            col2img_h = (ho_max - 1) * stride_h + dilation_h * kernel_h;
            if (hoverlap == 0) {
                col2img_h = ho_max * stride_h;
            }
        }

        tiling_params.wo_max = wo_max;
        tiling_params.ho_max = ho_max;
        tiling_params.col2img_h = col2img_h;
        tiling_params.col2img_w = col2img_w;
    }

    void CalCol2Img3(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info) {
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t stride_h = tiling_params.stride_h;
        int32_t stride_w = tiling_params.stride_w;
        int32_t dyw = tiling_params.dyw;
        int32_t woverlap = tiling_params.woverlap;
        int32_t hoverlap = tiling_params.hoverlap;

        int32_t wo_max = NumDiv(dyw, 16) * 16;
        int32_t ho_max = 1;
        int32_t col2img_w = 0;
        int32_t col2img_h = 0;

        if (woverlap == 0) {
            col2img_w = wo_max * stride_w;
        } else {
            col2img_w = (wo_max - 1) * stride_w + kernel_w;
        }

        if (hoverlap == 0) {
            col2img_h = ho_max * stride_h;
        } else {
            col2img_h = (ho_max - 1) * stride_h + kernel_h;
        }

        tiling_params.wo_max = wo_max;
        tiling_params.ho_max = ho_max;
        tiling_params.col2img_h = col2img_h;
        tiling_params.col2img_w = col2img_w;
    }

    void CalCol2Img4(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                     int32_t ub_limit) {

        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t dtype_size = compile_info.dtype_size;
        int32_t stride_h = tiling_params.stride_h;
        int32_t stride_w = tiling_params.stride_w;
        int32_t dyw = tiling_params.dyw;
        int32_t woverlap = tiling_params.woverlap;
        int32_t hoverlap = tiling_params.hoverlap;

        int32_t wo_max = NumDiv(dyw, 16) * 16;
        int32_t ho_max_every = 2;
        int32_t col2img_h_every = (ho_max_every - 1) * stride_h + kernel_h;
        int32_t col2img_w = (wo_max - 1) * stride_w + kernel_w;

        if (hoverlap == 0) {
            ho_max_every = 1;
            col2img_h_every = ho_max_every * stride_h;
        }
        int32_t col2img_h_last = col2img_h_every;
        int32_t ho_max_last = ho_max_every;

        if (woverlap == 0) {
            col2img_w = wo_max * stride_w;
        }

        while (col2img_w * col2img_h_every * dtype_size * 16 > ub_limit) {
            wo_max -= 16;
            col2img_w = (wo_max - 1) * stride_w + kernel_w;
            if (woverlap == 0) {
                col2img_w = wo_max * stride_w;
            }
        }

        tiling_params.wo_max = wo_max;
        tiling_params.ho_max_every = ho_max_every;
        tiling_params.ho_max_last = ho_max_last;
        tiling_params.col2img_h_every = col2img_h_every;
        tiling_params.col2img_h_last = col2img_h_last;
        tiling_params.col2img_w = col2img_w;
    }

    void CalCol2Img5(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                     int32_t ub_limit) {
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t dtype_size = compile_info.dtype_size;
        int32_t dilation_h = compile_info.dilation_h;
        int32_t dilation_w = compile_info.dilation_w;
        int32_t stride_h = tiling_params.stride_h;
        int32_t stride_w = tiling_params.stride_w;
        int32_t dyw = tiling_params.dyw;
        int32_t dxw = tiling_params.dxw;
        int32_t woverlap = tiling_params.woverlap;
        int32_t hoverlap = tiling_params.hoverlap;
        int32_t ho_every = tiling_params.ho_every;
        int32_t ho_last = tiling_params.ho_last;
        int32_t flag = 0;
        int32_t flag_last = 0;

        int32_t wo_max = NumDiv(dyw, 16) * 16;
        int32_t ho_max_every = 2;
        int32_t col2img_w = (wo_max - 1) * stride_w + dilation_w * kernel_w;

        if (woverlap == 0) {
            col2img_w = wo_max * stride_w;
        }
        int32_t col2img_h_every = (ho_max_every - 1) * stride_h + dilation_h * kernel_h;


        if (col2img_w < dxw) {
            col2img_w = dxw;
        }
        if (hoverlap == 0) {
            ho_max_every = 1;
            col2img_h_every = ho_max_every * stride_h;
        }

        int32_t col2img_h_last = col2img_h_every;
        int32_t ho_max_last = ho_max_every;

        while (col2img_w * col2img_h_every * dtype_size * 16 < ub_limit && ho_max_every <= ho_every) {
            ho_max_every += 1;
            flag = 1;
            col2img_h_every = (ho_max_every - 1) * stride_h + dilation_h * kernel_h;
            if (hoverlap == 0) {
                col2img_h_every = ho_max_every * stride_h;
            }
        }
        if (flag == 1) {
            ho_max_every -= 1;
            col2img_h_every = (ho_max_every - 1) * stride_h + dilation_h * kernel_h;
            if (hoverlap == 0) {
                col2img_h_every = ho_max_every * stride_h;
            }
        }

        while (col2img_w * col2img_h_last * dtype_size * 16 < ub_limit && ho_max_last <= ho_last) {
            ho_max_last += 1;
            flag_last = 1;
            col2img_h_last = (ho_max_last - 1) * stride_h + dilation_h * kernel_h;
            if (hoverlap == 0) {
                col2img_h_last = ho_max_last * stride_h;
            }
        }
        if (flag_last == 1) {
            ho_max_last -= 1;
            col2img_h_last = (ho_max_last - 1) * stride_h + dilation_h * kernel_h;
            if (hoverlap == 0) {
                col2img_h_last = ho_max_last * stride_h;
            }
        }

        tiling_params.wo_max = wo_max;
        tiling_params.ho_max_every = ho_max_every;
        tiling_params.ho_max_last = ho_max_last;
        tiling_params.col2img_h_every = col2img_h_every;
        tiling_params.col2img_h_last = col2img_h_last;
        tiling_params.col2img_w = col2img_w;
    }

    void CalCount(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                  std::vector<int64_t> grad_shape, int32_t hoverlap) {
        int32_t core_num = compile_info.core_num;
        int32_t batch = grad_shape[0];
        int32_t c1 = grad_shape[1];
        int32_t dyh = grad_shape[2];
        int32_t ho_count = core_num / (batch * c1);
        int32_t ho_every = 0;
        int32_t ho_last = 0;

        if (ho_count != 0) {
            if (hoverlap == 0) {
                ho_every = dyh / ho_count;
                ho_last = dyh - ho_every * (ho_count - 1);
            } else {
                ho_every = (dyh + ho_count - 1) / ho_count;
                if (ho_every == 1) {
                    ho_count = ho_count / 2;
                    ho_every = (dyh + ho_count - 1) / ho_count;
                }
                ho_last = dyh + ho_count - 1 - ho_every * (ho_count - 1);
            }
        }

        tiling_params.ho_count = ho_count;
        tiling_params.ho_every = ho_every;
        tiling_params.ho_last = ho_last;
    }

    void CalCoreInfo(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                     int32_t block) {
        int32_t core_num = compile_info.core_num;
        int32_t tiling_mode = tiling_params.tiling_mode;
        int32_t ho_count = tiling_params.ho_count;
        int32_t dyh = tiling_params.dyh;
        int32_t cut_block = block;

        int32_t real_block = 0;
        int32_t block_cycle = 0;
        int32_t block_index = 0;

        if (tiling_mode == 3 || tiling_mode == 4 || tiling_mode == 5) {
            cut_block = ho_count * block;
            if (dyh * block < core_num) {
                core_num = dyh * block;
            }
        }

        if (core_num > cut_block) {
            real_block = cut_block;
            block_cycle = 1;
            block_index = 0;
        } else {
            real_block = core_num;
            block_cycle = cut_block / real_block;
            block_index = cut_block % real_block;
        }

        tiling_params.real_block = real_block;
        tiling_params.block_cycle = block_cycle;
        tiling_params.block_index = block_index;
    }

    void CalRunningInfo(MaxPoolGradWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                        std::vector<int64_t> grad_shape, std::vector<int64_t> input_shape)
    {
        int32_t dxh = input_shape[2];
        int32_t dxw = input_shape[3];
        int32_t dyh = grad_shape[2];
        int32_t dyw = grad_shape[3];
        int32_t batch = grad_shape[0];
        int32_t c1 = grad_shape[1];
        int32_t block = batch * c1;
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t stride_h = compile_info.ori_stride_h;
        int32_t stride_w = compile_info.ori_stride_w;
        int32_t pad_bottom = compile_info.pad_h;
        int32_t pad_right = compile_info.pad_w;
        int32_t ceil_mode = compile_info.ceil_mode;
        int32_t offset_h = dxh;
        int32_t offset_w = dxw;
        int32_t ub_limit = CalUbLimit(compile_info);
        int32_t hoverlap = 0;
        int32_t woverlap = 0;

        if (stride_h > dxh) {
            stride_h = dxh;
        }

        if (stride_w > dxw) {
            stride_w = dxw;
        }

        if (ceil_mode == 1) {
            pad_bottom = compile_info.pad_h + stride_h - 1;
            pad_right = compile_info.pad_w + stride_w - 1;
        }

        if (kernel_h > stride_h) {
            hoverlap = kernel_h - stride_h;
        }

        if (kernel_w > stride_w) {
            woverlap = kernel_w - stride_w;
        }
        tiling_params.dxh = dxh;
        tiling_params.dxw = dxw;
        tiling_params.dyh = dyh;
        tiling_params.dyw = dyw;
        tiling_params.stride_h = stride_h,
        tiling_params.stride_w = stride_w;
        tiling_params.pad_bottom = pad_bottom;
        tiling_params.pad_right = pad_right;
        tiling_params.hoverlap = hoverlap;
        tiling_params.woverlap = woverlap;
        tiling_params.offset_h = offset_h;
        tiling_params.offset_w = offset_w;

        int32_t tiling_mode = CalTilingMode(compile_info, hoverlap, woverlap, ub_limit, grad_shape);
        tiling_params.tiling_mode = tiling_mode;
        CalCount(tiling_params, compile_info, grad_shape, hoverlap);

        if (tiling_mode == 0) {
            CalCol2Img0(tiling_params, compile_info);
        }

        if (tiling_mode == 1) {
            CalCol2Img1(tiling_params, compile_info, ub_limit);
        }

        if (tiling_mode == 2) {
            CalCol2Img2(tiling_params, compile_info, ub_limit);
        }

        if (tiling_mode == 3) {
            CalCol2Img3(tiling_params, compile_info);
        }

        if (tiling_mode == 4) {
            CalCol2Img4(tiling_params, compile_info, ub_limit);
        }

        if (tiling_mode == 5) {
            CalCol2Img5(tiling_params, compile_info, ub_limit);
        }

        CalCoreInfo(tiling_params, compile_info, block);
    }

    void SetRunningInfo(const MaxPoolGradWithArgmaxV1TilingParams& tiling_params, OpRunInfo& run_info)
    {
        ByteBufferPut(run_info.tiling_data, tiling_params.tiling_mode);
        ByteBufferPut(run_info.tiling_data, tiling_params.real_block);
        ByteBufferPut(run_info.tiling_data, tiling_params.block_cycle);
        ByteBufferPut(run_info.tiling_data, tiling_params.block_index);
        ByteBufferPut(run_info.tiling_data, tiling_params.dxh);
        ByteBufferPut(run_info.tiling_data, tiling_params.dxw);
        ByteBufferPut(run_info.tiling_data, tiling_params.dyh);
        ByteBufferPut(run_info.tiling_data, tiling_params.dyw);
        ByteBufferPut(run_info.tiling_data, tiling_params.stride_h);
        ByteBufferPut(run_info.tiling_data, tiling_params.stride_w);
        ByteBufferPut(run_info.tiling_data, tiling_params.pad_bottom);
        ByteBufferPut(run_info.tiling_data, tiling_params.pad_right);
        ByteBufferPut(run_info.tiling_data, tiling_params.offset_h);
        ByteBufferPut(run_info.tiling_data, tiling_params.offset_w);
        ByteBufferPut(run_info.tiling_data, tiling_params.hoverlap);
        ByteBufferPut(run_info.tiling_data, tiling_params.woverlap);
        ByteBufferPut(run_info.tiling_data, tiling_params.col2img_h);
        ByteBufferPut(run_info.tiling_data, tiling_params.col2img_w);
        ByteBufferPut(run_info.tiling_data, tiling_params.col2img_h_every);
        ByteBufferPut(run_info.tiling_data, tiling_params.col2img_h_last);
        ByteBufferPut(run_info.tiling_data, tiling_params.ho_max);
        ByteBufferPut(run_info.tiling_data, tiling_params.wo_max);
        ByteBufferPut(run_info.tiling_data, tiling_params.ho_max_every);
        ByteBufferPut(run_info.tiling_data, tiling_params.ho_max_last);
        ByteBufferPut(run_info.tiling_data, tiling_params.ho_every);
        ByteBufferPut(run_info.tiling_data, tiling_params.ho_last);
        ByteBufferPut(run_info.tiling_data, tiling_params.ho_count);
    }

    void PrintTilingParams(const MaxPoolGradWithArgmaxV1TilingParams& tiling_params)
    {
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : tiling_mode=%d.", tiling_params.tiling_mode);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : real_block=%d.", tiling_params.real_block);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : block_cycle=%d.", tiling_params.block_cycle);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : block_index=%d.", tiling_params.block_index);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : dxh=%d.", tiling_params.dxh);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : dxw=%d.", tiling_params.dxw);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : dyh=%d.", tiling_params.dyh);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : dyw=%d.", tiling_params.dyw);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : stride_h=%d.", tiling_params.stride_h);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : stride_w=%d.", tiling_params.stride_w);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : pad_bottom=%d.", tiling_params.pad_bottom);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : pad_right=%d.", tiling_params.pad_right);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : offset_h=%d.", tiling_params.offset_h);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : offset_w=%d.", tiling_params.offset_w);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : hoverlap=%d.", tiling_params.hoverlap);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : woverlap=%d.", tiling_params.woverlap);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : col2img_h=%d.", tiling_params.col2img_h);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : col2img_w=%d.", tiling_params.col2img_w);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : col2img_h_every=%d.", tiling_params.col2img_h_every);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : col2img_h_last=%d.", tiling_params.col2img_h_last);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : ho_max=%d.", tiling_params.ho_max);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : wo_max=%d.", tiling_params.wo_max);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : ho_max_every=%d.", tiling_params.ho_max_every);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : ho_max_last=%d.", tiling_params.ho_max_last);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : ho_every=%d.", tiling_params.ho_every);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : ho_last=%d.", tiling_params.ho_last);
        GELOGI("op [MaxPoolGradWithArgmaxV1Tiling] : ho_count=%d.", tiling_params.ho_count);
    }

    bool MaxPoolGradWithArgmaxV1Tiling(const std::string& op_type, const TeOpParas& op_paras,
                                       const nlohmann::json& op_compile_info, OpRunInfo& run_info)
    {
        using namespace ge;
        CompileInfoParams compile_params;

        bool get_compile_info = GetCompileInfos(op_type, op_compile_info, compile_params);
        if (!get_compile_info) {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "MaxPoolGradWithArgmaxV1Tiling: GetCompileInfos error.");
            return false;
        }

        MaxPoolGradWithArgmaxV1TilingParams tiling_params;
        InitTilingParams(tiling_params);

        const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
        const std::vector<int64_t>& grad_shape = op_paras.inputs[1].tensor[0].shape;
        CalRunningInfo(tiling_params, compile_params, grad_shape, input_shape);
        SetRunningInfo(tiling_params, run_info);
        PrintTilingParams(tiling_params);

        run_info.block_dim = tiling_params.real_block;
        return true;
    }
    // register tiling interface of the MaxPoolGradWithArgmaxV1 op.
    REGISTER_OP_TILING_FUNC_BUFFERED(MaxPoolGradWithArgmaxV1, MaxPoolGradWithArgmaxV1Tiling);
} // namespace optiling.