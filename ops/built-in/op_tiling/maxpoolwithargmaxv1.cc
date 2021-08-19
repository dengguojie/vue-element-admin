#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling {

    struct MaxPoolWithArgmaxV1TilingParams
    {
        int32_t tiling_mode;
        int32_t need_core_num;
        int32_t nc1_per_core;
        int32_t nc1_last_core;
        int32_t batch_size;
        int32_t c1_size;
        int32_t input_h;
        int32_t input_w;
        int32_t input_wh;
        int32_t nc1;
        int32_t output_w;
        int32_t output_h;
        int32_t fmap_h;
        int32_t fmap_h_num;
        int32_t output_wh;
        int32_t mask_tmp;
        int32_t cut_h_size;
        int32_t cut_stride;
        int32_t cut_h_num;
        int32_t flag_cut_h;
        int32_t cut_w_size;
        int32_t cut_w_stride;
        int32_t cut_w_num;

    };

    struct CompileInfoParams {
        int32_t core_num;
        int32_t ub_size;
        int32_t l1_size;
        int32_t kernel_h;
        int32_t kernel_w;
        int32_t stride_h;
        int32_t stride_w;
        int32_t pad_h;
        int32_t pad_w;
        int32_t dilation_h;
        int32_t dilation_w;
        int32_t ceil_mode;
    };

    struct Pad {
        int32_t pad_l;
        int32_t pad_r;
        int32_t pad_t;
        int32_t pad_b;
    };

    void InitTilingParams(MaxPoolWithArgmaxV1TilingParams& params)
    {
        params.tiling_mode = 0;
        params.need_core_num = 0;
        params.nc1_per_core = 0;
        params.nc1_last_core = 0;
        params.batch_size = 0;
        params.c1_size = 0;
        params.input_h = 0;
        params.input_w = 0;
        params.input_wh = 0;
        params.nc1 = 0;
        params.output_w = 0;
        params.output_h = 0;
        params.fmap_h = 0;
        params.fmap_h_num = 0;
        params.output_wh = 0;
        params.mask_tmp = 0;
        params.cut_h_size = 0;
        params.cut_stride = 0;
        params.cut_h_num = 0;
        params.flag_cut_h = 0;
        params.cut_w_size = 0;
        params.cut_w_stride = 0;
        params.cut_w_num = 0;
    }

    bool GetCompileInfo(const std::string& op_type, const nlohmann::json& op_compile_info,
                        CompileInfoParams& compile_params) {
        using namespace nlohmann;
        auto all_vars = op_compile_info["vars"];
        if (all_vars.count("core_num") == 0) {
            VECTOR_INNER_ERR_REPORT_TILIING("MaxPoolWithArgmaxV1Tiling", "GetCompileInfo, get core_num error");
            return false;
        }
        compile_params.core_num = all_vars["core_num"].get<std::int32_t>();
        compile_params.ub_size = all_vars["ub_size"].get<std::int32_t>();
        compile_params.l1_size = all_vars["l1_size"].get<std::int32_t>();
        compile_params.kernel_h = all_vars["kernel_h"].get<std::int32_t>();
        compile_params.kernel_w = all_vars["kernel_w"].get<std::int32_t>();
        compile_params.stride_h = all_vars["stride_h"].get<std::int32_t>();
        compile_params.stride_w = all_vars["stride_w"].get<std::int32_t>();
        compile_params.pad_h = all_vars["pad_h"].get<std::int32_t>();
        compile_params.pad_w = all_vars["pad_w"].get<std::int32_t>();
        compile_params.dilation_h = all_vars["dilation_h"].get<std::int32_t>();
        compile_params.dilation_w = all_vars["dilation_w"].get<std::int32_t>();
        compile_params.ceil_mode = all_vars["ceil_mode"].get<std::int32_t>();
        return true;
    }

    int32_t Ceildiv(int32_t div_a, int32_t div_b)
    {
        int32_t res = 0;
        res = (div_a + div_b - 1) / div_b;
        return res;
    }

    int32_t CalOutPutH(CompileInfoParams& compile_info, std::vector<int64_t> input_shape) {
        int32_t output_h;
        int32_t input_h = input_shape[2];
        int32_t kernel_h = compile_info.kernel_h;
        int32_t stride_h = compile_info.stride_h;
        int32_t pad_h = compile_info.pad_h;
        int32_t dilation_h = compile_info.dilation_h;
        int32_t ceil_mode = compile_info.ceil_mode;
        int32_t tmp_h = input_h + 2 * pad_h - dilation_h * (kernel_h -1) - 1;

        if (ceil_mode == 1) {
            output_h = Ceildiv(tmp_h, stride_h) + 1;
        } else {
            output_h = tmp_h / stride_h + 1;
        }

        if (pad_h > 0) {
            if ((output_h - 1) * stride_h >= (input_h + pad_h)) {
                output_h = output_h - 1;
            }
        }
        return output_h;
    }

    int32_t CalOutPutW(CompileInfoParams& compile_info, std::vector<int64_t> input_shape) {
        int32_t output_w;
        int32_t input_w = input_shape[3];
        int32_t kernel_w = compile_info.kernel_w;
        int32_t stride_w = compile_info.stride_w;
        int32_t pad_w = compile_info.pad_w;
        int32_t dilation_w = compile_info.dilation_w;
        int32_t ceil_mode = compile_info.ceil_mode;
        int32_t tmp_w = input_w + 2 * pad_w - dilation_w * (kernel_w -1) - 1;

        if (ceil_mode == 1) {
            output_w = Ceildiv(tmp_w, stride_w) + 1;
        } else {
            output_w = tmp_w / stride_w + 1;
        }

        if (pad_w > 0) {
            if ((output_w - 1) * stride_w >= (input_w + pad_w)) {
                output_w = output_w - 1;
            }
        }
        return output_w;
    }

    void CalPad(CompileInfoParams& compile_info, Pad& pads) {
        int32_t ceil_mode = compile_info.ceil_mode;
        int32_t pad_h = compile_info.pad_h;
        int32_t pad_w = compile_info.pad_w;
        int32_t stride_h = compile_info.stride_h;
        int32_t stride_w = compile_info.stride_w;

        if (ceil_mode != 1) {
            pads.pad_t = pad_h;
            pads.pad_b = pad_h;
            pads.pad_l = pad_w;
            pads.pad_r = pad_w;
        } else {
            pads.pad_t = pad_h;
            pads.pad_b = pad_h + stride_h - 1;
            pads.pad_l = pad_w;
            pads.pad_r = pad_w + stride_w - 1;
        }
    }

    void CheckNeedCut(MaxPoolWithArgmaxV1TilingParams& tiling_params, int32_t& need_cut,
                      int32_t& need_cut_h, int32_t& need_cut_h_w, CompileInfoParams& compile_info) {
        int32_t input_wh = tiling_params.input_wh;
        int32_t output_w = tiling_params.output_w;
        int32_t output_wh = tiling_params.output_wh;
        int32_t ub_size = compile_info.ub_size;
        int32_t l1_size = compile_info.l1_size;
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t ub_size_used = output_wh * kernel_h * kernel_w * 16 * 4;
        int32_t ub_size_cut = output_w * kernel_h * kernel_w * 16 *4;

        if (ub_size_used > ub_size) {
            need_cut_h = 1;
        }

        if (ub_size_cut > ub_size) {
            need_cut_h_w = 1;
        }

        if (need_cut_h != 1) {
            if (input_wh * 16 * 2 > l1_size) {
                need_cut = 1;
            }
        }
    }

    void CalCutSize(MaxPoolWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info, Pad& pads,
                    int32_t need_cut) {
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t stride_h = compile_info.stride_h;
        int32_t stride_w = compile_info.stride_w;
        int32_t input_h = tiling_params.input_h;
        int32_t input_w = tiling_params.input_w;
        int32_t pad_l = pads.pad_l;
        int32_t pad_r = pads.pad_r;
        int32_t pad_t = pads.pad_t;
        int32_t pad_b = pads.pad_b;
        int32_t ub_size = compile_info.ub_size;
        int32_t l1_size = compile_info.l1_size;
        int32_t cut_h_size = 0;
        int32_t cut_stride = 0;
        int32_t cut_h_num = 0;
        int32_t length = 0;
        int32_t need_cut_tmp = need_cut;

        int32_t img2col_w = kernel_h * kernel_w * 16;
        int32_t img2col_h = (ub_size / 2) / (img2col_w * 2 + 160);
        int32_t pool_w = (input_w + pad_l + pad_r) / stride_w + 1;

        if (kernel_h >= stride_h) {
            cut_h_size = (img2col_h / pool_w - 1) * stride_h + kernel_h - stride_h;

            if (cut_h_size < kernel_h) {
                cut_h_size = kernel_h;
            }
            cut_stride = cut_h_size - (kernel_h - stride_h);
        } else {
            cut_h_size = (img2col_h / pool_w - 1) * stride_h;

            if (cut_h_size < kernel_h) {
                cut_h_size  = kernel_h;
                cut_stride = stride_h;
            } else {
                cut_stride = cut_h_size;
            }
        }

        int32_t pool_h = input_h + pad_t + pad_b;
        if (cut_h_size >= cut_stride) {
            cut_h_num = Ceildiv(pool_h - cut_h_size, cut_stride) + 1;
            length = cut_h_num * stride_h - 1 + kernel_h - 1;
            if (length > pool_h) {
                cut_h_num = cut_h_num - 1;
            }
        } else {
            if (pool_h % cut_stride == 0) {
                cut_h_num = pool_h / cut_stride;
            } else {
                cut_h_num = Ceildiv(pool_h, cut_stride);
            }
        }

        if (cut_h_size * input_w * 16 * 2 > l1_size) {
            need_cut_tmp = 1;
        }

        if (need_cut_tmp == 1) {
            cut_h_size = kernel_h;
            cut_stride = stride_h;
            if (cut_h_size >= cut_stride) {
                cut_h_num = Ceildiv(pool_h - cut_h_size, cut_stride) + 1;
            } else {
                if (pool_h % cut_stride == 0) {
                    cut_h_num = pool_h / cut_stride;
                } else {
                    cut_h_num = Ceildiv(pool_h, cut_stride);
                }
            }
        }

        tiling_params.cut_h_size = cut_h_size;
        tiling_params.cut_stride = cut_stride;
        tiling_params.cut_h_num = cut_h_num;
    }

    void CalCutSizeW(MaxPoolWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info, Pad& pads) {
        int32_t kernel_h = compile_info.kernel_h;
        int32_t kernel_w = compile_info.kernel_w;
        int32_t stride_w = compile_info.stride_w;
        int32_t input_w = tiling_params.input_w;
        int32_t pad_l = pads.pad_l;
        int32_t pad_r = pads.pad_r;
        int32_t ub_size = compile_info.ub_size;
        int32_t cut_w_size = 0;
        int32_t cut_w_stride = 0;
        int32_t cut_w_num = 0;

        int32_t img2col_w = kernel_h * kernel_w * 16;
        int32_t img2col_h = (ub_size / 2) / (img2col_w * 2 + 160);

        if (kernel_w >= stride_w) {
            cut_w_size = (img2col_h - 1) * stride_w + kernel_w - stride_w;
            cut_w_stride = cut_w_size - (kernel_w - stride_w);
        } else {
            cut_w_size = (img2col_h - 1) * stride_w;
            cut_w_stride = cut_w_size;
        }

        int32_t pool_w = input_w + pad_l + pad_r;
        if (cut_w_size >= cut_w_stride) {
            cut_w_num = Ceildiv(pool_w - cut_w_size, cut_w_stride) + 1;
        } else {
            if (pool_w % cut_w_stride == 0) {
                cut_w_num = pool_w / cut_w_stride;
            } else {
                cut_w_num = Ceildiv(pool_w, cut_w_stride);
            }
        }

        tiling_params.cut_w_size = cut_w_size;
        tiling_params.cut_w_stride = cut_w_stride;
        tiling_params.cut_w_num = cut_w_num;
    }

    void CalFlag(MaxPoolWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info, int32_t& nc1_cuth) {
        int32_t flag_cut_h = 0;
        int32_t cut_h_size = tiling_params.cut_h_size;
        int32_t output_w = tiling_params.output_w;
        int32_t nc1 = tiling_params.nc1;
        int32_t kernel_h = compile_info.kernel_h;
        int32_t stride_h = compile_info.stride_h;
        int32_t cut_h_num = tiling_params.cut_h_num;
        int32_t cut_w_size = tiling_params.cut_w_size;
        int32_t cut_w_stride = tiling_params.cut_w_stride;
        int32_t cut_w_num = tiling_params.cut_w_num;
        int32_t tiling_mode = tiling_params.tiling_mode;
        int32_t input_w = tiling_params.input_w;
        int32_t pad_l = compile_info.pad_w;

        int32_t out_size_h = (cut_h_size - kernel_h + stride_h) / stride_h;
        int32_t fmap_cut_h = output_w * out_size_h;
        if (fmap_cut_h % 16 == 0) {
            flag_cut_h = 1;
            nc1_cuth = nc1 * cut_h_num;
        } else {
            nc1_cuth = nc1;
        }

        if (tiling_mode == 2) {
            int32_t cut_w_tail = input_w + pad_l - cut_w_stride * (cut_w_num - 1);
            if (cut_w_tail % 16 == 0 && cut_w_size > 0) {
                flag_cut_h = 1;
                nc1_cuth = nc1 * cut_h_num;
            } else {
                flag_cut_h = 0;
                nc1_cuth = nc1;
            }
        }
        tiling_params.flag_cut_h = flag_cut_h;
    }

    void CalTilingMode(MaxPoolWithArgmaxV1TilingParams& tiling_params, int32_t need_cut, int32_t need_cut_h,
                       int32_t need_cut_h_w) {
        int32_t tiling_mode = 0;
        if ((need_cut_h == 1) || (need_cut == 1)) {
            if (need_cut_h_w == 1) {
                tiling_mode = 2;
            } else {
                tiling_mode = 1;
            }
        } else {
            tiling_mode = 0;
        }

        tiling_params.tiling_mode = tiling_mode;
    }

    void CalCoreInfo(MaxPoolWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                     int32_t nc1_cuth) {
        int32_t need_core_num = 0;
        int32_t nc1_per_core = 0;
        int32_t nc1_last_core = 0;
        int32_t tail = 0;
        int32_t is_same_percore = 0;
        int32_t nc1 = tiling_params.nc1;
        int32_t core_num = compile_info.core_num;
        int32_t tiling_mode = tiling_params.tiling_mode;

        if (tiling_mode != 0) {
            nc1 = nc1_cuth;
        }

        if (nc1 % core_num > 0) {
            tail = 1;
        }
        nc1_per_core = nc1 / core_num + tail;

        if ((nc1 % core_num == 0) || (nc1 % nc1_per_core == 0)) {
            is_same_percore = 0;
        } else {
            is_same_percore = 1;
        }

        need_core_num = nc1 / nc1_per_core;
        if (nc1 / core_num != 0) {
            need_core_num = need_core_num + is_same_percore;
        }
        nc1_last_core = nc1 - (need_core_num - 1) * nc1_per_core;

        tiling_params.need_core_num = need_core_num;
        tiling_params.nc1_per_core = nc1_per_core;
        tiling_params.nc1_last_core = nc1_last_core;
    }

    void CalRunningInfo(MaxPoolWithArgmaxV1TilingParams& tiling_params, CompileInfoParams& compile_info,
                        std::vector<int64_t> input_shape)
    {
        int32_t batch_size = input_shape[0];
        int32_t c1_size = input_shape[1];
        int32_t input_h = input_shape[2];
        int32_t input_w = input_shape[3];
        int32_t input_wh = input_h * input_w;
        int32_t nc1 = batch_size * c1_size;
        int32_t output_w = CalOutPutW(compile_info, input_shape);
        int32_t output_h = CalOutPutH(compile_info, input_shape);
        int32_t output_wh = output_w * output_h;
        int32_t fmap_h = output_w * output_h;
        int32_t fmap_h_num = Ceildiv(fmap_h, 16);
        int32_t mask_tmp = fmap_h_num * 16 - fmap_h;
        int32_t need_cut = 0;
        int32_t need_cut_h = 0;
        int32_t need_cut_h_w = 0;
        int32_t nc1_cuth = 0;

        Pad pads;
        CalPad(compile_info, pads);

        tiling_params.batch_size = batch_size;
        tiling_params.c1_size = c1_size;
        tiling_params.input_h = input_h;
        tiling_params.input_w = input_w;
        tiling_params.input_wh = input_wh;
        tiling_params.nc1 = nc1;
        tiling_params.output_w = output_w;
        tiling_params.output_h = output_h;
        tiling_params.output_wh = output_wh;
        tiling_params.fmap_h = fmap_h;
        tiling_params.fmap_h_num = fmap_h_num;
        tiling_params.mask_tmp = mask_tmp;
        CheckNeedCut(tiling_params, need_cut, need_cut_h, need_cut_h_w, compile_info);
        CalTilingMode(tiling_params, need_cut, need_cut_h, need_cut_h_w);
        CalCutSize(tiling_params, compile_info, pads, need_cut);
        CalCutSizeW(tiling_params, compile_info, pads);
        CalFlag(tiling_params, compile_info, nc1_cuth);
        CalCoreInfo(tiling_params, compile_info, nc1_cuth);
    }

    void SetRunningInfo(const MaxPoolWithArgmaxV1TilingParams& tiling_params, OpRunInfo& run_info)
    {
        ByteBufferPut(run_info.tiling_data, tiling_params.tiling_mode);
        ByteBufferPut(run_info.tiling_data, tiling_params.need_core_num);
        ByteBufferPut(run_info.tiling_data, tiling_params.nc1_per_core);
        ByteBufferPut(run_info.tiling_data, tiling_params.nc1_last_core);
        ByteBufferPut(run_info.tiling_data, tiling_params.batch_size);
        ByteBufferPut(run_info.tiling_data, tiling_params.c1_size);
        ByteBufferPut(run_info.tiling_data, tiling_params.input_h);
        ByteBufferPut(run_info.tiling_data, tiling_params.input_w);
        ByteBufferPut(run_info.tiling_data, tiling_params.input_wh);
        ByteBufferPut(run_info.tiling_data, tiling_params.nc1);
        ByteBufferPut(run_info.tiling_data, tiling_params.output_w);
        ByteBufferPut(run_info.tiling_data, tiling_params.output_h);
        ByteBufferPut(run_info.tiling_data, tiling_params.fmap_h);
        ByteBufferPut(run_info.tiling_data, tiling_params.fmap_h_num);
        ByteBufferPut(run_info.tiling_data, tiling_params.output_wh);
        ByteBufferPut(run_info.tiling_data, tiling_params.mask_tmp);
        ByteBufferPut(run_info.tiling_data, tiling_params.cut_h_size);
        ByteBufferPut(run_info.tiling_data, tiling_params.cut_stride);
        ByteBufferPut(run_info.tiling_data, tiling_params.cut_h_num);
        ByteBufferPut(run_info.tiling_data, tiling_params.flag_cut_h);
        ByteBufferPut(run_info.tiling_data, tiling_params.cut_w_size);
        ByteBufferPut(run_info.tiling_data, tiling_params.cut_w_stride);
        ByteBufferPut(run_info.tiling_data, tiling_params.cut_w_num);
    }

    void PrintTilingParams(const MaxPoolWithArgmaxV1TilingParams& tiling_params)
    {
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : tiling_mode=%d.", tiling_params.tiling_mode);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : need_core_num=%d.", tiling_params.need_core_num);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : nc1_per_core=%d.", tiling_params.nc1_per_core);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : nc1_last_core=%d.", tiling_params.nc1_last_core);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : batch_size=%d.", tiling_params.batch_size);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : c1_size=%d.", tiling_params.c1_size);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : input_h=%d.", tiling_params.input_h);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : input_w=%d.", tiling_params.input_w);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : input_wh=%d.", tiling_params.input_wh);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : nc1=%d.", tiling_params.nc1);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : output_w=%d.", tiling_params.output_w);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : output_h=%d.", tiling_params.output_h);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : fmap_h=%d.", tiling_params.fmap_h);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : fmap_h_num=%d.", tiling_params.fmap_h_num);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : output_wh=%d.", tiling_params.output_wh);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : mask_tmp=%d.", tiling_params.mask_tmp);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : cut_h_size=%d.", tiling_params.cut_h_size);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : cut_stride=%d.", tiling_params.cut_stride);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : cut_h_num=%d.", tiling_params.cut_h_num);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : flag_cut_h=%d.", tiling_params.flag_cut_h);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : cut_w_size=%d.", tiling_params.cut_w_size);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : cut_w_stride=%d.", tiling_params.cut_w_stride);
        GELOGD("op [MaxPoolWithArgmaxV1Tiling] : cut_w_num=%d.", tiling_params.cut_w_num);
    }

    bool MaxPoolWithArgmaxV1Tiling(const std::string& op_type, const TeOpParas& op_paras,
                                   const nlohmann::json& op_compile_info, OpRunInfo& run_info)
    {
        using namespace ge;
        CompileInfoParams compile_params;

        bool get_compile_info = GetCompileInfo(op_type, op_compile_info, compile_params);
        if (!get_compile_info) {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "MaxPoolWithArgmaxV1Tiling: GetCompileInfo error.");
            return false;
        }

        MaxPoolWithArgmaxV1TilingParams tiling_params;
        InitTilingParams(tiling_params);

        const std::vector<int64_t>& input_shape = op_paras.inputs[0].tensor[0].shape;
        CalRunningInfo(tiling_params, compile_params, input_shape);
        SetRunningInfo(tiling_params, run_info);
        PrintTilingParams(tiling_params);

        run_info.block_dim = tiling_params.need_core_num;
        return true;
    }
    // register tiling interface of the MaxPoolWithArgmaxV1 op.
    REGISTER_OP_TILING_FUNC_BUFFERED(MaxPoolWithArgmaxV1, MaxPoolWithArgmaxV1Tiling);
} // namespace optiling.