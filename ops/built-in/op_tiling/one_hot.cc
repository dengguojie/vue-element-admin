#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "op_log.h"

namespace optiling{
    struct OneHotTilingParams
    {
        int32_t is_zero_off_value;
        int32_t not_last_core_numel;
        int32_t mode_of_cal_with_axis;
        int32_t core_num;
        int32_t numel_shape_x;
        int32_t first_dim_x;
        int32_t last_dim_x;
        int32_t numel_shape_off_value_tensor;
        int32_t last_core_numel;
        int32_t not_last_core_index;
        int32_t last_core_index;
    };

    void InitTilingParams(OneHotTilingParams &params)
    {
        OP_LOGD("InitTilingParams is running");
        params.is_zero_off_value = 0;
        params.not_last_core_numel = 0;
        params.mode_of_cal_with_axis = 0;
        params.core_num = 0;
        params.numel_shape_x = 0;
        params.first_dim_x = 0;
        params.last_dim_x = 0;
        params.numel_shape_off_value_tensor = 0;
        params.last_core_numel = 0;
        params.not_last_core_index = 0;
        params.last_core_index = 0;
    }

    bool GetCompileInfo(const std::string &op_type, const nlohmann::json &op_compile_info, 
                        int32_t &total_core_num, int32_t &axis)
    {
        OP_LOGD("GetCompileInfo is running");
        using namespace nlohmann;
        auto all_vars = op_compile_info["vars"];
        if (all_vars.count("core_num") == 0)
        {
            OP_LOGE("op [OneHotTiling] : GetCompileInfo, get total_core_num error");
            return false;
        }
        total_core_num = all_vars["core_num"].get<std::int32_t>();
        axis = all_vars["axis"].get<std::int32_t>();
        return true;
    }

    bool MergeAxis(int32_t axis, int32_t depth, std::vector<int64_t> x_shape, 
                    std::vector<int64_t> &merged_x_shape)
    {
        OP_LOGD("MergeAxis is running");
        int32_t first_dim_size = 1;
        int32_t last_dim_size = 1;
        int32_t i = 0;
        int32_t axis_size = depth;
        int32_t x_shape_size = x_shape.size();
        if (axis == 0)
        {
            while (i < x_shape_size)
            {
                last_dim_size *= x_shape[i];
                ++i;
            }
            merged_x_shape.push_back(axis_size);
            merged_x_shape.push_back(last_dim_size);
            return true;
        }
        else if (axis > 0 && axis < x_shape_size)
        {
            while (i < axis)
            {
                first_dim_size = first_dim_size * x_shape[i];
                ++i;
            }
            i = axis;
            merged_x_shape.push_back(first_dim_size);
            merged_x_shape.push_back(axis_size);
            while (i < x_shape_size)
            {
                last_dim_size *= x_shape[i];
                ++i;
            }
            merged_x_shape.push_back(last_dim_size);
            return true;
        }
        else if (axis == -1 || axis == x_shape_size)
        {
            while (i < x_shape_size)
            {
                first_dim_size *= x_shape[i];
                ++i;
            }
            merged_x_shape.push_back(first_dim_size);
            merged_x_shape.push_back(axis_size);
            return true;
        }
        return false;
    }

    int32_t UpDiv(int32_t first_value, int32_t second_value)
    {
        int32_t result = 1;
        result = (first_value + second_value - 1) / second_value;
        return result;
    }

    int32_t CalTensorNumel(std::vector<int64_t> tensor_shape)
    {
        int32_t numel = 1;
        for (auto ele : tensor_shape)
        {
            numel *= ele;
        }
        return numel;
    }

    int32_t get_core_num(int32_t numel, std::vector<int64_t> x_shape, int32_t depth, int32_t axis)
    {
        int32_t total_core_num = 32;
        auto ele_per_core = (numel - 1) / total_core_num + 1;
        auto core_num = (numel - 1) / ele_per_core + 1;
        auto numel_x = CalTensorNumel(x_shape);
        std::vector<int64_t> merged_x_shape;
        MergeAxis(axis, depth, x_shape, merged_x_shape);
        int32_t block = 16;
        int32_t x_shape_size = x_shape.size();

        if (axis == 0)
        {
            auto per_core_index = UpDiv(depth, core_num);
            while ((per_core_index * numel_x < block) && (core_num > 1))
            {
                core_num -= 1;
                per_core_index = UpDiv(depth, core_num);
            }
            core_num = UpDiv(depth, per_core_index);
        }
        else if (axis == x_shape_size || axis == -1)
        {
            auto per_core_numel = UpDiv(numel_x, core_num);
            while (per_core_numel * depth < block && core_num > 1)
            {
                core_num -= 1;
                per_core_numel = UpDiv(numel_x, core_num);
            }
            core_num = UpDiv(numel_x, per_core_numel);
        }
        else
        {
            auto first_dim_x = merged_x_shape[0];
            auto last_dim_x = merged_x_shape[2];
            auto per_core_numel = UpDiv(first_dim_x, core_num);
            while (per_core_numel * last_dim_x * depth < block && core_num > 1)
            {
                core_num -= 1;
                per_core_numel = UpDiv(first_dim_x, core_num);
            }
            core_num = UpDiv(first_dim_x, per_core_numel);
        }
        return core_num;
    }

    int32_t GetIsZeroOffValue(int32_t off_value)
    {
        int32_t is_zero_off_value = 0;
        if (off_value == 0)
        {
            is_zero_off_value = 1;
        }
        return is_zero_off_value;
    }

    int32_t CalTilingMode(std::vector<int64_t> x_shape, int32_t depth, int32_t axis)
    {
        OP_LOGD("CalTilingMode is running");
        int32_t tiling_mode = 0;
        int32_t x_numel = CalTensorNumel(x_shape);
        int32_t x_shape_size = x_shape.size();

        if (axis == -1 || axis == x_shape_size)
        {
            auto core_num = get_core_num(x_numel, x_shape, depth, axis);
            auto per_core_numel = UpDiv(x_numel, core_num);
            if (per_core_numel <= 19792 && per_core_numel * depth <= 39584)
            {
                tiling_mode = 1;
            }
            else if (per_core_numel <= 19792 && per_core_numel * depth > 39584 && depth >= 2 && depth <= 39584)
            {
                tiling_mode = 2;
            }
            else if (per_core_numel <= 19792 && depth > 39584)
            {
                tiling_mode = 3;
            }
            else if (per_core_numel > 19792 && depth >= 2 && depth <= 39584)
            {
                tiling_mode = 4;
            }
            else
            {
                tiling_mode = 5;
            }
        }
        else if (axis == 0)
        {
            if (x_numel <= 19792 && x_numel * depth <= 39584)
            {
                tiling_mode = 6;
            }
            else if (x_numel <= 19792 && x_numel * depth > 39584)
            {
                tiling_mode = 7;
            }
            else if (x_numel <= 39584 && x_numel > 19792 && x_numel * depth > 39584)
            {
                tiling_mode = 8;
            }
            else
            {
                tiling_mode = 9;
            }
        }
        else if (axis < x_shape_size && axis > 0)
        {
            std::vector<int64_t> merged_x_shape;
            MergeAxis(axis, depth, x_shape, merged_x_shape);
            auto first_dim_x = merged_x_shape[0];
            auto core_num = get_core_num(first_dim_x, x_shape, depth, axis);
            auto per_core_numel = UpDiv(first_dim_x, core_num);
            auto last_dim_x = merged_x_shape[2];
            if (per_core_numel * last_dim_x <= 19792 && per_core_numel * last_dim_x * depth <= 39584)
            {
                tiling_mode = 10;
            }
            else if (per_core_numel * last_dim_x <= 19792 && per_core_numel * 
                last_dim_x * depth > 39584 && last_dim_x >= 1 && last_dim_x <= 39584)
            {
                tiling_mode = 11;
            }
            else if (per_core_numel * last_dim_x > 19792 && per_core_numel * last_dim_x * depth > 39584 && 
                     last_dim_x >= 1 && last_dim_x <= 39584)
            {
                tiling_mode = 12;
            }
            else
            {
                tiling_mode = 13;
            }
        }

        return tiling_mode;
    }

    void CalCoreInfo(OneHotTilingParams &tiling_params, int32_t &core_num, int32_t depth,
                     int32_t axis, std::vector<int64_t> x_shape)
    {
        OP_LOGD("CalCoreInfo is running");
        std::vector<int64_t> merged_x_shape;
        MergeAxis(axis, depth, x_shape, merged_x_shape);

        int32_t per_core_numel = 0;
        int32_t last_core_numel = 0;
        int32_t numel_x = 0;
        int32_t last_core_index = 0;
        int32_t per_core_index = 0;
        int32_t x_shape_size = x_shape.size();

        numel_x = CalTensorNumel(x_shape);

        if (axis == 0)
        {
            core_num = get_core_num(depth, x_shape, depth, axis);
            per_core_index = UpDiv(depth, core_num);
            last_core_index = depth - (core_num - 1) * per_core_index;
        }
        else if (axis == -1 || axis == x_shape_size)
        {
            core_num = get_core_num(numel_x, x_shape, depth, axis);
            per_core_numel = UpDiv(numel_x, core_num);
            last_core_numel = numel_x - (core_num - 1) * per_core_numel;
        }
        else
        {
            auto first_dim_x = merged_x_shape[0];
            core_num = get_core_num(first_dim_x, x_shape, depth, axis);
            per_core_numel = UpDiv(first_dim_x, core_num);
            last_core_numel = first_dim_x - (core_num - 1) * per_core_numel;
        }

        tiling_params.not_last_core_numel = per_core_numel;
        tiling_params.core_num = core_num;
        tiling_params.last_core_numel = last_core_numel;
        tiling_params.last_core_index = last_core_index;
        tiling_params.not_last_core_index = per_core_index;
    }

    void CalRunningInfo(OneHotTilingParams &tiling_params, int32_t core_num, int32_t depth,
                        int32_t axis, int32_t off_value, std::vector<int64_t> x_shape)
    {
        OP_LOGD("CalRunningInfo is running");
        std::vector<int64_t> merged_x_shape;
        MergeAxis(axis, depth, x_shape, merged_x_shape);

        int32_t first_dim_x = 1;
        int32_t last_dim_x = 1;
        int32_t numel_x = 1;
        int32_t numel_merged_x = 1;
        int32_t x_shape_size = x_shape.size();

        numel_x = CalTensorNumel(x_shape);
        numel_merged_x = CalTensorNumel(merged_x_shape);
        tiling_params.numel_shape_x = numel_x;
        if (axis > 0 && axis < x_shape_size)
        {
            first_dim_x = merged_x_shape[0];
            last_dim_x = merged_x_shape[2];
        }
        else if (axis == x_shape_size || axis == -1)
        {
            first_dim_x = merged_x_shape[0];
        }
        else
        {
            last_dim_x = merged_x_shape[1];
        }
        tiling_params.first_dim_x = first_dim_x;
        tiling_params.last_dim_x = last_dim_x;
        tiling_params.numel_shape_off_value_tensor = numel_merged_x;
        tiling_params.mode_of_cal_with_axis = CalTilingMode(x_shape, depth, axis);
        tiling_params.is_zero_off_value = GetIsZeroOffValue(off_value);
        CalCoreInfo(tiling_params, core_num, depth, axis, x_shape);
    }

    void SetRunningInfo(const OneHotTilingParams &tiling_params, OpRunInfo &run_info)
    {
        OP_LOGD("SetRunningInfo is running");
        ByteBufferPut(run_info.tiling_data, tiling_params.is_zero_off_value);
        ByteBufferPut(run_info.tiling_data, tiling_params.not_last_core_numel);
        ByteBufferPut(run_info.tiling_data, tiling_params.mode_of_cal_with_axis);
        ByteBufferPut(run_info.tiling_data, tiling_params.core_num);
        ByteBufferPut(run_info.tiling_data, tiling_params.numel_shape_x);
        ByteBufferPut(run_info.tiling_data, tiling_params.first_dim_x);
        ByteBufferPut(run_info.tiling_data, tiling_params.last_dim_x);
        ByteBufferPut(run_info.tiling_data, tiling_params.numel_shape_off_value_tensor);
        ByteBufferPut(run_info.tiling_data, tiling_params.last_core_numel);
        ByteBufferPut(run_info.tiling_data, tiling_params.not_last_core_index);
        ByteBufferPut(run_info.tiling_data, tiling_params.last_core_index);
    }

    void PrintTilingParams(const OneHotTilingParams &tiling_params)
    {
        OP_LOGD("PrintTilingParams is running");
        OP_LOGD("op [OneHotTiling] : is_zero_off_value=%d.", tiling_params.is_zero_off_value);
        OP_LOGD("op [OneHotTiling] : not_last_core_numel=%d.", tiling_params.not_last_core_numel);
        OP_LOGD("op [OneHotTiling] : mode_of_cal_with_axis=%d.", tiling_params.mode_of_cal_with_axis);
        OP_LOGD("op [OneHotTiling] : core_num=%d.", tiling_params.core_num);
        OP_LOGD("op [OneHotTiling] : numel_shape_x=%d.", tiling_params.numel_shape_x);
        OP_LOGD("op [OneHotTiling] : first_dim_x=%d.", tiling_params.first_dim_x);
        OP_LOGD("op [OneHotTiling] : last_dim_x=%d.", tiling_params.last_dim_x);
        OP_LOGD("op [OneHotTiling] : numel_shape_off_value_tensor=%d.", tiling_params.numel_shape_off_value_tensor);
        OP_LOGD("op [OneHotTiling] : last_core_numel=%d.", tiling_params.last_core_numel);
        OP_LOGD("op [OneHotTiling] : not_last_core_index=%d.", tiling_params.not_last_core_index);
        OP_LOGD("op [OneHotTiling] : last_core_index=%d.", tiling_params.last_core_index);
    }

    bool OneHotTiling(const std::string &op_type, const TeOpParas &op_paras,
                      const nlohmann::json &op_compile_info, OpRunInfo &run_info)
    {
        OP_LOGD("OneHotTiling is running");
        using namespace ge;
        int32_t core_num;
        int32_t axis;
        bool get_compile_info = GetCompileInfo(op_type, op_compile_info, core_num, axis);
        if (!get_compile_info)
        {
            OP_LOGE("op[%s] OneHotTiling: GetCompileInfo error.", op_type.c_str());
            return false;
        }

        OneHotTilingParams tiling_params;
        InitTilingParams(tiling_params);

        std::vector<int64_t> x_shape = op_paras.inputs[0].tensor[0].shape;
        int32_t depth;
        const int32_t* depth_ptr = reinterpret_cast<const int32_t *>(std::get<0>(
            op_paras.const_inputs.at("depth")));
        depth = *depth_ptr;
        const int32_t* off_value_ptr = reinterpret_cast<const int32_t *>(std::get<0>(
            op_paras.const_inputs.at("off_value")));
        auto off_value = *off_value_ptr;
        CalRunningInfo(tiling_params, core_num, depth, axis, off_value, x_shape);
        SetRunningInfo(tiling_params, run_info);
        PrintTilingParams(tiling_params);

        run_info.block_dim = tiling_params.core_num;
        return true;
    }
    REGISTER_OP_TILING_FUNC_BUFFERED(OneHot, OneHotTiling);
} // namespace optiling.

