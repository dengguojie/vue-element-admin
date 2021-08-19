#include <string>
#include <math.h>
#include <nlohmann/json.hpp>
#include "op_tiling.h"
#include "graph/debug/ge_log.h"
#include "op_log.h"
#include "error_log.h"

namespace optiling
{
    struct EmbeddingDenseGradTilingParams
    {
        int32_t numel_indices;
        int32_t embedding_dim;
        int32_t mode_of_cal;
        int32_t core_num;
    };

    void InitTilingParams(EmbeddingDenseGradTilingParams &params)
    {
        params.numel_indices = 0;
        params.embedding_dim = 0;
        params.mode_of_cal = 0;
        params.core_num = 0;
    }

    bool GetCompileInfo(const std::string &op_type, const nlohmann::json &op_compile_info, int32_t &num_weights,
                        int32_t & padding_idx, int32_t & core_num)
    {
        using namespace nlohmann;
        auto all_vars = op_compile_info["vars"];
        num_weights = all_vars["num_weights"].get<std::int32_t>();
        padding_idx = all_vars["padding_idx"].get<std::int32_t>();
        core_num = all_vars["core_num"].get<std::int32_t>();
        return true;
    }

    int32_t CalculateTensorNumel(std::vector<int64_t> tensor_shape)
    {
        int32_t numel = 1;
        for (auto ele : tensor_shape)
        {
            numel *= ele;
        }
        return numel;
    }

    int32_t CalTilingMode(std::vector<int64_t> grad_shape)
    {
        int32_t tiling_mode = 0;
        int32_t embedding_dim = grad_shape[grad_shape.size() - 1];

        if (embedding_dim >= 32 && embedding_dim <= 1024)
        {
            tiling_mode = 1;
        }
        return tiling_mode;
    }

    void CalRunningInfo(const std::string &op_type, const nlohmann::json &op_compile_info,
        EmbeddingDenseGradTilingParams &tiling_params,
        const std::vector<int64_t> &indices_shape, const std::vector<int64_t> &grad_shape)
    {
        int32_t numel_indices = 1;
        numel_indices = CalculateTensorNumel(indices_shape);
        tiling_params.numel_indices = numel_indices;
        tiling_params.embedding_dim = grad_shape[grad_shape.size() - 1];
        tiling_params.mode_of_cal = CalTilingMode(grad_shape);
        int32_t padding_idx;
        int32_t num_weights;
        int32_t core_num;
        (void)GetCompileInfo(op_type, op_compile_info, num_weights, padding_idx, core_num);
        tiling_params.core_num = core_num;
    }

    void SetRunningInfo(const EmbeddingDenseGradTilingParams &tiling_params, OpRunInfo &run_info)
    {
        ByteBufferPut(run_info.tiling_data, tiling_params.numel_indices);
        ByteBufferPut(run_info.tiling_data, tiling_params.embedding_dim);
        ByteBufferPut(run_info.tiling_data, tiling_params.mode_of_cal);
        ByteBufferPut(run_info.tiling_data, tiling_params.core_num);
    }

    void PrintTilingParams(const EmbeddingDenseGradTilingParams &tiling_params)
    {
        GELOGD("op [EmbeddingDenseGradTiling] : numel_indices=%d.", tiling_params.numel_indices);
        GELOGD("op [EmbeddingDenseGradTiling] : embedding_dim=%d.", tiling_params.embedding_dim);
        GELOGD("op [EmbeddingDenseGradTiling] : mode_of_cal=%d.", tiling_params.mode_of_cal);
        GELOGD("op [EmbeddingDenseGradTiling] : core_num=%d.", tiling_params.core_num);
    }

    bool EmbeddingDenseGradTiling(const std::string &op_type, const TeOpParas &op_paras,
                                  const nlohmann::json &op_compile_info, OpRunInfo &run_info)
    {
        using namespace ge;
        int32_t padding_idx;
        int32_t num_weights;
        int32_t core_num;
        bool get_compile_info = GetCompileInfo(op_type, op_compile_info, num_weights, padding_idx, core_num);
        if (!get_compile_info)
        {
            VECTOR_INNER_ERR_REPORT_TILIING(op_type, "EmbeddingDenseGradTiling: GetCompileInfo error.");
            return false;
        }

        EmbeddingDenseGradTilingParams tiling_params;
        InitTilingParams(tiling_params);

        std::vector<int64_t> grad_shape = op_paras.inputs[0].tensor[0].shape;
        std::vector<int64_t> indices_shape = op_paras.inputs[1].tensor[0].shape;
        CalRunningInfo(op_type, op_compile_info, tiling_params, indices_shape, grad_shape);
        SetRunningInfo(tiling_params, run_info);
        PrintTilingParams(tiling_params);

        run_info.block_dim = tiling_params.core_num;
        std::vector<int64_t> workspace;
        run_info.workspaces = workspace;
        return true;
    }
    REGISTER_OP_TILING_FUNC_BUFFERED(EmbeddingDenseGrad, EmbeddingDenseGradTiling);
} // namespace optiling.
