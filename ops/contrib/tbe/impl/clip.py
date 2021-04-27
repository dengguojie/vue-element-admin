from te import tik


def clip(x, y, min, max, kernel_name="clip"):
    shape = x.get("shape")
    input_min, input_max = min, max
    input_n, input_c1, input_h, input_w = shape[0], shape[1], shape[2], shape[3]
    tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))

    # define input&output tensor
    input_data = tik_instance.Tensor("float16", (input_n, input_c1, input_h, input_w, 16),
                                     name="input_data", scope=tik.scope_gm)
    output_data = tik_instance.Tensor("float16", (input_n, input_c1, input_h, input_w, 16),
                                      name="output_data", scope=tik.scope_gm)
    min_ub = tik_instance.Tensor("float16", (16, ), name="min_ub", scope=tik.scope_ubuf)
    max_ub = tik_instance.Tensor("float16", (16, ), name="max_ub", scope=tik.scope_ubuf)

    tik_instance.vector_dup(16, min_ub, input_min, 1, 0, 0, 0)
    tik_instance.vector_dup(16, max_ub, input_max, 1, 0, 0, 0)

    tmp_len = input_n * input_c1 * input_h * input_w * 16
    # judge n*c*h*w ,if big than 120KB, slice
    # when big than 120KB, compute smallest slice piece

    max_num = 56320
    data_ub_tmp1 = tik_instance.Tensor("float16", (max_num, ), name="data_ub_tmp1", scope=tik.scope_ubuf)
    data_ub_tmp2 = tik_instance.Tensor("float16", (max_num, ), name="data_ub_tmp2", scope=tik.scope_ubuf)

    burst_len_tmp = max_num // 16
    if tmp_len >= max_num:
        with tik_instance.for_range(0, tmp_len // max_num) as cnt:
            tik_instance.data_move(data_ub_tmp1, input_data[max_num * cnt], 0, 1, burst_len_tmp, 0, 0, 0)
            with tik_instance.for_range(0, 2) as min_times:
                tik_instance.vmin(128, data_ub_tmp2[220 * 128 * min_times],
                                  data_ub_tmp1[220 * 128 * min_times], max_ub, 220, 1, 1, 0, 8, 8, 0)

            with tik_instance.for_range(0, 2) as max_times:
                tik_instance.vmax(128, data_ub_tmp1[220 * 128 * max_times],
                                  data_ub_tmp2[220 * 128 * max_times], min_ub, 220, 1, 1, 0, 8, 8, 0)

            tik_instance.data_move(output_data[max_num * cnt], data_ub_tmp1, 0, 1, burst_len_tmp, 0, 0, 0)

    burst_len_last = (tmp_len % max_num) // 16
    if burst_len_last != 0:
        tik_instance.data_move(data_ub_tmp1, input_data[tmp_len - tmp_len % max_num], 0, 1, burst_len_last, 0, 0, 0)
        if (burst_len_last // 8) > 255:
            tik_instance.vmin(128, data_ub_tmp2, data_ub_tmp1, max_ub, 255, 1, 1, 0, 8, 8, 0)
            tik_instance.vmax(128, data_ub_tmp1, data_ub_tmp2, min_ub, 255, 1, 1, 0, 8, 8, 0)
        if (burst_len_last // 8) % 255 != 0:
            start_addr = (burst_len_last // (8 * 255)) * 255
            tik_instance.vmin(128, data_ub_tmp2[start_addr * 128], data_ub_tmp1[start_addr * 128],
                              max_ub, (burst_len_last // 8) % 255, 1, 1, 0, 8, 8, 0)
            tik_instance.vmax(128, data_ub_tmp1[start_addr * 128], data_ub_tmp2[start_addr * 128],
                              min_ub, (burst_len_last // 8) % 255, 1, 1, 0, 8, 8, 0)
        if (tmp_len % max_num) % 128 != 0:
            tik_instance.vmin((tmp_len % max_num) % 128, data_ub_tmp2[(burst_len_last // 8) * 128],
                              data_ub_tmp1[(burst_len_last // 8) * 128], max_ub, 1, 1, 1, 0, 8, 8, 0)
            tik_instance.vmax((tmp_len % max_num) % 128, data_ub_tmp1[(burst_len_last // 8) * 128],
                              data_ub_tmp2[(burst_len_last // 8) * 128], min_ub, 1, 1, 1, 0, 8, 8, 0)

        tik_instance.data_move(output_data[max_num * (tmp_len // max_num)], data_ub_tmp1, 0, 1, burst_len_last, 0, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_data], outputs=[output_data])
    return tik_instance
