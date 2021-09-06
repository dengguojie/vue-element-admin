# -*- coding:utf-8 -*-
from te import tik
from impl.util.platform_adapter import para_check


class Utils:
    def __init__(self, tik_inst):
        self.tik_inst = tik_inst
        self.fp16_min_value = -32768.0
        self.support_vector_max_len = 16320

    def align_offline(self, value, to_align):
        return value if value % to_align == 0 else (value // to_align + 1) * to_align

    def align_online(self, value, to_align):
        result = self.tik_inst.Scalar("int32")
        with self.tik_inst.if_scope(value % to_align == 0):
            result.set_as(value)
        with self.tik_inst.else_scope():
            result.set_as(value // to_align + 1)

    def _max_last_branch(self, cmp_scalar, position, tmp_scalar, tmp_index_int16, repeat_time,
                         input_vector, max_index_int16, last_data_len, vector_len):
        if last_data_len != 0:
            position.set_as(repeat_time * 128 + tmp_index_int16)
            with self.tik_inst.if_scope(position < vector_len):
                cmp_scalar.set_as(input_vector[position])
                with self.tik_inst.if_scope(tik.all(cmp_scalar == tmp_scalar,
                                                    position < max_index_int16)):
                    max_index_int16.set_as(position)

    def _max_last(self, cmp_scalar, vmax_dst, position, tmp_scalar, valid_index, res_vec, tmp_index_int16,
                  repeat_time, input_vector, max_index_int16, last_data_len, vector_len):
        with self.tik_inst.for_range(0, 128) as i:
            cmp_scalar.set_as(vmax_dst[i])
            position.set_as(i)
            with self.tik_inst.if_scope(tmp_scalar == cmp_scalar):
                res_vec[valid_index].set_as(position)
                valid_index.set_as(valid_index + 1)
        with self.tik_inst.for_range(0, valid_index) as i:
            tmp_index_int16.set_as(res_vec[i])
            with self.tik_inst.for_range(0, repeat_time) as j:
                position.set_as(j * 128 + tmp_index_int16)
                cmp_scalar.set_as(input_vector[position])
                with self.tik_inst.if_scope(tik.all(tmp_scalar == cmp_scalar, position < max_index_int16)):
                    max_index_int16.set_as(position)
                with self.tik_inst.else_scope():
                    pass
            self._max_last_branch(cmp_scalar, position, tmp_scalar, tmp_index_int16, repeat_time,
                                  input_vector, max_index_int16, last_data_len, vector_len)

    def _post_process(self, max_value_out, max_value, index_type, max_index_int32, max_index_int16, max_index_out):
        max_value_out.set_as(max_value)
        if index_type == "int32":
            max_index_int32.set_as(max_index_int16)
            max_index_out.set_as(max_index_int32)
        elif index_type == "uint16":
            max_index_out.set_as(max_index_int16)
        else:
            raise TypeError("index_type only support int32 or uint16")

    def _last_max(self, repeat_time, vmax_dst_tmp, vmax_dst, input_vector, last_data_len):
        with self.tik_inst.for_range(0, repeat_time) as i:
            self.tik_inst.vmax(128, vmax_dst_tmp, vmax_dst, input_vector[i * 128], 1, 1, 1, 1, 0, 0, 0)
            self.tik_inst.vector_dup(128, vmax_dst, 0, 1, 1, 8)
            self.tik_inst.vadd(128, vmax_dst, vmax_dst, vmax_dst_tmp, 1, 1, 1, 1, 0, 0, 0)

        if last_data_len != 0:
            last_data = self.tik_inst.Tensor("float16", (last_data_len,), scope=tik.scope_ubuf,
                                             name="last_data")
            self.tik_inst.vector_dup(last_data_len, last_data, 1, 1, 1, 0)
            self.tik_inst.vmul(last_data_len, last_data, last_data, input_vector[128 * repeat_time],
                               1, 1, 1, 1, 0, 0, 0)
            self.tik_inst.vmax(last_data_len, vmax_dst, vmax_dst, last_data, 1, 1, 1, 1, 0, 0, 0)

    def group_max_fp16(self, input_vector, valid_data_len, max_value_out, max_index_out, index_type="int32"):
        vector_len = input_vector.shape[0]
        repeat_time = valid_data_len // 128

        with self.tik_inst.new_stmt_scope():
            max_value = self.tik_inst.Scalar("float16", init_value=self.fp16_min_value)
            max_index_int16 = self.tik_inst.Scalar("uint16", init_value=self.support_vector_max_len)
            max_index_int32 = self.tik_inst.Scalar("int32", init_value=-1)

            if repeat_time == 0:
                with self.tik_inst.new_stmt_scope():
                    vcmax_dst = self.tik_inst.Tensor("float16", (16,), scope=tik.scope_ubuf, name="vcmax_dst")
                    self.tik_inst.vcmax(valid_data_len, vcmax_dst, input_vector, 1, 1, 1, 0)
                    max_value.set_as(vcmax_dst[0])
                    max_index_int16.set_as(vcmax_dst[1])
            elif repeat_time > 255:
                raise IndexError("support max input len is 255 * 128")
            else:
                with self.tik_inst.new_stmt_scope():
                    last_data_len = valid_data_len - repeat_time * 128

                    tmp_index_int16 = self.tik_inst.Scalar("uint16")
                    tmp_scalar = self.tik_inst.Scalar("int16")
                    cmp_scalar = self.tik_inst.Scalar("int16")
                    valid_index = self.tik_inst.Scalar("int16", init_value=0)
                    position = self.tik_inst.Scalar("uint16", init_value=0)

                    vmax_dst = self.tik_inst.Tensor("float16", (128,), scope=tik.scope_ubuf, name="vmax_dst")
                    vmax_dst_tmp = self.tik_inst.Tensor("float16", (128,), scope=tik.scope_ubuf,
                                                        name="vmax_dst_tmp")
                    vcmax_dst = self.tik_inst.Tensor("float16", (16,), scope=tik.scope_ubuf, name="vcmax_dst")
                    res_vec = self.tik_inst.Tensor("uint16", (128,), scope=tik.scope_ubuf, name="res_vec")
                    self.tik_inst.vector_dup(128, res_vec, 0, 1, 1, 0)

                    self.tik_inst.vector_dup(128, vmax_dst, self.fp16_min_value, 1, 1, 8)
                    self._last_max(repeat_time, vmax_dst_tmp, vmax_dst, input_vector, last_data_len)

                    self.tik_inst.vcmax(128, vcmax_dst, vmax_dst, 1, 1, 1, 0)
                    max_value.set_as(vcmax_dst[0])
                    tmp_scalar.set_as(vcmax_dst[0])
                    tmp_index_int16.set_as(vcmax_dst[1])

                    self._max_last(cmp_scalar, vmax_dst, position, tmp_scalar, valid_index, res_vec, tmp_index_int16,
                                   repeat_time, input_vector, max_index_int16, last_data_len, vector_len)
            self._post_process(max_value_out, max_value, index_type, max_index_int32, max_index_int16, max_index_out)


class CTCGreedyDecoder:
    def __init__(self, input_shape, sequence_length, merge_repeated=True, default_value=0,
                 kernel_name="ctc_greedy_decode"):
        """
        CTCGreedyDecoder
        :param input_shape: [batchsize, timestep, classnum]
        :param sequence_length: [vaild sequence length]
        :param max_sequence_len: int value
        :param merge_repeated: True / False
        :param default_value: the value of invalid position
        :param kernel_name_value: op kernel name
        :return:NA
        """
        self._param_check(input_shape, sequence_length)

        self.tik_inst = tik.Tik(tik.Dprofile("v100", "mini"))
        self.utils = Utils(self.tik_inst)

        self.max_times = -1
        self.batchsize = input_shape[0]
        self.sequence_length = self.tik_inst.Tensor("int32", (self.batchsize + 1,), scope=tik.scope_ubuf,
                                                    name="sequence_length")
        tmp_s = self.tik_inst.Scalar("int32")

        for i in range(0, self.batchsize):
            if sequence_length[i] <= 0:
                raise RuntimeError("sequence_length is invalid")
            else:
                tmp_s.set_as(sequence_length[i])
                self.sequence_length[i].set_as(tmp_s)
                if sequence_length[i] > self.max_times:
                    self.max_times = sequence_length[i]

        self.input_shape = input_shape
        self.output_len = self.max_times + 1

        self.class_num = input_shape[2]
        self.merge_repeated = merge_repeated
        self.default_value = default_value
        self.min_value = -32768

        self.kernel_name = kernel_name

        self.input_tensor = self.tik_inst.Tensor("float16", input_shape, scope=tik.scope_gm, name="input_tensor")
        self.output_tensor = self.tik_inst.Tensor("float16", (self.batchsize, self.output_len,),
                                                  scope=tik.scope_gm, name="output_tensor")

    def _param_check(self, input_shape, sequence_length):
        if len(input_shape) != 3:
            raise RuntimeError("input_shape is [batchsize, timestep, class_num]")
        if not isinstance(sequence_length, (list, tuple)):
            raise TypeError("sequence_length must be list or tuple")
        if len(sequence_length) != input_shape[0]:
            raise RuntimeError("sequence_length do not equal to batchsize")
        if input_shape[0] * input_shape[1] * input_shape[2] * 2 / 1024 > 200:
            raise RuntimeError("support input data size < 200K")
        if input_shape[2] > 2048:
            raise RuntimeError("support max class_num is 2048")

    def compute(self, ):
        self._mod_1_compute()
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_tensor], outputs=[self.output_tensor])

    def _init_mask(self, tmp_vector_mask, repeat_time, last_data_len, tmp_vector_len):
        tmp_scalar = self.tik_inst.Scalar("float16", init_value=self.min_value)
        if repeat_time == 0:
            self.tik_inst.vector_dup(tmp_vector_len, tmp_vector_mask, 0, 1, 1, 0)
        else:
            self.tik_inst.vector_dup(128, tmp_vector_mask, 0, repeat_time, 1, 8)
            if last_data_len != 0:
                self.tik_inst.vector_dup(last_data_len, tmp_vector_mask[repeat_time * 128], 0, 1, 1, 0)

        if tmp_vector_len != self.class_num:
            with self.tik_inst.for_range(0, tmp_vector_len - self.class_num) as i:
                tmp_vector_mask[self.class_num + i].set_as(tmp_scalar)

    def _reset_vector(self, res_repeat_time, tmp_result_len, tmp_result, res_last_data_len):
        if res_repeat_time == 0:
            self.tik_inst.vector_dup(tmp_result_len, tmp_result, self.default_value, 1, 1, 0)
        else:
            self.tik_inst.vector_dup(64, tmp_result, self.default_value, res_repeat_time, 1, 8)
            if res_last_data_len != 0:
                self.tik_inst.vector_dup(res_last_data_len, tmp_result[res_repeat_time * 64],
                                         self.default_value, 1, 1, 0)

    def _exe(self, loop_scalar, tmp_vector, tmp_vector_len, repeat_time, tmp_vector_mask, last_data_len, max_value,
             max_index, blank_index, pre_max_index, tmp_result, valid_index, batch_num):
        with self.tik_inst.for_range(0, loop_scalar) as i:
            self.tik_inst.data_move(tmp_vector, self.input_tensor[batch_num, i, 0], 0, 1, tmp_vector_len // 16, 0, 0)
            if repeat_time == 0:
                self.tik_inst.vadd(tmp_vector_len, tmp_vector, tmp_vector, tmp_vector_mask, 1, 1, 1, 1, 0, 0, 0)
            else:
                self.tik_inst.vadd(128, tmp_vector, tmp_vector, tmp_vector_mask, repeat_time, 1, 1, 1, 8, 8, 8)
                if last_data_len != 0:
                    self.tik_inst.vadd(last_data_len, tmp_vector[repeat_time * 128], tmp_vector[repeat_time * 128],
                                       tmp_vector_mask[repeat_time * 128], 1, 1, 1, 1, 0, 0, 0)
            self.utils.group_max_fp16(tmp_vector, tmp_vector_len, max_value, max_index)

            if self.merge_repeated:
                with self.tik_inst.if_scope(tik.all(max_index != blank_index, max_index != pre_max_index)):
                    tmp_result[valid_index].set_as(max_index)
                    valid_index.set_as(valid_index + 1)
            else:
                with self.tik_inst.if_scope(max_index != blank_index):
                    tmp_result[valid_index].set_as(max_index)
                    valid_index.set_as(valid_index + 1)
            pre_max_index.set_as(max_index)

    def _process(self, valid_index, pre_max_index, res_repeat_time, tmp_result, tmp_result_len, res_last_data_len,
                 tmp_vector, tmp_vector_len, repeat_time, tmp_vector_mask, last_data_len, blank_index,
                 vconv_repeat_time, tmp_result_fp16, vconv_last_data_len):
        loop_scalar = self.tik_inst.Scalar("int32")
        max_index = self.tik_inst.Scalar("int32")
        max_value = self.tik_inst.Scalar("float16")
        with self.tik_inst.for_range(0, self.batchsize) as batch_num:
            valid_index.set_as(0)
            pre_max_index.set_as(-1)
            self._reset_vector(res_repeat_time, tmp_result_len, tmp_result, res_last_data_len)
            loop_scalar.set_as(self.sequence_length[batch_num])
            self._exe(loop_scalar, tmp_vector, tmp_vector_len, repeat_time, tmp_vector_mask, last_data_len, max_value,
                      max_index, blank_index, pre_max_index, tmp_result, valid_index, batch_num)

            tmp_result[self.max_times].set_as(valid_index)
            if vconv_repeat_time == 0:
                self.tik_inst.vconv(tmp_result_len, "", tmp_result_fp16, tmp_result, 1, 1, 1, 4, 8, 1.0)
            else:
                self.tik_inst.vconv(64, "", tmp_result_fp16, tmp_result, vconv_repeat_time, 1, 1, 4, 8, 1.0)
                if vconv_last_data_len != 0:
                    self.tik_inst.vconv(vconv_last_data_len, "", tmp_result_fp16[vconv_repeat_time * 64],
                                        tmp_result[vconv_repeat_time * 64], 1, 1, 1, 4, 8, 1.0)
            self.tik_inst.data_move(self.output_tensor[batch_num, 0], tmp_result_fp16, 0, 1, tmp_result_len // 16, 0, 0)

    def _mod_1_compute(self):
        tmp_vector_len = self.utils.align_offline(self.class_num, 16)
        tmp_vector = self.tik_inst.Tensor("float16", (tmp_vector_len,), scope=tik.scope_ubuf, name="tmp_vector")
        tmp_vector_mask = self.tik_inst.Tensor("float16", (tmp_vector_len,), scope=tik.scope_ubuf,
                                               name="tmp_vector_mask")

        repeat_time = tmp_vector_len // 128
        last_data_len = tmp_vector_len - repeat_time * 128

        self._init_mask(tmp_vector_mask, repeat_time, last_data_len, tmp_vector_len)

        tmp_result_len = self.utils.align_offline(self.output_len, 16)
        tmp_result = self.tik_inst.Tensor("int32", (tmp_result_len,), scope=tik.scope_ubuf, name="tmp_result")
        tmp_result_fp16 = self.tik_inst.Tensor("float16", (tmp_result_len,), scope=tik.scope_ubuf,
                                               name="tmp_result_fp16")

        res_repeat_time = tmp_result_len // 64
        res_last_data_len = tmp_result_len - res_repeat_time * 64

        vconv_repeat_time = tmp_result_len // 64
        vconv_last_data_len = tmp_result_len - vconv_repeat_time * 64

        blank_index = self.tik_inst.Scalar("int16")
        blank_index.set_as(self.class_num - 1)
        pre_max_index = self.tik_inst.Scalar("int16")
        valid_index = self.tik_inst.Scalar("int32")

        self._process(valid_index, pre_max_index, res_repeat_time, tmp_result, tmp_result_len, res_last_data_len,
                      tmp_vector, tmp_vector_len, repeat_time, tmp_vector_mask, last_data_len, blank_index,
                      vconv_repeat_time, tmp_result_fp16, vconv_last_data_len)

# pylint: disable=unused-argument
def ctc_greedy_decoder(inputs, output,
                       sequence_length,
                       merge_repeated=True,
                       default_value=0,
                       kernel_name="ctc_greedy_decoder",
                       test=False):
    """
    Parameters
    ----------
    inputs : dict
        shape and dtype of input
    output : dict
        shape and dtype of output, should be same shape and type as input
    param input_shape: [batchsize, timestep, class_num]
    param sequence_length: [vaild sequence length]
    param merge_repeated: True / False
    param default_value: the value of invalid position
    kernel_name : str, kernel name, default value is "ctc_greedy_decoder"

    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    shape = inputs.get("shape")
    dtype = inputs.get("dtype")

    para_check.check_shape_rule(shape)
    para_check.check_tensor_shape_size(shape)

    check_list = ["float16"]
    if dtype not in check_list:
        raise RuntimeError("only support %s while dtype is %s" % (str(check_list), dtype))

    obj_tik = CTCGreedyDecoder(shape, sequence_length, merge_repeated, default_value,
                               kernel_name=kernel_name)
    obj_tik.compute()
