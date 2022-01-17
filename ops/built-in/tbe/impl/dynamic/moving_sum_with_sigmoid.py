from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

MAX_INT32 = 2 ** 31 - 1
ALIGN = 64

@register_operator("moving_sum_with_sigmoid")
class MovingSumWithSigmoid(object):
    def __init__(self, alpha, energy, frame_size, y, window_size, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.kernel_name = kernel_name

        self.dtype = alpha.get("dtype").lower()
        if self.dtype == "float32":
            self.block = 8
        else:
            raise RuntimeError("Unexpected dtype.")

        self.alpha_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], name="alpha", scope=tik.scope_gm)
        self.energy_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], name="energy", scope=tik.scope_gm)
        self.frame_size_gm = self.tik_instance.Tensor("int32", [1], name="frame_size", scope=tik.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.dtype, [MAX_INT32], name="y", scope=tik.scope_gm)

        self.tiling_gm = self.tik_instance.Tensor('int32', [1], name="tiling_gm", scope=tik.scope_gm)

        self.window_size = window_size
        self.window_size_align = (self.window_size + ALIGN  - 1) // ALIGN * ALIGN

        self.used_aicore_num = tik.Dprofile().get_aicore_num()

        self.frame_size = None
        self.task_num = None
        self.batch_num_per_aicore = None
        self.batch_tail = None

    def moving_sum_with_sigmoid_compute(self):
        frame_size_ub = self.tik_instance.Tensor("int32", [self.block ], name="frame_size_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(frame_size_ub, self.frame_size_gm, 0, 1, 1, 0, 0)
        self.frame_size = self.tik_instance.Scalar("int32", init_value=frame_size_ub[0])
        self.task_num = self.tik_instance.Scalar("int32", init_value=(self.frame_size + self.block  - 1) // self.block )

        self.batch_num_per_aicore = self.tik_instance.Scalar("int32",
                                                             init_value=self.task_num // self.used_aicore_num)
                                                             
        self.batch_tail = self.tik_instance.Scalar("int32", init_value=self.task_num % self.used_aicore_num)

        with self.tik_instance.for_range(0, self.used_aicore_num, block_num=self.used_aicore_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_aicore) as j:
                self.moving_sum_with_sigmoid_compute_core(i + j * self.used_aicore_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.moving_sum_with_sigmoid_compute_core(self.batch_num_per_aicore * self.used_aicore_num + i)

        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        tbe_context.get_context().add_compile_info(
            "vars", {
                "core_num": self.used_aicore_num,
        })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.alpha_gm, self.energy_gm, self.frame_size_gm],
                                   outputs=[self.y_gm], flowtable=[self.tiling_gm],config=opt_config)

        return self.tik_instance

    def moving_sum_with_sigmoid_compute_core(self, task_idx):
        alpha_ub = self.tik_instance.Tensor(self.dtype, [self.block  + self.window_size_align], name="alpha_ub",
                                            scope=tik.scope_ubuf)
        energy_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="energy_ub",
                                             scope=tik.scope_ubuf)
        y_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="y_ub", scope=tik.scope_ubuf)

        self.tik_instance.data_move(alpha_ub, self.alpha_gm[task_idx * self.block], 0, 1,
                                    self.window_size_align // self.block  + 1, 0, 0)
        self.tik_instance.data_move(energy_ub, self.energy_gm[task_idx * self.block], 0, 1, 1, 0, 0)

        ones_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="ones_ub", scope=tik.scope_ubuf)
        zero_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="zero_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="tmp_ub", scope=tik.scope_ubuf)
        sigmoid_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="sigmoid_ub", scope=tik.scope_ubuf)
        sum_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="sum_ub", scope=tik.scope_ubuf)
        work_tensor_ub = self.tik_instance.Tensor(self.dtype, [self.window_size_align], name="work_tensor_ub",
                                                  scope=tik.scope_ubuf)
        self.tik_instance.vector_dup(self.block , ones_ub, 1, 1, 1, 1)
        self.tik_instance.vector_dup(self.block , zero_ub, 0, 1, 1, 1)

        # func '1 / (1 + np.exp(-x))'
        exp_ub = self.tik_instance.Tensor(self.dtype, [self.block], name="exp_ub", scope=tik.scope_ubuf)
        self.tik_instance.vec_sub(self.block , tmp_ub, zero_ub, energy_ub, 1, 1, 1, 1)
        self.tik_instance.vec_exp(self.block , exp_ub, tmp_ub, 1, 1, 1)
        self.tik_instance.vec_add(self.block , tmp_ub, exp_ub, ones_ub, 1, 1, 1, 1)
        self.tik_instance.vec_rec_high_preci(self.block , sigmoid_ub, tmp_ub, work_tensor_ub, 1, 1, 1)

        tmp_val = self.tik_instance.Scalar(self.dtype)
        sum_val = self.tik_instance.Scalar(self.dtype)
        loop = self.tik_instance.Scalar("int32", init_value= self.frame_size - task_idx * self.block)

        with self.tik_instance.if_scope(loop > self.window_size):
            with self.tik_instance.if_scope(self.window_size > ALIGN):
                with self.tik_instance.if_scope(self.window_size % ALIGN > 0):
                    self.tik_instance.vec_reduce_add(ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                    self.window_size_align // ALIGN - 1, 8)
                    tmp_val.set_as(sum_ub[0])
                    self.tik_instance.vec_reduce_add((self.window_size % ALIGN), sum_ub,
                                                    alpha_ub[self.window_size_align - ALIGN],
                                                    work_tensor_ub, 1, 0)
                    sum_val.set_as(sum_ub[0])
                    sum_ub[0].set_as(sum_val + tmp_val)
                with self.tik_instance.else_scope():
                    self.tik_instance.vec_reduce_add(ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                     self.window_size // ALIGN, 8)
            with self.tik_instance.else_scope():
                self.tik_instance.vec_reduce_add(self.window_size, sum_ub, alpha_ub, work_tensor_ub, 1, 1)

        with self.tik_instance.else_scope():
            with self.tik_instance.if_scope(loop > ALIGN):
                with self.tik_instance.if_scope(loop % ALIGN > 0):
                    self.tik_instance.vec_reduce_add(ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                    (loop + ALIGN - 1) // ALIGN - 1, 8)
                    tmp_val.set_as(sum_ub[0])
                    self.tik_instance.vec_reduce_add(loop % ALIGN , sum_ub,
                                                    alpha_ub[loop - loop % ALIGN], work_tensor_ub, 1, 0)
                    sum_val.set_as(sum_ub[0])
                    sum_ub[0].set_as(sum_val + tmp_val)
                with self.tik_instance.else_scope():
                    self.tik_instance.vec_reduce_add(ALIGN, sum_ub, alpha_ub, work_tensor_ub,
                                                     loop // ALIGN, 8)

            with self.tik_instance.else_scope():
                self.tik_instance.vec_reduce_add(loop, sum_ub, alpha_ub, work_tensor_ub, 1, 1)

        sum_val.set_as(sum_ub[0])
        with self.tik_instance.for_range(1, self.block) as idx:
 
            tmp_val.set_as(alpha_ub[idx - 1])
            sum_val.set_as(sum_val - tmp_val)
            loop.set_as(loop - 1)
            with self.tik_instance.if_scope(loop >= self.window_size):
                tmp_val.set_as(alpha_ub[idx + self.window_size - 1])
                sum_val.set_as(sum_val + tmp_val)

            sum_ub[idx].set_as(sum_val)

        self.tik_instance.vec_mul(self.block, y_ub, sum_ub, sigmoid_ub, 1, 0, 0, 0)
        self.tik_instance.data_move(self.y_gm[task_idx * self.block], y_ub, 0, 1, 1, 0, 0)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def moving_sum_with_sigmoid(alpha, energy, frame_size, y, window_size,
                            kernel_name="moving_sum_with_sigmoid"):
    """
    To do: Implement the operator by referring to the
           TBE Operator Development Guide.
    """

    op_obj = MovingSumWithSigmoid(alpha, energy, frame_size, y, window_size, kernel_name)

    return op_obj.moving_sum_with_sigmoid_compute()
