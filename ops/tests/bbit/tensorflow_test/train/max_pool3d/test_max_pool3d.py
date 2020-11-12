import tensorflow as tf


class TFBBITTest:
    def __init__(self):
        pass

    def _init_config(self, execute_type):
        if execute_type == 'ai_core':
            session_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            custom_op.parameter_map["enable_data_pre_proc"].b = True
            custom_op.parameter_map["mix_compile_mode"].b = True
            custom_op.parameter_map["use_off_line"].b = True
            custom_op.parameter_map["min_group_size"].b = 1
        elif execute_type == 'cpu':
            session_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
        else:
            print("Unknown execute type %s" % execute_type)
            session_config = None

        return session_config

    #
    # def test_method(self, *args, **kwargs):
    #     return []
    #
    #
    # def set_graph_params(self):
    #     return self
    #
    # def feed_graph(self):
    #     return self
    #
    # def run_test(self):
    #     out_tensors = self.test_method(kwargs)
    #     with tf.Session(config=self.config('cpu')) as sess:
    #         result_cpu = sess.run(out_tensors, feed_dict=feed_dict)
    #         sess.close()
    #
    #     with tf.Session(config=self.config('ai_core')) as sess:
    #         result_ai_core = sess.run(out_tensors, feed_dict=feed_dict)
    #         sess.close()
    #
    #     return result_cpu, result_ai_core


from tensorflow.python.ops import gen_nn_ops


class MaxPool3DTest(TFBBITTest):

    def __init__(self):
        super(TFBBITTest).__init__()

    def define_graph(self, x_shape, dtype, ksize, strides, padding):
        self.max_pool_3d_input_x = tf.placeholder(x_shape, dtype)
        out_tensor = gen_nn_ops.max_pool3d(self.max_pool_3d_input_x, ksize, strides, padding, 'NDHWC', 'max_pool3d')
        return out_tensor

    def feed_and_run(self, input_x):
        self.feed_dict



if __name__ == "__main__":
    tests = MaxPool3DTest()
    tests.run_test()

