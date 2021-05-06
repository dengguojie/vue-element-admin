# -*- coding: utf-8 -*-
from te import tik
from .image_detection import ImageDetection, Param

CLASSES_NUM = 80
MIN_SCORE = 3
IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
ANCHOR_13 = 13
ANCHOR_26 = 26
ANCHOR_52 = 52
SHAPE_13_WIDTH = 169
SHAPE_26_WIDTH = 676
SHAPE_52_WIDTH = 2704
SHAPE_PAD_HEIGHT = 288


class Yolov3PostProcessor(object):
    def __init__(self, input0_shape, input1_shape, input2_shape, anchor13_shape, anchor26_shape, anchor52_shape,
                 grid13_shape, grid26_shape, grid52_shape, mask512_shape, obj_classes_shape, output_shape,
                 kernel_name="yolov3postprocessor"):
        """
        Introduction
            TF Yolov3 detectoutput Fusion op
        ------------
        Parameters
        ----------
            @input0: Reshape_1,13x13x3x85, shape: ND [169 x 288]
            @input1: Reshape_7,26x26x3x85, shape: ND [676 x 288]
            @input2: Reshape_13,52x52x3x85, shape: ND  [2704 x 288]
            @anchor13: reshape_1 13x13 anchors, op: const, dtype: float32, shape: ND [2x512]
            @anchor26: reshape_7 26x26 anchors, op: const, dtype: float32, shape: ND [8x512]
            @anchor52: reshape_13 52x52 anchors, op: const, dtype: float32, shape: ND [32x512]
            @grid13: reshape_1 grids, op: const, dtype: float32, shape: ND [2x512]
            @grid26: reshape_7 grids, op: const, dtype: float32, shape: ND [2x512]
            @grid52: reshape_13 grids, op: const, dtype: float32, shape: ND [2x512]
            @mask512: mask for score, op: const, dtype: float32, shape: ND [512]
            @objclasses: coco 80 classes, op: const, dtype: float32, shape: ND [80x128]
            @output: dectect results for proposal format [ymin, xmin, ymax, xmax, score, self.label, 0, 0],
                     dtype: float16, shape: ND[100x8]
            @kernel_name

        """
        self.tik_inst = tik.Tik(tik.Dprofile("v100", "mini"))
        self.kernel_name = kernel_name

        self.input_shape = [IMAGE_WIDTH, IMAGE_HEIGHT]
        self.image_shape = [IMAGE_WIDTH, IMAGE_HEIGHT]
        self.grid_size_13 = [ANCHOR_13, ANCHOR_13]
        self.grid_size_26 = [ANCHOR_26, ANCHOR_26]
        self.grid_size_52 = [ANCHOR_52, ANCHOR_52]

        if input0_shape != (1, SHAPE_13_WIDTH, SHAPE_PAD_HEIGHT):
            raise RuntimeError("input1 shape is not match [1, 169, 288]")

        if input1_shape != (1, SHAPE_26_WIDTH, SHAPE_PAD_HEIGHT):
            raise RuntimeError("input2 shape is not match [1, 676, 288]")

        if input2_shape != (1, SHAPE_52_WIDTH, SHAPE_PAD_HEIGHT):
            raise RuntimeError("input3 shape is not match [1, 2704, 288]")

        self.reshape_1_gm = self.tik_inst.Tensor("float16", input0_shape, name="reshape_1_gm", scope=tik.scope_gm)
        self.reshape_7_gm = self.tik_inst.Tensor("float16", input1_shape, name="reshape_7_gm", scope=tik.scope_gm)
        self.reshape_13_gm = self.tik_inst.Tensor("float16", input2_shape, name="reshape_13_gm", scope=tik.scope_gm)
        self.anchor13_gm = self.tik_inst.Tensor("float16", anchor13_shape, name="anchor13_gm", scope=tik.scope_gm)
        self.anchor26_gm = self.tik_inst.Tensor("float16", anchor26_shape, name="anchor26_gm", scope=tik.scope_gm)
        self.anchor52_gm = self.tik_inst.Tensor("float16", anchor52_shape, name="anchor52_gm", scope=tik.scope_gm)
        self.grid13_gm = self.tik_inst.Tensor("float16", grid13_shape, name="grid13_gm", scope=tik.scope_gm)
        self.grid26_gm = self.tik_inst.Tensor("float16", grid26_shape, name="grid26_gm", scope=tik.scope_gm)
        self.grid52_gm = self.tik_inst.Tensor("float16", grid52_shape, name="grid52_gm", scope=tik.scope_gm)
        self.mask_gm = self.tik_inst.Tensor("float16", mask512_shape, name="mask_gm", scope=tik.scope_gm)
        self.objclass_gm = self.tik_inst.Tensor("float16", obj_classes_shape, name="objclass_gm", scope=tik.scope_gm)
        self.output_gm = self.tik_inst.Tensor("float16", output_shape, name="output_gm", scope=tik.scope_gm)

        self.index = self.tik_inst.Scalar("int32")
        self.score_temp = self.tik_inst.Scalar("int32")
        self.score_threshold = self.tik_inst.Scalar("int32")
        self.score_threshold.set_as(10)
        self.label = self.tik_inst.Scalar("float16")

        self.proposal_l1buf = self.tik_inst.Tensor("float16", (80, 640, 8), name="proposal_l1buf",
                                                   scope=tik.scope_cbuf)
        self.label_l1buf = self.tik_inst.Tensor("float16", (80, 640), name="label_l1buf", scope=tik.scope_cbuf)
        self.score_l1buf = self.tik_inst.Tensor("float16", (80, 512), name="score_l1buf", scope=tik.scope_cbuf)

        self.detection_handle = ImageDetection(self.tik_inst)

    def _get_feats(self, featscores, box_x, box_y, box_w, box_h, box_conf, grid, gridsize, anchors, inputshape,
                   length):
        """
        Introduction
        ------------
            get features [x, y, w, h, score, conf] from shapes
        Parameters
        ----------
            @featscores, score IN: shape[2X512]
            @box_x, IN/OUT: shape[512]
            @box_y, IN/OUT: shape[512]
            @box_w, IN/OUT: shape[512]
            @box_h, IN/OUT: shape[512]
            @box_conf, IN/OUT: shape[512X80]
            grid, IN: shape[512]
            gridsize, generated grid, IN:(416, 416)
            anchors, generateed anchors, IN:shape[2,512]
            inputshape, input shape, IN:(416, 416)
            length: total ceils
        Returns
        -------
        :param self.tik_inst:
        """
        coeffw = self.tik_inst.Scalar(dtype="float16")
        coeffw.set_as(1. / gridsize[0])
        coeffh = self.tik_inst.Scalar(dtype="float16")
        coeffh.set_as(1. / gridsize[1])
        input_shape_w = self.tik_inst.Scalar(dtype="float16")
        input_shape_w.set_as(1.0 / inputshape[0])
        input_shape_h = self.tik_inst.Scalar(dtype="float16")
        input_shape_h.set_as(1.0 / inputshape[1])

        grid_x = self.tik_inst.Tensor("float16", (512,), name="grid_x", scope=tik.scope_ubuf)
        grid_y = self.tik_inst.Tensor("float16", (512,), name="grid_y", scope=tik.scope_ubuf)

        repeat_num = length // 128

        with self.tik_inst.new_stmt_scope():
            self._sigmoid_2d(featscores, 128)  # box_class_probs
        with self.tik_inst.new_stmt_scope():
            self._sigmoid_1d(box_x, 128)  # box_xy
        with self.tik_inst.new_stmt_scope():
            self._sigmoid_1d(box_y, 128)
        with self.tik_inst.new_stmt_scope():
            self._sigmoid_1d(box_conf, 128)  # box_confidence

        self.tik_inst.data_move(grid_x[0], grid[0, 0], 0, 1, 512 // 16, 0, 0)
        self.tik_inst.data_move(grid_y[0], grid[1, 0], 0, 1, 512 // 16, 0, 0)

        # box_xy equals to (sigmoidout + grid) / tf.cast(grid_size[::-1], tf.float32)
        self.tik_inst.vadd(128, box_x, box_x, grid_x, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(128, box_y, box_y, grid_y, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmuls(128, box_x, box_x, coeffw, repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_y, box_y, coeffh, repeat_num, 1, 1, 8, 8)

        # box_wh equals to tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        with self.tik_inst.new_stmt_scope():
            self._expfunc_1d(box_w, 128)
            self._expfunc_1d(box_h, 128)

        self.tik_inst.data_move(grid_x[0], anchors[0, 0], 0, 1, 512 // 16, 0, 0)
        self.tik_inst.data_move(grid_y[0], anchors[1, 0], 0, 1, 512 // 16, 0, 0)

        self.tik_inst.vmul(128, box_w, box_w, grid_x, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmul(128, box_h, box_h, grid_y, repeat_num, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vmuls(128, box_w, box_w, input_shape_w, repeat_num, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_h, box_h, input_shape_h, repeat_num, 1, 1, 8, 8)

    def _sigmoid_1d(self, feats, mask_length):
        """
        Introduction
        ------------
            sigmoid function, sigmoid(x) equals to 1/(1 + e(-x))
        Parameters
        ----------
            feats: input/output feats tensor, 16 align
            length: feats length, 16 align
            masklength: 16 align
        Returns
        -------
        """
        length = feats.shape[0]
        iter_num_1 = length // 16
        iter_num_2 = length // mask_length

        exp_ub = self.tik_inst.Tensor("float16", (length,), name="exp_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_inst.Tensor("float16", (length,), name="tmp_ub", scope=tik.scope_ubuf)

        self.tik_inst.data_move(tmp_ub, feats, 0, 1, iter_num_1, 0, 0)
        self.tik_inst.vexp(mask_length, exp_ub, tmp_ub, iter_num_2, 1, 1, mask_length // 16,
                           mask_length // 16)  # e^(x)
        self.tik_inst.vadds(mask_length, tmp_ub, exp_ub, 1.0, iter_num_2, 1, 1, mask_length // 16,
                            mask_length // 16)  # 1 + e^(x)
        self.tik_inst.vrec(mask_length, tmp_ub, tmp_ub, iter_num_2, 1, 1, mask_length // 16,
                           mask_length // 16)  # 1/(1 + e^(x))
        self.tik_inst.vmul(mask_length, exp_ub, exp_ub, tmp_ub, iter_num_2, 1, 1, 1, mask_length // 16,
                           mask_length // 16, mask_length // 16)
        self.tik_inst.data_move(feats, exp_ub, 0, 1, iter_num_1, 0, 0)

    def _sigmoid_2d(self, feats, mask_length):
        """
        Introduction
        ------------
            sigmoid function, sigmoid(x) equals to 1/(1 + e(-x))
        Parameters
        ----------
            feats: input/output feats tensor, 16 align
            length: feats length, 16 align
            masklength: 16 align
        Returns
        -------
        """
        n_0 = feats.shape[0]
        n_1 = feats.shape[1]
        iter_num1 = n_1 // 16
        iter_num2 = n_1 // mask_length

        exp_ub = self.tik_inst.Tensor("float16", (n_1,), name="exp_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_inst.Tensor("float16", (n_1,), name="tmp_ub", scope=tik.scope_ubuf)

        with self.tik_inst.for_range(0, n_0, thread_num=2) as i:
            self.tik_inst.data_move(tmp_ub, feats[i, 0], 0, 1, iter_num1, 0, 0)
            self.tik_inst.vexp(mask_length, exp_ub, tmp_ub, iter_num2, 1, 1, mask_length // 16,
                               mask_length // 16)  # e^(x)
            self.tik_inst.vadds(mask_length, tmp_ub, exp_ub, 1.0, iter_num2, 1, 1, mask_length // 16,
                                mask_length // 16)  # 1 + e^(x)
            self.tik_inst.vrec(mask_length, tmp_ub, tmp_ub, iter_num2, 1, 1, mask_length // 16,
                               mask_length // 16)  # 1/(1 + e^(x))
            self.tik_inst.vmul(mask_length, exp_ub, exp_ub, tmp_ub, iter_num2, 1, 1, 1, mask_length // 16,
                               mask_length // 16, mask_length // 16)  # e^(x) / (1 + e^(x))
            self.tik_inst.data_move(feats[i, 0], exp_ub, 0, 1, iter_num1, 0, 0)

    def _expfunc_1d(self, feats, mask_length):
        """
        Introduction
        ------------
            exp function, exp(x)
        Parameters
        ----------
            feats: input/output feats tensor, 16 align
            length: feats length, 16 align
            masklength: 16 align
        Returns
        -------
        """
        length = feats.shape[0]
        iter_num1 = length // 16
        iter_num2 = length // mask_length

        exp_ub = self.tik_inst.Tensor("float16", (length,), name="exp_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_inst.Tensor("float16", (length,), name="tmp_ub", scope=tik.scope_ubuf)

        self.tik_inst.data_move(tmp_ub, feats, 0, 1, iter_num1, 0, 0)
        self.tik_inst.vexp(mask_length, exp_ub, tmp_ub, iter_num2, 1, 1, mask_length // 16, mask_length // 16)
        self.tik_inst.data_move(feats, exp_ub, 0, 1, iter_num1, 0, 0)

    def _calc_new_shape(self, image_w_ub, image_h_ub, shape_w_ub, shape_h_ub, temp_ub):
        self.tik_inst.vrec(16, temp_ub, image_w_ub, 1, 1, 1, 1, 1)
        self.tik_inst.vmul(16, temp_ub, shape_w_ub, temp_ub, 1, 1, 1, 1, 1, 1, 1)
        self.tik_inst.vmul(16, image_w_ub, image_w_ub, temp_ub, 1, 1, 1, 1, 1, 1, 1)
        self.tik_inst.vrec(16, temp_ub, image_h_ub, 1, 1, 1, 1, 1)
        self.tik_inst.vmul(16, temp_ub, shape_h_ub, temp_ub, 1, 1, 1, 1, 1, 1, 1)
        self.tik_inst.vmul(16, image_h_ub, image_h_ub, temp_ub, 1, 1, 1, 1, 1, 1, 1)

    def _calc_offset_shape(self, offset_x_ub, offset_y_ub, recx_ub, recy_ub, image_w_ub,
                           image_h_ub, shape_w_ub, shape_h_ub):
        self.tik_inst.vsub(16, offset_x_ub, shape_w_ub, image_w_ub, 1, 1, 1, 1, 1, 1, 1)
        self.tik_inst.vsub(16, offset_y_ub, shape_h_ub, image_h_ub, 1, 1, 1, 1, 1, 1, 1)
        self.tik_inst.vmuls(16, offset_x_ub, offset_x_ub, 0.5, 1, 1, 1, 1, 1)
        self.tik_inst.vmuls(16, offset_y_ub, offset_y_ub, 0.5, 1, 1, 1, 1, 1)
        self.tik_inst.vrec(16, recx_ub, shape_w_ub, 1, 1, 1, 1, 1)
        self.tik_inst.vrec(16, recy_ub, shape_h_ub, 1, 1, 1, 1, 1)
        self.x_0.set_as(recx_ub[0])
        self.y_0.set_as(recy_ub[0])
        self.tik_inst.vmuls(16, offset_x_ub, offset_x_ub, self.x_0, 1, 1, 1, 1, 1)
        self.tik_inst.vmuls(16, offset_y_ub, offset_y_ub, self.y_0, 1, 1, 1, 1, 1)
        self.offsetx.set_as(offset_x_ub[0])
        self.offsety.set_as(offset_x_ub[0])

    def _calc_scale_shape(self, scale_x_ub, scale_y_ub, image_w_ub, image_h_ub, shape_w_ub, shape_h_ub):
        self.tik_inst.vrec(16, image_w_ub, image_w_ub, 1, 1, 1, 1, 1)
        self.tik_inst.vrec(16, image_h_ub, image_h_ub, 1, 1, 1, 1, 1)
        self.x_0.set_as(image_w_ub[0])
        self.y_0.set_as(image_h_ub[0])
        self.tik_inst.vmuls(16, scale_x_ub, shape_w_ub, self.x_0, 1, 1, 1, 1, 1)
        self.tik_inst.vmuls(16, scale_y_ub, shape_h_ub, self.y_0, 1, 1, 1, 1, 1)
        self.scalex1.set_as(scale_x_ub[0])
        self.scaley1.set_as(scale_y_ub[0])

    def _correct_boxes(self, box_x, box_y, box_w, box_h, box_min_x, box_min_y, box_max_x, box_max_y,
                       input_shape, image_shape, length):
        """
        Introduction
        ------------
            generated boxes, [ymin, xmin ,ymax, xmax]
        Parameters
        ----------
            self.tik_inst, IN: tik handle
            box_x, IN: shape[512]
            box_y, IN: shape[512]
            box_w, IN: shape[512]
            box_h, IN: shape[512]
            box_min_x, OUT: shape[512]
            box_min_y, OUT: shape[512]
            box_max_x, OUT: shape[512]
            box_max_y, OUT: shape[512]
            input_shape, IN:(416, 416)
            image_shape, IN:(416, 416)
            length, IN: total ceil length
        -------
        """
        self.x_0 = self.tik_inst.Scalar("float16")
        self.y_0 = self.tik_inst.Scalar("float16")
        self.scalex1 = self.tik_inst.Scalar("float16")
        self.scaley1 = self.tik_inst.Scalar("float16")
        self.offsetx = self.tik_inst.Scalar("float16")
        self.offsety = self.tik_inst.Scalar("float16")
        repeatnum = length // 128

        inputshape_w_ub = self.tik_inst.Tensor("float16", (16,), name="inputshape_w_ub", scope=tik.scope_ubuf)
        inputshape_h_ub = self.tik_inst.Tensor("float16", (16,), name="inputshape_h_ub", scope=tik.scope_ubuf)
        imageshape_w_ub = self.tik_inst.Tensor("float16", (16,), name="imageshape_w_ub", scope=tik.scope_ubuf)
        imageshape_h_ub = self.tik_inst.Tensor("float16", (16,), name="imageshape_h_ub", scope=tik.scope_ubuf)

        temp_ub = self.tik_inst.Tensor("float16", (16,), name="temp_ub", scope=tik.scope_ubuf)
        recx_ub = self.tik_inst.Tensor("float16", (16,), name="recx_ub", scope=tik.scope_ubuf)
        recy_ub = self.tik_inst.Tensor("float16", (16,), name="recy_ub", scope=tik.scope_ubuf)
        offset_x_ub = self.tik_inst.Tensor("float16", (16,), name="offset_x_ub", scope=tik.scope_ubuf)
        offset_y_ub = self.tik_inst.Tensor("float16", (16,), name="offset_y_ub", scope=tik.scope_ubuf)
        scale_x_ub = self.tik_inst.Tensor("float16", (16,), name="scale_x_ub", scope=tik.scope_ubuf)
        scale_y_ub = self.tik_inst.Tensor("float16", (16,), name="scale_y_ub", scope=tik.scope_ubuf)
        zero_ub = self.tik_inst.Tensor("float16", (16,), name="zero_ub", scope=tik.scope_ubuf)

        # input_shape
        self.tik_inst.vector_dup(16, inputshape_w_ub, input_shape[0], 1, 1, 1)
        self.tik_inst.vector_dup(16, inputshape_h_ub, input_shape[1], 1, 1, 1)

        # image_shape
        self.tik_inst.vector_dup(16, imageshape_w_ub, image_shape[0], 1, 1, 1)
        self.tik_inst.vector_dup(16, imageshape_h_ub, image_shape[1], 1, 1, 1)

        # new_shape equals to tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        self._calc_new_shape(imageshape_w_ub, imageshape_h_ub, inputshape_w_ub, imageshape_h_ub, temp_ub)

        # offset equals to (input_shape - new_shape) / 2. / input_shape
        self._calc_offset_shape(offset_x_ub, offset_y_ub, recx_ub, recy_ub, imageshape_w_ub, imageshape_h_ub,
                                inputshape_w_ub, inputshape_h_ub)

        # scale equals to input_shape / new_shape
        self._calc_scale_shape(scale_x_ub, scale_y_ub, imageshape_w_ub, imageshape_h_ub,
                               inputshape_w_ub, inputshape_h_ub)

        # box_yx equals to (box_yx - offset) * scale
        self.tik_inst.vmuls(16, zero_ub, zero_ub, 0., 1, 1, 1, 1, 1)
        self.tik_inst.vsub(16, offset_x_ub, zero_ub, offset_x_ub, 1, 1, 1, 1, 1, 1, 1)
        self.tik_inst.vsub(16, offset_y_ub, zero_ub, offset_y_ub, 1, 1, 1, 1, 1, 1, 1)
        self.x_0.set_as(offset_x_ub[0])
        self.y_0.set_as(offset_y_ub[0])

        self.tik_inst.vadds(128, box_x, box_x, self.x_0, repeatnum, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_x, box_x, self.scalex1, repeatnum, 1, 1, 8, 8)
        self.tik_inst.vadds(128, box_y, box_y, self.y_0, repeatnum, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_y, box_y, self.scaley1, repeatnum, 1, 1, 8, 8)

        # box_hw equals to box_hw x scale
        self.tik_inst.vmuls(128, box_w, box_w, self.scalex1, repeatnum, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_h, box_h, self.scaley1, repeatnum, 1, 1, 8, 8)

        # (box_hw / 2.)
        self.tik_inst.vmuls(128, box_w, box_w, 0.5, repeatnum, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_h, box_h, 0.5, repeatnum, 1, 1, 8, 8)

        # box_mins equals to box_yx - (box_hw / 2.)
        self.tik_inst.vsub(128, box_min_x, box_x, box_w, repeatnum, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vsub(128, box_min_y, box_y, box_h, repeatnum, 1, 1, 1, 8, 8, 8)

        # box_maxes equals to box_yx + (box_hw / 2.)
        self.tik_inst.vadd(128, box_max_x, box_x, box_w, repeatnum, 1, 1, 1, 8, 8, 8)
        self.tik_inst.vadd(128, box_max_y, box_y, box_h, repeatnum, 1, 1, 1, 8, 8, 8)

        # boxes equals to  boxes * tf.concat([image_shape, image_shape], axis=-1)
        self.tik_inst.vmuls(128, box_min_x, box_min_x, image_shape[0], repeatnum, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_min_y, box_min_y, image_shape[1], repeatnum, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_max_x, box_max_x, image_shape[0], repeatnum, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, box_max_y, box_max_y, image_shape[1], repeatnum, 1, 1, 8, 8)

    def _boxes_and_scores(self, boxes_x_ub, boxes_y_ub, boxes_w_ub, boxes_h_ub, conf_ub, score_ub, grid,
                          box_minx_ub, box_miny_ub, box_minw_ub, box_minh_ub, anchors, gridsize, classes_num,
                          input_shape, image_shape):
        """
        Introduction
            calc boxes and scores acord by shape[1x13x13x3x85], [1x26x26x3x85] and [1x52x52x3x85],
            85 describe as [x, y, w, h, label, 80 classe]
        ------------
        ----------
            boxes_x_ub: IN: [512]
            boxes_y_ub: IN: [512]
            boxes_w_ub: IN: [512]
            boxes_h_ub: IN: [512]
            conf_ub:    IN: [512]
            score_ub:   IN: [2X512]
            grid:       IN: [2X512]
            box_minx_ub: OUT: [512]
            box_miny_ub: OUT: [512]
            box_minw_ub: OUT: [512]
            box_minh_ub: OUT: [512]
            anchors:    IN: [2X512]
            gridsize:   IN: [2X512]
            classes_num:  IN: [80,]
            input_shape:  IN: (416, 416)
            image_shape:  IN: (416, 416)
        Returns
        """
        # box_xy, box_wh, box_confidence, box_class_probs equals to _get_feats(feats, anchors, classes_num, input_shape)
        with self.tik_inst.new_stmt_scope():
            self._get_feats(score_ub, boxes_x_ub, boxes_y_ub, boxes_w_ub, boxes_h_ub, conf_ub, grid, gridsize,
                            anchors, input_shape, 512)

        # boxes equals to self._correct_boxes(box_xy, box_wh, input_shape, image_shape)
        with self.tik_inst.new_stmt_scope():
            self._correct_boxes(boxes_x_ub, boxes_y_ub, boxes_w_ub, boxes_h_ub, box_minx_ub, box_miny_ub,
                                box_minw_ub, box_minh_ub,
                                input_shape, image_shape, 512)

        # box_scores equals to box_confidence * box_class_probs
        with self.tik_inst.for_range(0, classes_num, thread_num=2) as i:
            self.tik_inst.vmul(128, score_ub[i, 0], score_ub[i, 0], conf_ub, 512 // 128, 1, 1, 1, 8, 8, 8)

    def _tensor_transpose(self, data_convert_ub, data_ub):
        with self.tik_inst.for_range(0, 96 // 16, thread_num=2) as loop_i:
            src_list = [data_ub[0, loop_i * 16], data_ub[1, loop_i * 16], data_ub[2, loop_i * 16],
                        data_ub[3, loop_i * 16], data_ub[4, loop_i * 16], data_ub[5, loop_i * 16],
                        data_ub[6, loop_i * 16], data_ub[7, loop_i * 16], data_ub[8, loop_i * 16],
                        data_ub[9, loop_i * 16], data_ub[10, loop_i * 16], data_ub[11, loop_i * 16],
                        data_ub[12, loop_i * 16], data_ub[13, loop_i * 16], data_ub[14, loop_i * 16],
                        data_ub[15, loop_i * 16]]
            dst_list = [data_convert_ub[loop_i * 16 + 0, 0], data_convert_ub[loop_i * 16 + 1, 0],
                        data_convert_ub[loop_i * 16 + 2, 0], data_convert_ub[loop_i * 16 + 3, 0],
                        data_convert_ub[loop_i * 16 + 4, 0], data_convert_ub[loop_i * 16 + 5, 0],
                        data_convert_ub[loop_i * 16 + 6, 0], data_convert_ub[loop_i * 16 + 7, 0],
                        data_convert_ub[loop_i * 16 + 8, 0], data_convert_ub[loop_i * 16 + 9, 0],
                        data_convert_ub[loop_i * 16 + 10, 0], data_convert_ub[loop_i * 16 + 11, 0],
                        data_convert_ub[loop_i * 16 + 12, 0], data_convert_ub[loop_i * 16 + 13, 0],
                        data_convert_ub[loop_i * 16 + 14, 0], data_convert_ub[loop_i * 16 + 15, 0]]
            self.tik_inst.vnchwconv(True, True, dst_list, src_list, 512 // 16, 1, 96)  # 1, 96

    def _yolov3_anchor_13_score_processor(self):
        # 13x13 anchors
        data_convert_ub = self.tik_inst.Tensor("float16", (96, 512), name="data_convert_ub", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            data_ub = self.tik_inst.Tensor("float16", (512, 96), name="data_ub", scope=tik.scope_ubuf)
            # clear to zero
            with self.tik_inst.for_range(0, 2) as k:
                self.tik_inst.vmuls(128, data_ub[256 * k, 0], data_ub[256 * k, 0], 0, 256 * 96 // 128, 1, 1, 8, 8)
            self.tik_inst.data_move(data_ub[0, 0], self.reshape_1_gm[0, 0, 0], 0, 1,
                                    SHAPE_13_WIDTH * SHAPE_PAD_HEIGHT // 16, 0, 0)
            # represent transpose 512*96 to 96*512
            self._tensor_transpose(data_convert_ub, data_ub)

        score_ub = self.tik_inst.Tensor("float16", (80, 512), name="score_ub", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            # extract [x, y, w, h, conf] from featboxes
            self.tik_inst.data_move(self.boxes_x_ub[0], data_convert_ub[0, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_y_ub[0], data_convert_ub[1, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_w_ub[0], data_convert_ub[2, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_h_ub[0], data_convert_ub[3, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.conf_ub[0], data_convert_ub[4, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(score_ub[0, 0], data_convert_ub[5, 0], 0, 1, 80 * 512 // 16, 0, 0)
            self.tik_inst.data_move(self.anchor13_ub, self.anchor13_gm, 0, 1, 1024 // 16, 0, 0)
            self.tik_inst.data_move(self.grid13_ub, self.grid13_gm, 0, 1, 1024 // 16, 0, 0)

            # main process
            self._boxes_and_scores(self.boxes_x_ub, self.boxes_y_ub, self.boxes_w_ub, self.boxes_h_ub,
                                   self.conf_ub, score_ub, self.anchor13_ub, self.box_minx_ub, self.box_miny_ub,
                                   self.box_minw_ub, self.box_minh_ub, self.grid13_ub, self.grid_size_13,
                                   CLASSES_NUM, self.input_shape, self.image_shape)

            self.tik_inst.data_move(self.score_l1buf, score_ub, 0, 1, 80 * 512 // 16, 0, 0)

    def _concat_to_proposal(self, proposal_ub, boxes_x_ub, boxes_y_ub, boxes_w_ub, boxes_h_ub, score_ub, length):
        self.tik_inst.vconcat(proposal_ub, boxes_x_ub, length // 16, 0)
        self.tik_inst.vconcat(proposal_ub, boxes_y_ub, length // 16, 1)
        self.tik_inst.vconcat(proposal_ub, boxes_w_ub, length // 16, 2)
        self.tik_inst.vconcat(proposal_ub, boxes_h_ub, length // 16, 3)
        self.tik_inst.vconcat(proposal_ub, score_ub, length // 16, 4)

    def _extract_from_proposal(self, boxes_x_ub, boxes_y_ub, boxes_w_ub, boxes_h_ub, proposal_ub, length):
        self.tik_inst.vextract(boxes_x_ub, proposal_ub, length // 16, 0)
        self.tik_inst.vextract(boxes_y_ub, proposal_ub, length // 16, 1)
        self.tik_inst.vextract(boxes_w_ub, proposal_ub, length // 16, 2)
        self.tik_inst.vextract(boxes_h_ub, proposal_ub, length // 16, 3)

    def _yolov3_top_k(self, dst_ub, src_ub, dst_len, src_len):
        with self.tik_inst.new_stmt_scope():
            tmp_ub_size = 64 * 1024
            mem_ub = self.tik_inst.Tensor("float16", (1, tmp_ub_size // 16, 8), name="mem_ub",
                                          scope=tik.scope_ubuf)
            mem_intermediate_gm = self.tik_inst.Tensor("float16", (1, src_ub.shape[1], 8),
                                                       name="mem_intermediate_gm", scope=tik.scope_cbuf)
            self.detection_handle.topk(dst_ub, src_ub, src_len, dst_len, mem_ub,
                                       mem_intermediate_gm)

    def _yolov3_anchor_13_boxes_and_label_processor(self):
        # filter greatesize
        scorenew_ub = self.tik_inst.Tensor("float16", (512,), name="scorenew_ub", scope=tik.scope_ubuf)
        box13_proposal_ub = self.tik_inst.Tensor("float16", (1, 1024, 8), name="box13_proposal_ub",
                                                 scope=tik.scope_ubuf)
        box13_proposal_sort_ub = self.tik_inst.Tensor("float16", (1, 1024, 8), name="box13_proposal_sort_ub",
                                                      scope=tik.scope_ubuf)
        dst_ub = self.tik_inst.Tensor("float16", (80, 16), name="dst_ub", scope=tik.scope_ubuf)
        dst_int_ub = self.tik_inst.Tensor("int32", (80, 16), name="dst_int_ub", scope=tik.scope_ubuf)
        self.tik_inst.vmuls(128, dst_ub, dst_ub, 0., 80 * 16 // 128, 1, 1, 8, 8)

        with self.tik_inst.for_range(0, 80) as i:
            self.tik_inst.data_move(scorenew_ub, self.score_l1buf[i, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.vmuls(128, box13_proposal_ub, box13_proposal_ub, 0., 1024 * 8 // 128, 1, 1, 8, 8)
            self.tik_inst.vmuls(128, box13_proposal_sort_ub, box13_proposal_sort_ub, 0.,
                                1024 * 8 // 128, 1, 1, 8, 8)

            self._concat_to_proposal(box13_proposal_ub, self.box_minx_ub, self.box_miny_ub, self.box_minw_ub,
                                     self.box_minh_ub, scorenew_ub, 512)

            self._yolov3_top_k(box13_proposal_sort_ub, box13_proposal_ub, 1024, 1024)
            self._extract_from_proposal(self.boxes_x_ub, self.boxes_y_ub, self.boxes_w_ub, self.boxes_h_ub,
                                        box13_proposal_sort_ub, 512)
            self.tik_inst.vconcat(box13_proposal_ub, scorenew_ub, 512 // 16, 0)
            self._yolov3_top_k(box13_proposal_sort_ub, box13_proposal_ub, 1024, 1024)

            self.tik_inst.vextract(scorenew_ub, box13_proposal_sort_ub, 512 // 16, 0)
            self.tik_inst.vmul(128, scorenew_ub, scorenew_ub, self.mask_ub, 512 // 128, 1, 1, 1, 8, 8, 8)
            self.tik_inst.vcadd(32, dst_ub[i, 0], scorenew_ub, 512 // 32, 1, 1, 2)
            self.tik_inst.vmuls(16, dst_ub[i, 0], dst_ub[i, 0], 10., 1, 1, 1, 1, 1)
            self.tik_inst.vconv(16, "round", dst_int_ub[i, 0], dst_ub[i, 0], 1, 1, 1, 2, 1)

            with self.tik_inst.for_range(0, 16) as j:
                self.score_temp.set_as(dst_int_ub[i, j])
                self.tik_inst.vconv(16, "round", self.realint_ub, self.reallength_ub[i, 0], 1, 1, 1, 2, 1)
                self.index.set_as(self.realint_ub[0])
                with self.tik_inst.if_scope(tik.all(self.score_temp > self.score_threshold, self.index < 20)):
                    self.tik_inst.vadds(16, self.reallength_ub[i, 0], self.reallength_ub[i, 0], 1.0, 1, 1, 1, 1, 1)
                    self.tik_inst.vconcat(box13_proposal_ub, self.boxes_x_ub[32 * j], 32 // 16, 0)
                    self.tik_inst.vconcat(box13_proposal_ub, self.boxes_y_ub[32 * j], 32 // 16, 1)
                    self.tik_inst.vconcat(box13_proposal_ub, self.boxes_w_ub[32 * j], 32 // 16, 2)
                    self.tik_inst.vconcat(box13_proposal_ub, self.boxes_h_ub[32 * j], 32 // 16, 3)
                    self.tik_inst.vconcat(box13_proposal_ub, scorenew_ub[32 * j], 32 // 16, 4)
                    self.tik_inst.data_move(self.classes_ub, self.objclass_gm[i, 0], 0, 1, 32 // 16, 0, 0)
                    self.label.set_as(self.classes_ub[0])
                    self.tik_inst.vadds([0x2020202020202020, 0x2020202020202020], box13_proposal_ub,
                                        box13_proposal_ub, self.label, 8, 1, 1, 8, 8)
                    self.tik_inst.data_move(self.proposal_l1buf[i, self.index * 32, 0], box13_proposal_ub, 0, 1,
                                            32 * 8 // 16, 0, 0)
                    self.tik_inst.data_move(self.label_l1buf[i, self.index * 32], self.classes_ub[0],
                                            0, 1, 32 // 16, 0, 0)
                with self.tik_inst.else_scope():
                    pass

    def _yolov3_anchor_13_processor(self):
        with self.tik_inst.new_stmt_scope():
            self._yolov3_anchor_13_score_processor()
        with self.tik_inst.new_stmt_scope():
            self._yolov3_anchor_13_boxes_and_label_processor()

    def _yolov3_anchor_26_score_processor(self, block_m, block_n):
        data_convert_ub = self.tik_inst.Tensor("float16", (96, 512), name="data_convert_ub", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            data_ub = self.tik_inst.Tensor("float16", (512, 96), name="data_ub", scope=tik.scope_ubuf)
            # clear to zero
            with self.tik_inst.for_range(0, 2) as k:
                self.tik_inst.vmuls(128, data_ub[256 * k, 0], data_ub[256 * k, 0], 0, 256 * 96 // 128,
                                    1, 1, 8, 8)

            with self.tik_inst.for_range(0, 13) as i:
                self.tik_inst.data_move(data_ub[39 * i, 0],
                                        self.reshape_7_gm[0, (26 * (13 * block_m + i) + 13 * block_n), 0],
                                        0, 1, 13 * SHAPE_PAD_HEIGHT // 16, 0, 0)

            # refer to transpose 512x96 to 96x512
            self._tensor_transpose(data_convert_ub, data_ub)

        score_ub = self.tik_inst.Tensor("float16", (80, 512), name="score_ub", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            # extract [x, y, w, h, conf] from featboxes
            self.tik_inst.data_move(self.boxes_x_ub[0], data_convert_ub[0, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_y_ub[0], data_convert_ub[1, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_w_ub[0], data_convert_ub[2, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_h_ub[0], data_convert_ub[3, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.conf_ub[0], data_convert_ub[4, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(score_ub[0, 0], data_convert_ub[5, 0], 0, 1, 80 * 512 // 16, 0, 0)

        self.tik_inst.data_move(self.anchor13_ub, self.anchor26_gm[2 * (2 * block_m + block_n), 0], 0,
                                1,
                                1024 // 16, 0, 0)
        self.tik_inst.data_move(self.grid13_ub, self.grid26_gm, 0, 1, 1024 // 16, 0, 0)

        # main process
        self._boxes_and_scores(self.boxes_x_ub, self.boxes_y_ub, self.boxes_w_ub, self.boxes_h_ub,
                               self.conf_ub, score_ub, self.anchor13_ub, self.box_minx_ub, self.box_miny_ub,
                               self.box_minw_ub, self.box_minh_ub, self.grid13_ub, self.grid_size_26, CLASSES_NUM,
                               self.input_shape, self.image_shape)

        self.tik_inst.data_move(self.score_l1buf, score_ub, 0, 1, 80 * 512 // 16, 0, 0)

    def _yolov3_anchor_52_score_processor(self, block_m, block_n):
        data_convert_ub = self.tik_inst.Tensor("float16", (96, 512), name="data_convert_ub",
                                               scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            data_ub = self.tik_inst.Tensor("float16", (512, 96), name="data_ub", scope=tik.scope_ubuf)
            # clear to zero
            with self.tik_inst.for_range(0, 2) as k:
                self.tik_inst.vmuls(128, data_ub[256 * k, 0], data_ub[256 * k, 0], 0, 256 * 96 // 128, 1, 1, 8, 8)

            with self.tik_inst.for_range(0, 13) as i:
                self.tik_inst.data_move(data_ub[39 * i, 0],
                                        self.reshape_13_gm[0, (52 * (13 * block_m + i) + 13 * block_n), 0],
                                        0, 1, 13 * SHAPE_PAD_HEIGHT // 16, 0, 0)

            # refer to transpose 512x96 to 96x512
            self._tensor_transpose(data_convert_ub, data_ub)

        score_ub_13 = self.tik_inst.Tensor("float16", (80, 512), name="score_ub_13", scope=tik.scope_ubuf)
        with self.tik_inst.new_stmt_scope():
            # extract [x, y, w, h, conf] from featboxes
            self.tik_inst.data_move(self.boxes_x_ub[0], data_convert_ub[0, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_y_ub[0], data_convert_ub[1, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_w_ub[0], data_convert_ub[2, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.boxes_h_ub[0], data_convert_ub[3, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(self.conf_ub[0], data_convert_ub[4, 0], 0, 1, 512 // 16, 0, 0)
            self.tik_inst.data_move(score_ub_13[0, 0], data_convert_ub[5, 0], 0, 1, 80 * 512 // 16, 0, 0)
            self.tik_inst.data_move(self.anchor13_ub, self.anchor52_gm[2 * (4 * block_m + block_n), 0],
                                    0, 1, 1024 // 16, 0, 0)
            self.tik_inst.data_move(self.grid13_ub, self.grid52_gm, 0, 1, 1024 // 16, 0, 0)

            self._boxes_and_scores(self.boxes_x_ub, self.boxes_y_ub, self.boxes_w_ub, self.boxes_h_ub,
                                   self.conf_ub, score_ub_13, self.anchor13_ub, self.box_minx_ub, self.box_miny_ub,
                                   self.box_minw_ub, self.box_minh_ub, self.grid13_ub, self.grid_size_52,
                                   CLASSES_NUM, self.input_shape, self.image_shape)

            self.tik_inst.data_move(self.score_l1buf, score_ub_13, 0, 1, 80 * 512 // 16, 0, 0)

    def _yolov3_anchor_26_processor(self):
        # 26x26 anchors
        with self.tik_inst.for_range(0, 2) as block_m:
            with self.tik_inst.for_range(0, 2) as block_n:
                with self.tik_inst.new_stmt_scope():
                    self._yolov3_anchor_26_score_processor(block_m, block_n)
                with self.tik_inst.new_stmt_scope():
                    self._yolov3_anchor_13_boxes_and_label_processor()

    def _yolov3_anchor_52_processor(self):
        # 52x52 anchors
        with self.tik_inst.for_range(0, 4) as block_m:
            with self.tik_inst.for_range(0, 4) as block_n:
                with self.tik_inst.new_stmt_scope():
                    self._yolov3_anchor_52_score_processor(block_m, block_n)
                with self.tik_inst.new_stmt_scope():
                    self._yolov3_anchor_13_boxes_and_label_processor()

    def _yolov3_extract_proposal_by_score(self, selected_num):
        with self.tik_inst.if_scope(selected_num > 0):
            with self.tik_inst.for_range(0, selected_num) as num:
                with self.tik_inst.if_scope(self.outproposals_num < 1024):
                    self.all_cls_ub[0, self.outproposals_num, 0] = self.proposal_out_ub[num, 0]
                    self.all_cls_ub[0, self.outproposals_num, 1] = self.proposal_out_ub[num, 1]
                    self.all_cls_ub[0, self.outproposals_num, 2] = self.proposal_out_ub[num, 2]
                    self.all_cls_ub[0, self.outproposals_num, 3] = self.proposal_out_ub[num, 3]
                    self.all_cls_ub[0, self.outproposals_num, 4] = self.proposal_out_ub[num, 4]
                    self.labelnew_ub[self.outproposals_num] = self.sort_cls_ub[0, num, 0]
                    self.outproposals_num.set_as(1 + self.outproposals_num)
        with self.tik_inst.else_scope():
            pass

    def _filter_greater_than(self, proposals, filter_num):
        """
        Introduction
            filter proposals by scores
        ------------
        Parameters
        ----------
            @self.tik_inst, tik handle
            @proposals, shape: [1 x 1024 x 8]
            @filter_num, return filter number greater than score
        """
        # get score
        max_output_box_num = 128
        score_threshold = 0.3
        score_vector_ub = self.tik_inst.Tensor("float16", (max_output_box_num,), name="score_vector_ub",
                                               scope=tik.scope_ubuf)
        score_thresh_vector_ub = self.tik_inst.Tensor("float16", (max_output_box_num,), name="score_thresh_vector_ub",
                                                      scope=tik.scope_ubuf)
        for i in range(max_output_box_num):
            score_vector_ub[i] = proposals[0, i, 4]
        self.tik_inst.vector_dup(score_vector_ub.shape[0], score_thresh_vector_ub[0], score_threshold, 1, 1, 1)
        # compare score
        cmp_mask = self.tik_inst.vcmp_lt(score_vector_ub.shape[0], score_vector_ub[0], score_thresh_vector_ub[0], 1, 1)

        # fill 0
        zeros_ub = self.tik_inst.Tensor("float16", (score_vector_ub.shape[0],), name="zeros_ub", scope=tik.scope_ubuf)
        ones_ub = self.tik_inst.Tensor("float16", (score_vector_ub.shape[0],), name="zeros_ub", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(score_vector_ub.shape[0], zeros_ub[0], 0, 1, 1, 1)
        self.tik_inst.vector_dup(score_vector_ub.shape[0], ones_ub[0], 1, 1, 1, 1)
        self.tik_inst.vsel(score_vector_ub.shape[0], 0, score_vector_ub[0], cmp_mask, ones_ub[0],
                           zeros_ub[0], 1, 1, 1, 1, 1, 1)
        mask_scalar = self.tik_inst.Scalar("uint16", name="mask_scalar")
        zeros_scalar = self.tik_inst.Scalar('float16')
        zeros_scalar.set_as(0)
        filter_num.set_as(0)
        with self.tik_inst.for_range(0, score_vector_ub.shape[0]) as i:
            mask_scalar.set_as(score_vector_ub[i])
            with self.tik_inst.if_scope(mask_scalar == 15360):
                pass
            with self.tik_inst.else_scope():
                filter_num.set_as(filter_num + 1)

    def _yolov3_score_processor(self):
        with self.tik_inst.for_range(0, CLASSES_NUM) as idx:
            self.tik_inst.vmuls(128, self.proposal_out_ub, self.proposal_out_ub, 0., 1024 * 8 // 128, 1, 1, 8, 8)
            self.tik_inst.data_move(self.proposal_ub, self.proposal_l1buf[idx, 0, 0], 0, 1, 640 * 8 // 16, 0, 0)
            self.tik_inst.data_move(self.prop_label_ub, self.proposal_ub, 0, 1, 640 * 8 // 16, 0, 0)
            self.tik_inst.data_move(self.label_tmp_ub, self.label_l1buf[idx, 0], 0, 1, 640 // 16, 0, 0)
            self.tik_inst.vconcat(self.prop_label_ub, self.label_tmp_ub, 640 // 16, 0)
            proposals_sorted_l1 = self.tik_inst.Tensor("float16", self.proposal_ub.shape,
                                                       name="proposals_sorted_l1", scope=tik.scope_cbuf)
            self._yolov3_top_k(proposals_sorted_l1, self.proposal_ub, 1024, 1024)

            selected_num = self.tik_inst.Scalar('uint16')
            selected_num.set_as(0)
            with self.tik_inst.new_stmt_scope():
                tmp_ub = self.tik_inst.Tensor("float16", proposals_sorted_l1.shape,
                                              name="mem_intermediate_gm", scope=tik.scope_ubuf)
                self.tik_inst.data_move(tmp_ub, proposals_sorted_l1, 0, 1, proposals_sorted_l1.shape[1] * 2 // 32, 0,
                                        0)
                self._filter_greater_than(tmp_ub, selected_num)

            output_num = 128
            burst_num = 512
            input_num = 1024
            down_factor = 0.5
            nms_thresh = 0.5
            param = Param(input_num, burst_num, output_num, down_factor, nms_thresh, "tensorflow")
            self.proposal_out_ub = self.proposal_out_ub.reshape((1, self.proposal_out_ub.shape[0],
                                                                 self.proposal_out_ub.shape[1]))
            with self.tik_inst.new_stmt_scope():
                local_tensor = self.detection_handle.gen_nms_local_tensor(output_num, burst_num, tik.scope_ubuf)
                self.detection_handle.nms_after_sorted(self.proposal_out_ub, proposals_sorted_l1, param, local_tensor)
                self.proposal_out_ub = self.proposal_out_ub.reshape((self.proposal_out_ub.shape[1], 8))
            with self.tik_inst.new_stmt_scope():
                self._yolov3_top_k(self.sort_cls_ub, self.prop_label_ub, 1024, 1024)
                self._yolov3_extract_proposal_by_score(selected_num)

    def _clear_to_zero(self):
        self.tik_inst.vmuls(128, self.all_cls_ub, self.all_cls_ub, 0., 1024 * 8 // 128, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, self.sort_cls_ub, self.sort_cls_ub, 0., 1024 * 8 // 128, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, self.label_tmp_ub, self.label_tmp_ub, 0., 1024 // 128, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, self.labelnew_ub, self.labelnew_ub, 0., 1024 // 128, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, self.proposal_ub, self.proposal_ub, 0., 1024 * 8 // 128, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, self.prop_label_ub, self.prop_label_ub, 0., 1024 * 8 // 128, 1, 1, 8, 8)
        self.tik_inst.vmuls(128, self.box_out_ub, self.box_out_ub, 0., 128 * 8 // 128, 1, 1, 8, 8)

    def _yolov3_merge_post_processor(self):
        self.proposal_ub = self.tik_inst.Tensor("float16", (1, 1024, 8), name="proposal_ub", scope=tik.scope_ubuf)
        self.prop_label_ub = self.tik_inst.Tensor("float16", (1, 1024, 8), name="prop_label_ub", scope=tik.scope_ubuf)
        self.proposal_out_ub = self.tik_inst.Tensor("float16", (1024, 8), name="proposal_out_ub", scope=tik.scope_ubuf)
        self.label_tmp_ub = self.tik_inst.Tensor("float16", (1024,), name="label_tmp_ub", scope=tik.scope_ubuf)
        self.labelnew_ub = self.tik_inst.Tensor("float16", (1024,), name="labelnew_ub", scope=tik.scope_ubuf)
        self.all_cls_ub = self.tik_inst.Tensor("float16", (1, 1024, 8), name="all_cls_ub", scope=tik.scope_ubuf)
        self.sort_cls_ub = self.tik_inst.Tensor("float16", (1, 1024, 8), name="sort_cls_ub", scope=tik.scope_ubuf)
        self.box_out_ub = self.tik_inst.Tensor("float16", (128, 8), name="box_out_ub", scope=tik.scope_ubuf)
        self.outproposals_num = self.tik_inst.Scalar("uint16")
        self.outproposals_num.set_as(0)
        self._clear_to_zero()

        self._yolov3_score_processor()
        self._yolov3_top_k(self.sort_cls_ub, self.all_cls_ub, 1024, 1024)
        self.tik_inst.vconcat(self.all_cls_ub, self.labelnew_ub, 1024 // 16, 0)

        self._yolov3_top_k(self.prop_label_ub, self.all_cls_ub, 1024, 1024)
        self.tik_inst.vextract(self.labelnew_ub, self.prop_label_ub, 1024 // 16, 0)
        self.index.set_as(0)
        score_thr = self.tik_inst.Scalar("float16")
        score_float16 = self.tik_inst.Tensor("float16", (16,), name="score_float16", scope=tik.scope_ubuf)
        score_fp16 = self.tik_inst.Tensor("float16", (16,), name="score_fp16", scope=tik.scope_ubuf)
        score_int32 = self.tik_inst.Tensor("int32", (16,), name="score_int32", scope=tik.scope_ubuf)
        score_min = self.tik_inst.Tensor("float16", (16,), name="score_min", scope=tik.scope_ubuf)
        score_max = self.tik_inst.Tensor("float16", (16,), name="score_max", scope=tik.scope_ubuf)
        self.tik_inst.vector_dup(16, score_min, 0.0, 1, 1, 1)
        self.tik_inst.vector_dup(16, score_max, 1.0, 1, 1, 1)

        with self.tik_inst.for_range(0, 100) as idx:
            score_thr.set_as(self.sort_cls_ub[0, idx, 4])
            self.tik_inst.vector_dup(16, score_float16, score_thr, 1, 1, 1)
            self.tik_inst.vector_dup(16, score_fp16, score_thr, 1, 1, 1)
            self.tik_inst.vmax(16, score_fp16, score_fp16, score_min, 1, 1, 1, 1, 1, 1, 1)
            self.tik_inst.vmin(16, score_fp16, score_fp16, score_max, 1, 1, 1, 1, 1, 1, 1)
            self.tik_inst.vmuls(16, score_float16, score_float16, 10.0, 1, 1, 1, 1, 1)
            self.tik_inst.vconv(16, "floor", score_int32, score_float16, 1, 1, 1, 2, 1)
            self.score_threshold.set_as(score_int32[0])
            with self.tik_inst.if_scope(self.score_threshold >= MIN_SCORE):
                self.box_out_ub[self.index, 0] = self.sort_cls_ub[0, idx, 1]
                self.box_out_ub[self.index, 1] = self.sort_cls_ub[0, idx, 0]
                self.box_out_ub[self.index, 2] = self.sort_cls_ub[0, idx, 3]
                self.box_out_ub[self.index, 3] = self.sort_cls_ub[0, idx, 2]
                self.box_out_ub[self.index, 4] = score_fp16[0]
                self.box_out_ub[self.index, 5] = self.labelnew_ub[idx]
                self.index.set_as(self.index + 1)
            with self.tik_inst.else_scope():
                pass
        self.tik_inst.data_move(self.output_gm, self.box_out_ub, 0, 1, 100 * 8 // 16, 0, 0)

    def compute(self):
        """
        Introduction
            TF Yolov3 detectoutput Fusion op
        ------------
        Parameters
        ----------
            @input0: Reshape_1,13x13x3x85, shape: ND [SHAPE_13_WIDTH x 288]
            @input1: Reshape_7,26x26x3x85, shape: ND [676 x 288]
            @input2: Reshape_13,52x52x3x85, shape: ND  [2704 x 288]
            @anchor13: reshape_1 13x13 anchors, op: const, dtype: float32, shape: ND [2x512]
            @anchor26: reshape_7 26x26 anchors, op: const, dtype: float32, shape: ND [8x512]
            @anchor52: reshape_13 52x52 anchors, op: const, dtype: float32, shape: ND [32x512]
            @grid13: reshape_1 grids, op: const, dtype: float32, shape: ND [2x512]
            @grid26: reshape_7 grids, op: const, dtype: float32, shape: ND [2x512]
            @grid52: reshape_13 grids, op: const, dtype: float32, shape: ND [2x512]
            @mask512: mask for score, op: const, dtype: float32, shape: ND [512]
            @objclasses: coco 80 classes, op: const, dtype: float32, shape: ND [80x128]
            @output: dectect results for proposal format [ymin, xmin, ymax, xmax, score, label, 0, 0],
                     dtype: float16, shape: ND[100x8]
            @kernel_name
        """
        with self.tik_inst.new_stmt_scope():
            self.anchor13_ub = self.tik_inst.Tensor("float16", (2, 512), name="anchor13_ub", scope=tik.scope_ubuf)
            self.grid13_ub = self.tik_inst.Tensor("float16", (2, 512), name="grid13_ub", scope=tik.scope_ubuf)
            self.mask_ub = self.tik_inst.Tensor("float16", (512,), name="mask_ub", scope=tik.scope_ubuf)
            self.box_minx_ub = self.tik_inst.Tensor("float16", (512,), name="box_minx_ub", scope=tik.scope_ubuf)
            self.box_miny_ub = self.tik_inst.Tensor("float16", (512,), name="box_miny_ub", scope=tik.scope_ubuf)
            self.box_minw_ub = self.tik_inst.Tensor("float16", (512,), name="box_minw_ub", scope=tik.scope_ubuf)
            self.box_minh_ub = self.tik_inst.Tensor("float16", (512,), name="box_minh_ub", scope=tik.scope_ubuf)
            self.boxes_x_ub = self.tik_inst.Tensor("float16", (512,), name="boxes_x_ub", scope=tik.scope_ubuf)
            self.boxes_y_ub = self.tik_inst.Tensor("float16", (512,), name="boxes_y_ub", scope=tik.scope_ubuf)
            self.boxes_w_ub = self.tik_inst.Tensor("float16", (512,), name="boxes_w_ub", scope=tik.scope_ubuf)
            self.boxes_h_ub = self.tik_inst.Tensor("float16", (512,), name="boxes_h_ub", scope=tik.scope_ubuf)
            self.conf_ub = self.tik_inst.Tensor("float16", (512,), name="conf_ub", scope=tik.scope_ubuf)
            self.classes_ub = self.tik_inst.Tensor("float16", (128,), name="classes_ub", scope=tik.scope_ubuf)
            self.reallength_ub = self.tik_inst.Tensor("float16", (80, 16), name="reallength_ub", scope=tik.scope_ubuf)
            self.realint_ub = self.tik_inst.Tensor("int32", (16,), name="realint_ub", scope=tik.scope_ubuf)

            self.tik_inst.vmuls(128, self.reallength_ub, self.reallength_ub, 0, 80 * 16 // 128, 1, 1, 8, 8)
            self.tik_inst.data_move(self.mask_ub, self.mask_gm, 0, 1, 512 // 16, 0, 0)

            with self.tik_inst.new_stmt_scope():
                zero_ub = self.tik_inst.Tensor("float16", (640, 8), name="zero_ub", scope=tik.scope_ubuf)
                self.tik_inst.vmuls(128, zero_ub, zero_ub, 0, 640 * 8 // 128, 1, 1, 8, 8)
                # clear l0C
                with self.tik_inst.for_range(0, 80, thread_num=2) as i:
                    self.tik_inst.data_move(self.proposal_l1buf[i, 0, 0], zero_ub, 0, 1, 640 * 8 // 16, 0, 0)
                    self.tik_inst.data_move(self.label_l1buf[i, 0], zero_ub, 0, 1, 640 // 16, 0, 0)

            # 13x13 anchors
            self._yolov3_anchor_13_processor()

            # 26x26 anchors
            self._yolov3_anchor_26_processor()

            # 52x52 anchors
            self._yolov3_anchor_52_processor()

        with self.tik_inst.new_stmt_scope():
            self._yolov3_merge_post_processor()

        self.tik_inst.BuildCCE(self.kernel_name,
                               inputs=[self.reshape_1_gm, self.reshape_7_gm, self.reshape_13_gm, self.anchor13_gm,
                                       self.anchor26_gm, self.anchor52_gm, self.grid13_gm, self.grid26_gm,
                                       self.grid52_gm, self.mask_gm, self.objclass_gm],
                               outputs=[self.output_gm],
                               enable_l2=False)
        return self.tik_inst


def yolov3_post_processor(input0, input1, input2, anchor13, anchor26, anchor52, grid13, grid26, grid52, mask512,
                          objclasses, output, kernel_name="yolov3postprocessor"):
    input0_shape = input0.get("shape")
    input1_shape = input1.get("shape")
    input2_shape = input2.get("shape")
    anchor13_shape = anchor13.get("shape")
    anchor26_shape = anchor26.get("shape")
    anchor52_shape = anchor52.get("shape")
    grid13_shape = grid13.get("shape")
    grid26_shape = grid26.get("shape")
    grid52_shape = grid52.get("shape")
    mask512_shape = mask512.get("shape")
    obj_classes_shape = objclasses.get("shape")
    output_shape = output.get("shape")
    obj = Yolov3PostProcessor(input0_shape, input1_shape, input2_shape, anchor13_shape, anchor26_shape,
                              anchor52_shape, grid13_shape, grid26_shape, grid52_shape, mask512_shape,
                              obj_classes_shape, output_shape, kernel_name)
    obj.compute()
