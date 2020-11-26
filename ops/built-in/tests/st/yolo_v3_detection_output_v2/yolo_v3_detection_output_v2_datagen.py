import numpy as np
import math
from dataFormat import *
import sys


FROM_FILE = False
BASE_DIR = "./../data/yolo_v3_detection_output_v2/"


def gen_yolo_v3_data():
    sys.stdout.write("Info: writing input for...\n")
    biases = [(116, 90, 156, 198, 373, 326), (30, 61, 62, 45, 59, 119),
              (10, 13, 16, 30, 33, 23)]
    biases_yolov2 = [(0.572730, 0.677385, 1.874460, 2.062530, 3.338430,
                      5.474340, 7.882820, 3.527780, 9.770520, 9.168280)]

    gen_data(1, "float16", [[4, 4]],
             biases_yolov2, boxes=5, classes=2, pre_nms_topn=1024,
             resize_origin_img_to_net=True)
    gen_data(1, "float16", [[13, 13]],
             biases_yolov2, boxes=5, classes=2, pre_nms_topn=1024,
             resize_origin_img_to_net=False)
    gen_data(1, "float16", [[19, 19]],
             biases_yolov2, boxes=5, classes=10, pre_nms_topn=1024,
             resize_origin_img_to_net=True)

    gen_data(1, "float16", [[4, 4], [4, 4], [4, 4]],
             biases, classes=2, pre_nms_topn=1024, resize_origin_img_to_net=True)

    gen_data(1, "float16", [[13, 13], [13, 13], [13, 13]],
             biases, classes=1, pre_nms_topn=1024, resize_origin_img_to_net=False)

    gen_data(1, "float16", [[13, 13], [26, 26], [52, 52]],
             biases, classes=80, pre_nms_topn=1024, resize_origin_img_to_net=True)


def gen_data(batch, dtype, box_info, biases,
             boxes=3, coords=4, classes=80,
             relative=True, obj_threshold=0.5,
             post_nms_topn=1024, score_threshold=0.5,
             iou_threshold=0.45, pre_nms_topn=512,
             N=10, resize_origin_img_to_net=False,
             max_box_number_per_batch=1024):
    box_info_detail = []
    fold = ''
    for info in box_info:
        box_info_detail.append({"shape": (batch, boxes * (4 + 1 + classes),
                                          info[0], info[1]), "dtype": dtype,
                                "format": "NCHW"})
        fold += str(info[0] * info[1]) + '_'
    fold += str(dtype)
    gen_detection_golden(box_info_detail, biases, boxes, coords, classes,
                         relative, obj_threshold, score_threshold, iou_threshold,
                         pre_nms_topn, resize_origin_img_to_net, fold)


def gen_detection_golden(box_info, biases, boxes, coords, classes, relative,
                         obj_threshold, score_threshold, iou_threshold,
                         pre_nms_topn, resize_origin_img_to_net, fold):
    correct_region_box(box_info, biases, boxes, relative,
                       resize_origin_img_to_net, fold)
    cls_prob(box_info, boxes, classes, obj_threshold, fold)
    nms(box_info, boxes, coords, classes, obj_threshold, pre_nms_topn,
        score_threshold, iou_threshold, fold)


def get_adj_hw(h, w,dsize):
    return math.ceil((h*w + (32 // dsize) - 1)/(32//dsize)) * (32//dsize)


def get_totalhw(box_info):
    hwtotal = 0
    for info in box_info:
        hwtotal += info.get("shape")[2] * info.get("shape")[3]
    return hwtotal


def correct_region_box(box_info, biases, boxes, relative,
                       resize_origin_img_to_net, fold):
    batch = box_info[0].get("shape")[0]
    dtype = box_info[0].get("dtype")

    if FROM_FILE:
        img_info = np.fromfile(
            BASE_DIR+fold+"/img_info.data",
            dtype=dtype).reshape((batch, 4))
    else:
        img_info = []
        for i in range(0, batch):
            # netH,netW,scaleH,scaleW
            # img_info += [416, 416, 576, 768]
            img_info += [416, 416, 768, 576]
        img_info = np.array(img_info).reshape(batch, 4)

        # img_info = np.random.randint(13, 14, size=(batch, 4))
        # img_info[0,:4] = np.array([416, 416, 576, 768]).astype(dtype)
        dumpData(img_info, "/img_info.data", fmt="binary", data_type=dtype,
                 path=BASE_DIR+fold)
    # print(img_info)
    # shape (batch, 4*n, adj_hw)
    hwtotal = get_totalhw(box_info)
    output_coords_ret = np.zeros(shape=(batch, 4, boxes * hwtotal))
    idx = 0
    offset = 0
    for info in box_info:
        output_coords = do_correct_region_box(
            info, boxes, img_info, biases[idx], relative, str(idx + 1),
            resize_origin_img_to_net, fold)
        h = info.get("shape")[2]
        w = info.get("shape")[3]
        for bi in range(0, batch):
            for i in range(0, 4):
                output_coords_ret[bi, i:i+1, offset:offset+h*w*boxes] = \
                    output_coords[bi, i*boxes:(i+1)*boxes,
                    0:h*w].reshape(1, h*w*boxes)
        offset = offset + h*w*boxes
        idx += 1

    dataExpect = output_coords_ret.flatten()
    for i in range(len(dataExpect)):
        if str(dataExpect[i]) =='nan' or str(dataExpect[i]) =='inf':
            print(i, dataExpect[i])

    print("output_coords_ret,", output_coords_ret.shape)
    dumpData(output_coords_ret, "/inter_coords.data", fmt="binary",
             data_type=dtype, path=BASE_DIR+fold)


def do_correct_region_box(box_info, boxes, img_info, biases, relative,
                          box_id, resize_origin_img_to_net, fold):
    shape = box_info.get("shape")
    dtype = box_info.get("dtype")
    dsize = 2 if dtype == "float16" else 4
    n = boxes  # V3=3,V2=5
    batch = shape[0]
    h = shape[2]
    w = shape[3]

    # ����(N,n*4,lh,lw)����������
    adj_hw = math.ceil((h*w+32//dsize-1)/(32//dsize)) * (32//dsize)

    if FROM_FILE:
        input_coords = np.fromfile(
            BASE_DIR+fold+"/coord_data"+box_id+".data",
            dtype=dtype).reshape((batch, boxes*4, adj_hw))
        print("coord_data"+box_id+",shape,", input_coords.shape)
    else:
        input_coords = \
            np.random.uniform(0, 3, size=(batch, 4*n, adj_hw)).astype(dtype)
        # input_coords[:,0:2*n,:] = np.random.randint(20, 80, size=(batch, 2*n, adj_hw)).astype(dtype)
        input_coords[:, 0:2*n, :] = \
            np.random.uniform(0, 1, size=(batch, 2*n, adj_hw)).astype(dtype)
        input_coords[:, 2*n:4*n, :] = \
            np.random.uniform(0, 1, size=(batch, 2*n, adj_hw)).astype(dtype)

        print("coord_data"+box_id+",shape,", input_coords.shape)
        dumpData(input_coords, "/coord_data"+box_id+".data", fmt="binary",
                 data_type=dtype, path=BASE_DIR+fold)

    if FROM_FILE:
        wIndex = np.fromfile(
            BASE_DIR+fold+"/windex"+box_id+".data",
            dtype=dtype).reshape((adj_hw,))
    else:
        # ���츨��index---W����
        wIndex_data = np.zeros(shape=(h, w))
        for i in range(0, h):
            wIndex_data[i, :] = np.arange(0, w)

        wIndex = np.zeros(shape=(adj_hw,))
        wIndex[0:h*w] = wIndex_data.flatten()

        print("wIndex"+box_id+",shape,", wIndex.shape)
        dumpData(wIndex, "/windex"+box_id+".data", fmt="binary",
                 data_type=dtype, path=BASE_DIR+fold)

    if FROM_FILE:
        hIndex = np.fromfile(
            BASE_DIR+fold+"/hindex"+box_id+".data",
            dtype=dtype).reshape((adj_hw,))
    else:
        # ���츨��index---W����
        hIndex_data = np.zeros(shape=(h, w))
        for i in range(0, h):
            hIndex_data[i, :] = i
        hIndex = np.zeros(shape=(adj_hw,))
        hIndex[0:h * w] = hIndex_data.flatten()

        print("hindex"+box_id+",shape,", hIndex.shape)
        dumpData(hIndex, "/hindex"+box_id+".data", fmt="binary",
                 data_type=dtype,
                 path=BASE_DIR+fold)

    output_coords = np.zeros(shape=(batch, 4*n, adj_hw)).astype(dtype)
    for bi in range(0, batch):
        netH = img_info[bi, 0]
        netW = img_info[bi, 1]
        scaleH = img_info[bi, 2]
        scaleW = img_info[bi, 3]
        print(netH, netW, scaleH, scaleW)
        # w > h
        if not resize_origin_img_to_net:
            if float(netW / scaleW) < float(netH / scaleH):
                # print(scaleH * netW)
                new_w = netW
                new_h = (scaleH / scaleW) * netW
            else:
                new_h = netH
                new_w = (scaleW / scaleH) * netH
        else:
            new_h = netH
            new_w = netW
        for ci in range(4*n):
            n_idx = ci % n
            # x
            if ci // n == 0:
                output_coords[bi, ci, :] = \
                    (input_coords[bi, ci, :] + wIndex) * (1.0/w)
                output_coords[bi, ci, :] = \
                    (output_coords[bi, ci, :] -
                     (netW - new_w) / 2.0 / netW) / (new_w / netW)
                # x_vmuls_val = netW / new_w
                # x_vadds_val = ((-1)*(netW / new_w) + 1)* 0.5
                # print("x_vmuls_val in  golden ", x_vmuls_val)
                # print("x_vadds_val in  golden ", x_vadds_val)
                if not relative:
                    output_coords[bi, ci, :] = output_coords[bi, ci, :] * scaleW
            # y
            elif ci // n == 1:
                output_coords[bi, ci, :] = (input_coords[bi, ci, :] + hIndex)
                output_coords[bi, ci, :] = output_coords[bi, ci, :] * (1.0/h)
                output_coords[bi, ci, :] = \
                    (output_coords[bi, ci, :] -
                     (netH - new_h) / 2.0 / netH) / (new_h / netH)
                # y_vmuls_val = netH / new_h
                # y_vadds_val = ((-1)*(netH / new_h) + 1)* 0.5
                # print("y_vmuls_val in  golden ", y_vmuls_val)
                # print("y_vadds_val in  golden ", y_vadds_val)
                # output_coords[bi, ci, :] = output_coords[bi, ci, :]*y_vmuls_val

                if not relative:
                    output_coords[bi, ci, :] = output_coords[bi, ci, :] * scaleH
            # h
            elif ci // n == 2:
                output_coords[bi, ci, :] = \
                    np.exp(input_coords[bi, ci, :]) * biases[2*n_idx+1] / netH
                # output_coords[bi, ci, :] = np.exp(input_coords[bi,ci,:]) * biases[n_idx,1] / netH
                output_coords[bi, ci, :] = output_coords[bi, ci, :] * (netH / new_h)
                if not relative:
                    output_coords[bi, ci, :] = output_coords[bi, ci, :] * scaleH
            # w
            else:
                output_coords[bi, ci, :] = \
                    np.exp(input_coords[bi, ci, :]) * biases[2*n_idx] / netW
                # output_coords[bi, ci, :] = np.exp(input_coords[bi, ci, :]) * biases[n_idx,0] / netW
                output_coords[bi, ci, :] = output_coords[bi, ci, :] * (netW / new_w)
                if not relative:
                    output_coords[bi, ci, :] = output_coords[bi, ci, :] * scaleW
        # print(output_coords[0,n:2*n,:])
    return output_coords


def cls_prob(box_info, boxes, classes, obj_threshold, fold):
    batch = box_info[0].get("shape")[0]
    dtype = box_info[0].get("dtype")

    hwtotal = get_totalhw(box_info)
    inter_obj = np.zeros(shape=(batch, boxes * hwtotal))
    output_class_ret = np.zeros(shape=(batch, classes, boxes*hwtotal))

    idx = 0
    offset = 0
    for info in box_info:
        h = info.get("shape")[2]
        w = info.get("shape")[3]
        ret_c, ret_b = do_cls_prob(info, boxes, classes, obj_threshold,
                                 str(idx+1), fold)
        inter_obj[0, offset:offset+boxes*h*w] = ret_b.flatten()[:boxes*h*w]

        for b in range(0, batch):
            for i in range(0, classes):
                output_class_ret[b, i, offset:offset+boxes*h*w] = \
                    ret_c[b, i, :boxes*h*w]
        offset += boxes*h*w
        idx += 1

    if not FROM_FILE:
        dumpData(inter_obj, "/inter_obj.data", fmt="binary", data_type=dtype,
                 path=BASE_DIR+fold)

    if not FROM_FILE:
        print("output_class_ret,", output_class_ret.shape)
        dumpData(output_class_ret, "/inter_class.data", fmt="binary",
                 data_type=dtype, path=BASE_DIR+fold)


def do_cls_prob(box_info, boxes, classes, obj_threshold, box_id, fold):
    shape = box_info.get("shape")
    dtype = box_info.get("dtype")
    dsize = 2 if dtype == "float16" else 4
    n = boxes  # V3=3,V2=5
    batch = shape[0]
    h = shape[2]
    w = shape[3]

    adj_len = get_adj_hw(boxes*h, w, dsize)
    if FROM_FILE:
        input_b = np.fromfile(BASE_DIR+fold+"/input_b"+box_id+".data",
                              dtype=dtype).reshape((batch * adj_len,))
    else:
        input_b = np.random.uniform(0.1, 1, size=(batch * adj_len)).astype(dtype)
        print("input_b"+box_id+",shape,", input_b.shape)
        dumpData(input_b, "/input_b"+box_id+".data", fmt="binary",
                 data_type=dtype, path=BASE_DIR+fold)

    if FROM_FILE:
        input_c = np.fromfile(BASE_DIR+fold+"/input_c"+box_id+".data",
                              dtype=dtype).reshape((batch, classes, adj_len))
    else:
        # ����classes
        input_c = np.random.uniform(
            0.6, 1, size=(batch, classes, adj_len)).astype(dtype)
        print("input_c"+box_id+",shape,", input_c.shape)
        dumpData(input_c, "/input_c"+box_id+".data", fmt="binary",
                 data_type=dtype, path=BASE_DIR+fold)

    for i in range(0, len(input_b)):
        if input_b[i] < obj_threshold:
            input_b[i] = 0
        # else:
        #     print("##############idx:",i,"value:",input_b[i])
    input_b = input_b.reshape((batch, adj_len))
    ret_c = np.zeros(shape=(batch, classes, adj_len)).astype(dtype)

    for b in range(0, batch):
        for i in range(0, classes):
            ret_c[b, i:i+1, :] = input_b[b, :]*input_c[b, i, :]
    return ret_c, input_b


def nms(box_info, boxes, coords, classes, obj_threshold, pre_nums_topn,
        classes_threshold, overlap_threshold, fold):
    shape = box_info[0].get("shape")
    dtype = box_info[0].get("dtype")
    dsize = 2 if dtype == "float16" else 4
    n = boxes  # V3=3,V2=5
    batch = shape[0]
    # h = shape[1]
    # w = shape[2]

    # batch, 4, boxes*(h1*w1+h2*w2+h3*w3)
    hwtotal = get_totalhw(box_info)
    coord_data = np.fromfile(BASE_DIR+fold+"/inter_coords.data",
                             dtype=dtype).reshape((batch, 4, boxes*hwtotal))

    total_wh = boxes*hwtotal
    input_obj = np.random.uniform(0, 1, size=(batch, total_wh)).astype(dtype)

    i = 0
    offset = 0
    for info in box_info:
        h = info.get("shape")[2]
        w = info.get("shape")[3]
        adj_len = get_adj_hw(h*w*boxes, 1, dsize)
        input_b = np.fromfile(BASE_DIR+fold+"/input_b"+str(i+1)+".data",
                              dtype=dtype).reshape((batch, adj_len))
        input_obj[:, offset:offset + h*w*boxes] = input_b[:, 0:h*w*boxes]
        offset += h*w*boxes
        i += 1

    count_1 = 0
    bindex_arr = []
    for i in range(0, total_wh):
        if input_obj[0, i] > obj_threshold:
            count_1 += 1
            # print("######### idx,",i,"value,",input_obj[0,i])
            # if i < 500:
            # print("------x,y,h,w",ori_coords[0,i],
            # ori_coords[1,i],ori_coords[2,i],ori_coords[3,i])
            print("------bindex, b,x,y,h,w", i, input_obj[0, i],
                  coord_data[0, 0, i], coord_data[0, 1, i],
                  coord_data[0, 2, i], coord_data[0, 3, i])
            bindex_arr.append(i+1)

    print("after b threshold!,", count_1)
    print("bindex_arr:", bindex_arr)

    input_score = np.fromfile(BASE_DIR+fold+"/inter_class.data",
                              dtype=dtype).reshape((batch, classes, total_wh))
    img_info = np.fromfile(BASE_DIR+fold+"/img_info.data",
                           dtype=dtype).reshape((batch, 4))


    # filter obj
    print("coord_data==", coord_data)
    print("input_score==", input_score)
    hw = hwtotal
    coord_data_filterd = \
        np.array([0.0] * n * 1 * hw * 5*batch).reshape(batch, 5, total_wh)
    input_score_filterd = \
        np.array([0.0] * n * 1 * hw * classes*batch).reshape(batch, classes,
                                                             total_wh)
    index = []
    print("hw===", n * 1 * hw)
    count_ret = np.array([0] * batch).reshape(batch, 1).astype("int32")
    for b in range(0, batch):
        count = 0
        if count >= pre_nums_topn:
            break
        for j in range(0, total_wh):
            # print("j===",j)
            if count >= pre_nums_topn:
                break

            if input_obj[b, j] > obj_threshold:
                coord_data_filterd[b, 0, count] = coord_data[b, 0, j]
                coord_data_filterd[b, 1, count] = coord_data[b, 1, j]
                coord_data_filterd[b, 2, count] = coord_data[b, 2, j]
                coord_data_filterd[b, 3, count] = coord_data[b, 3, j]
                coord_data_filterd[b, 4, count] = input_obj[b, j]
                index.append(j+1)
                for m in range(0, classes):
                    input_score_filterd[b, m, count] = input_score[b, m, j]
                count = count + 1
        print("count==", count)
        count_ret[b, 0] = count

    for i in range(0, batch):
        print("coord_data_filterd334==", coord_data_filterd[i, ])
        print("input_score_filterd334==", input_score_filterd[i, ])
        print("input_obj334==", input_obj[i, ])

    print("index===", index)
    # sort and nms
    # ���box_out
    box_out = []
    # ���Box_out_num
    box_out_num = []
    pre_flag = False
    selected_rou=0
    for b in range(0, batch):
        tmp_inbatch_score = []
        tmp_inbatch_label = []

        tmp_x_list = []
        tmp_y_list = []
        tmp_h_list = []
        tmp_w_list = []
        num_proposals = 0

        for ci in range(0, classes):
            class_filtered_data = []
            class_filtered_data123 = []
            selected_class = 0
            print("count:selected_class==", count_ret[b,0])
            # filter class
            for k in range(0, count_ret[b, 0]):
                if input_score_filterd[b, ci, k] > classes_threshold:
                    class_filtered_data.append((coord_data_filterd[b, 0, k],
                                                coord_data_filterd[b, 1, k],
                                                coord_data_filterd[b, 2, k],
                                                coord_data_filterd[b, 3, k],
                                                input_score_filterd[b, ci, k],
                                                coord_data_filterd[b, 4, k]))
                    class_filtered_data123.append(
                        (coord_data_filterd[b, 0, k] -
                         coord_data_filterd[b, 3, k]/2,
                         coord_data_filterd[b, 1, k] -
                         coord_data_filterd[b, 2, k]/2,
                         coord_data_filterd[b, 0, k] +
                         coord_data_filterd[b, 3, k]/2,
                         coord_data_filterd[b, 1, k] +
                         coord_data_filterd[b, 2, k]/2,
                         input_score_filterd[b, ci, k]))
                    selected_class = selected_class+1
            # x y h w score objectness
            print("selected_class==", selected_class)
            class_filtered_data.sort(reverse=True, key=lambda x: x[-2])
            print("lenght==", len(class_filtered_data))
            print("class_filtered_data==", class_filtered_data)
            class_filtered_data123.sort(reverse=True, key=lambda x: x[-1])
            print("class_filtered_data123==", class_filtered_data123)

            # do rou
            # ���濪ʼnmsɸѡ
            # ������бȽϣ��������Ƿ���
            length=len(class_filtered_data)
            suppress_vector = [False for i in range(length)]
            for i in range(length):
                if not suppress_vector[i]:
                    for j in range(i + 1, length):
                        if not suppress_vector[j]:
                            # (xyhwc)
                            data_i = class_filtered_data[i]
                            data_j = class_filtered_data[j]
                            x_i, y_i, h_i, w_i = data_i[0], data_i[1], data_i[2], data_i[3]
                            x_j, y_j, h_j, w_j = data_j[0], data_j[1], data_j[2], data_j[3]

                            # ת���ɽ�����
                            '''
                            x_i_1 = x_i - w_i / 2
                            y_i_1 = y_i - h_i / 2
                            x_i_2 = x_i + w_i / 2
                            y_i_2 = y_i + h_i / 2

                            x_j_1 = x_j - w_j / 2
                            y_j_1 = y_j - h_j / 2
                            x_j_2 = x_j + w_j / 2
                            y_j_2 = y_j + h_j / 2
                            '''
                            x_i_1 = (x_i - w_i / 2)* img_info[b,3]
                            y_i_1 = (y_i - h_i / 2)* img_info[b,2]
                            x_i_2 = (x_i + w_i / 2)* img_info[b,3]
                            y_i_2 = (y_i + h_i / 2)* img_info[b,2]

                            x_j_1 = (x_j - w_j / 2)* img_info[b,3]
                            y_j_1 = (y_j - h_j / 2)* img_info[b,2]
                            x_j_2 = (x_j + w_j / 2)* img_info[b,3]
                            y_j_2 = (y_j + h_j / 2)* img_info[b,2]

                            xx1 = max(x_i_1, x_j_1)
                            yy1 = max(y_i_1, y_j_1)
                            xx2 = min(x_i_2, x_j_2)
                            yy2 = min(y_i_2, y_j_2)

                            w_iou = max(0, xx2 - xx1 + 1)
                            h_iou = max(0, yy2 - yy1 + 1)

                            # ����ص�����
                            iou = w_iou * h_iou
                            # ������Ե��������
                            area_i = (x_i_2-x_i_1+1) * (y_i_2-y_i_1+1)
                            area_j = (y_j_2-y_j_1+1) * (x_j_2-x_j_1+1)
                            # ���nms����ֵ
                            if iou > (area_i + area_j)*(overlap_threshold/(1+overlap_threshold)):
                                '''
                                print("#########################################")
                                print("#########################################", data_j[4])
                                print("#########################################=",x_i_1,y_i_1,x_i_2,y_i_2)
                                print("#########################################=",x_j_1,y_j_1,x_j_2,y_j_2)
                                print("#########################################=",x_i,y_i,h_i,w_i)
                                print("#########################################=",x_j,y_j,h_j,w_j)
                                print("iou==",iou,"area_i + area_j==",area_i + area_j,area_i,area_j)
                                '''
                                suppress_vector[j] = True

            # nms��ֵɸѡ
            proposals = []
            for i in range(length):
                if not suppress_vector[i]:
                    proposals.append(class_filtered_data[i])


            num_proposals = num_proposals + len(proposals)
            print("length==", len(proposals))
            print("proposals==", proposals)

            # print("img_info, h, w",img_info[0,0],img_info[0,1])
            # ƴ�����յĽ��
            # (b,wi,hi,ni,ci,score) ci����label
            for i, p in enumerate(proposals):
                x_i, y_i, h_i, w_i, b_i = p[0],p[1],p[2],p[3],p[5]
                print("x_i, y_i, h_i, w_i, b_i",x_i, y_i, h_i, w_i, b_i)

                x_i_1 = (x_i - w_i / 2) * img_info[b,3]
                y_i_1 = (y_i - h_i / 2) * img_info[b,2]
                x_i_2 = (x_i + w_i / 2) * img_info[b,3]
                y_i_2 = (y_i + h_i / 2) * img_info[b,2]
                if x_i_1 < 0: x_i_1 = 0
                if y_i_1 < 0: y_i_1 = 0
                if x_i_2 > img_info[b, 3] - 1: x_i_2 = img_info[b, 3] - 1
                if y_i_2 > img_info[b, 2] - 1: y_i_2 = img_info[b, 2] - 1
                score = p[4]
                label = ci
                tmp_x_list.append(x_i_1)
                tmp_y_list.append(y_i_1)
                tmp_h_list.append(x_i_2)
                tmp_w_list.append(y_i_2)
                tmp_inbatch_score.append(score)
                tmp_inbatch_label.append(label)

        # ���1��bacth�����е�class�����ݳ�ȡ��������ʼƴ��
        # print("ret==", tmp_x_list + tmp_y_list + tmp_h_list + tmp_w_list + tmp_inbatch_score + tmp_inbatch_label)
        if num_proposals > 1024:
            num_proposals = 1024
            box_out = box_out + tmp_x_list[:1024] + tmp_y_list[:1024] + \
                      tmp_h_list[:1024] + tmp_w_list[:1024] + \
                      tmp_inbatch_score[:1024] + tmp_inbatch_label[:1024]
        else:
            box_out = box_out + tmp_x_list + tmp_y_list + tmp_h_list + \
                      tmp_w_list + tmp_inbatch_score + tmp_inbatch_label
        box_out_num = box_out_num + [num_proposals, 0, 0, 0, 0, 0, 0, 0]
    # print("box_out==", box_out)
    box_out = np.array(box_out)
    box_out_num = np.array(box_out_num)
    dumpData(box_out, "box_out.data", fmt="binary", data_type=dtype,
             path=BASE_DIR+fold)
    dumpData(box_out_num, "box_out_num.data", fmt="binary", data_type="int32",
             path=BASE_DIR+fold)

    print("box_out_num==", box_out_num)
    print("box_out==", box_out)
    fOutput = open("pygolden_result.txt", "w")
    # box = box_out.reshape((6, box_out_num[0]))
    # for i in range(0, box_out_num[0]):
    #     x1 = box[0, i]
    #     y1 = box[1, i]
    #     x2 = box[2, i]
    #     y2 = box[3, i]
    #     score = box[4, i]
    #     label = box[5, i]
    #     fOutput.write(
    #         str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + " " + str(score) + " " + " " + str(
    #             label) + " " + "\n")
    #
    # fOutput.close()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    gen_yolo_v3_data()























