class TopkHelp(object):
    def __init__(self, tik_instance):
        self.tik_inst = tik_instance
        self.region_size_inbyte = 16  # fp16 region size in byte
        self.region_size_infp16 = 8  # region size in term of fp16
        self.merge_sort_factor = 4  # we're doing 4-way merge sorting
        self.max_input_list_length = 4096
        self.min_ub_size = 32 * 1024
        self.ub_size = 248 * 1024  # UB size in byte

    def _div_sub_task(self, ub_buffer, regions_sorted, regions_orig, mem_interm, n_regions,
                      n_required, src_pos, dest_pos, task_id):
        # Divide the task
        div_factor = 4
        n_regions_subtsk = self._tik_topk_ceil_div(n_regions,
                                                   div_factor)  # number of input regions in subtsk
        n_regions_subtsk = self._tik_topk_ceil_div(n_regions_subtsk,
                                                   16) * 16  # align to 16
        n_required_subtsk = min(n_regions_subtsk,
                                n_required)  # number of required regions in subtsk
        n_remains = n_regions  # for dealing with tail part, because the last subtsk is
        subtsk_src_pos = src_pos  # for locating the beginning of each subtsk's input
        subtsk_dest_pos = dest_pos  # for locating the beginning of each subtsk's rslt

        subtsk_dest_pos_list = []  # save subtsk info for merging
        subtsk_n_required_list = []  # save substk info for merging

        level_from_leaf = 0
        for i in range(div_factor):
            subtsk_n_regions = min(n_regions_subtsk, n_remains)
            subtsk_n_required = min(n_required_subtsk, subtsk_n_regions)
            # solve the sub-task recursively
            level_from_leaf = self._tik_topk_recursive(ub_buffer, regions_sorted,
                                                       regions_orig, mem_interm,
                                                       subtsk_n_regions,
                                                       subtsk_n_required,
                                                       subtsk_src_pos,
                                                       subtsk_dest_pos,
                                                       task_id + (".%d" % i))
            if level_from_leaf == 0:
                subtsk_dest_pos_list.append(subtsk_src_pos)
            else:
                subtsk_dest_pos_list.append(subtsk_dest_pos)
            subtsk_n_required_list.append(subtsk_n_required)
            subtsk_src_pos += subtsk_n_regions
            subtsk_dest_pos += n_required_subtsk
            n_remains -= subtsk_n_regions
        if n_remains != 0:
            raise RuntimeError("After division of the problem, no regions should be left!")

        return level_from_leaf, task_id, subtsk_dest_pos_list, subtsk_n_required_list

    def _merge_sub_task(self, ub_buffer, regions_sorted, regions_orig, mem_interm, n_required,
                        subtsk_dest_pos_list, subtsk_n_required_list, dest_pos, level_from_leaf, task_id):
        # Merge sub-tasks
        if level_from_leaf % 2 == 0:
            merge_result = (regions_sorted if task_id == "0" else regions_orig)
            self._tik_topk_merge_subtsk(ub_buffer, merge_result, mem_interm,
                                        n_required, subtsk_dest_pos_list,
                                        subtsk_n_required_list,
                                        dest_pos, task_id)
        else:
            merge_result = (regions_sorted if task_id == "0" else mem_interm)
            self._tik_topk_merge_subtsk(ub_buffer, merge_result, regions_orig,
                                        n_required, subtsk_dest_pos_list,
                                        subtsk_n_required_list,
                                        dest_pos, task_id)
        level_from_leaf += 1
        return level_from_leaf

    def _tik_topk_recursive(self, ub_buffer, regions_sorted, regions_orig, mem_interm,
                            n_regions, n_required, src_pos, dest_pos, task_id):
        if n_regions < n_required:
            raise RuntimeError("n_regions: {} < n_required: {}".format(n_regions, n_required))

        if n_regions <= 16:
            raise RuntimeError("n_regions <= 16")

        level_from_leaf = 0
        if 16 < n_regions <= self.ub_size // 2 // self.region_size_inbyte:
            dest_pos = src_pos
            self._tik_topk_merge_sort(ub_buffer, mem_interm, regions_orig,
                                      [n_regions], n_required, [src_pos], dest_pos,
                                      task_id)
            level_from_leaf = 0
        else:
            # Divide the task
            level_from_leaf, task_id, subtsk_dest_pos_list, subtsk_n_required_list = \
                self._div_sub_task(ub_buffer, regions_sorted, regions_orig, mem_interm, n_regions,
                                   n_required, src_pos, dest_pos, task_id)
            level_from_leaf = self._merge_sub_task(ub_buffer, regions_sorted, regions_orig, mem_interm, n_required,
                                                   subtsk_dest_pos_list, subtsk_n_required_list, dest_pos,
                                                   level_from_leaf, task_id)
        return level_from_leaf

    def _tik_topk_merge_subtsk(self, ub_buffer, regions_sorted, regions_orig, n_required,
                               src_pos_list, src_region_num_list, dest_pos, task_id):
        """Act as the non-leaf node in divide and conqure tree.
           Do external merge sorting in this function."""

        if len(src_pos_list) != len(src_region_num_list):
            raise RuntimeError("POS_LIST.length != N_REGION.length")

        if sum(src_region_num_list) < n_required:
            raise RuntimeError("sum{ orig_region_num } < n_required")

        self._tik_topk_merge_sort_ext(ub_buffer, regions_sorted, regions_orig,
                                      src_region_num_list, n_required,
                                      src_pos_list, dest_pos, task_id)

    def _mov_out_to_ub(self, ub_buffer, n_input_list, src_region_num_list,
                       src_pos_ub, src_pos_list, regions_orig):
        offset = 0
        for i in range(n_input_list):
            if src_region_num_list[i] <= 4:
                raise RuntimeError("src_region_num_list <= 4")
            self.tik_inst.data_move(ub_buffer[0, src_pos_ub + offset, 0],
                                    regions_orig[0, src_pos_list[i], 0], sid=0,
                                    nburst=src_region_num_list[i] // 4,
                                    burst=self._tik_topk_ceil_div(self.region_size_inbyte * 4, 32),
                                    src_stride=0,
                                    dst_stride=0)
            offset += src_region_num_list[i]

    def _init_region_info_list(self, region_info_list, n_input_list, src_pos_list, src_region_num_list):
        if n_input_list > 1:
            for i in range(n_input_list):
                region_info_list.append(
                    {"offset": src_pos_list[i], "length": src_region_num_list[i],
                     "repeat": 1})
        else:
            region_info_list.append(
                {"offset": 0, "length": 1, "repeat": src_region_num_list[0]})

    def do_vbs(self, n_total_regions, region_info_list, ub_buffer, dest_pos_ub, src_pos_ub):
        if n_total_regions % 16 != 0:
            raise RuntimeError("n_total_regions should be multiple of 16")
        region_info_list = region_info_list[1:]
        offset = 0
        n_repeat_total = n_total_regions // 16
        while n_repeat_total > 0:
            n_repeat = min(n_repeat_total, 255)
            self.tik_inst.vrpsort16(dst=ub_buffer[0, dest_pos_ub + offset, 0],
                                    src=ub_buffer[0, src_pos_ub + offset, 0],
                                    repeat_times=n_repeat)
            offset += 16 * n_repeat
            n_repeat_total -= n_repeat
        if offset != n_total_regions:
            raise RuntimeError("offset != n_total_regions")

        region_info = {"offset": 0, "length": 16,
                       "repeat": n_total_regions // 16}
        region_info_list.append(region_info)
        return region_info_list

    def _do_vms(self, n_vms4, region_info_list, ub_buffer, dest_pos_ub, src_pos_ub):
        n_vms4 += 1
        new_region_info_list = []
        while len(region_info_list) > 0:
            region_info = region_info_list[0]
            if region_info["repeat"] <= 0:
                raise RuntimeError("repeat <= 0")

            if region_info_list[0]["repeat"] >= self.merge_sort_factor and \
                    region_info_list[0][
                        "length"] * self.merge_sort_factor < self.max_input_list_length:
                region_info_list = self._tik_topk_ms_with_repeat(ub_buffer,
                                                                 dest_pos_ub,
                                                                 src_pos_ub,
                                                                 region_info_list,
                                                                 new_region_info_list)
                continue

            if region_info_list[0]["repeat"] >= self.merge_sort_factor and \
                    region_info_list[0][
                        "length"] * self.merge_sort_factor >= self.max_input_list_length:
                region_info_list = self._tik_topk_ms_downgrade(ub_buffer,
                                                               dest_pos_ub,
                                                               src_pos_ub,
                                                               region_info_list,
                                                               new_region_info_list)
                continue
            region_info_list = self._tik_topk_ms_across_area(ub_buffer,
                                                             dest_pos_ub,
                                                             src_pos_ub,
                                                             region_info_list,
                                                             new_region_info_list)
        return new_region_info_list

    def _merge_sort(self, region_info_list, n_total_regions, ub_buffer, dest_pos_ub, src_pos_ub):
        n_vms4 = 0
        while True:
            if region_info_list[0]["length"] == 1:
                region_info_list = self.do_vbs(n_total_regions, region_info_list, ub_buffer, dest_pos_ub, src_pos_ub)
            else:
                # Do vms4 with repeat here
                new_region_info_list = self._do_vms(n_vms4, region_info_list, ub_buffer, dest_pos_ub, src_pos_ub)
                region_info_list = new_region_info_list
            if len(region_info_list) == 1 and region_info_list[0]["repeat"] == 1:
                break
            else:
                # Swap ub_buffer input and result and enlarge granuality
                temp = src_pos_ub
                src_pos_ub = dest_pos_ub
                dest_pos_ub = temp
        return ub_buffer, dest_pos_ub

    def _tik_topk_merge_sort(self, ub_buffer, regions_sorted, regions_orig,
                             src_region_num_list, n_required, src_pos_list, dest_pos,
                             task_id):
        if len(src_region_num_list) != len(src_pos_list):
            raise RuntimeError("Region num list and region pos list should be of the same length")
        if len(src_region_num_list) < 1 or len(src_region_num_list) > 4:
            raise RuntimeError("len(src_region_num_list) < 1 or len(src_region_num_list) > 4")

        n_input_list = len(src_region_num_list)
        n_total_regions = sum(src_region_num_list)

        src_pos_ub = 0
        dest_pos_ub = self.ub_size // 2 // self.region_size_inbyte if n_input_list == 1 else sum(
            src_region_num_list)

        # 1. Move data from OUT to UB
        self._mov_out_to_ub(ub_buffer, n_input_list, src_region_num_list, src_pos_ub, src_pos_list, regions_orig)
        # 2. Do on-chip merge sort
        region_info_list = []
        self._init_region_info_list(region_info_list, n_input_list, src_pos_list, src_region_num_list)

        ub_buffer, dest_pos_ub = self._merge_sort(region_info_list, n_total_regions, ub_buffer, dest_pos_ub, src_pos_ub)

        # 3. Move Data from UB to OUT
        required_tmp = min(n_required, regions_sorted.shape[1] - dest_pos)
        self.tik_inst.data_move(regions_sorted[0, dest_pos, 0], ub_buffer[0, dest_pos_ub, 0],
                                sid=0,
                                nburst=required_tmp // 4,
                                burst=self._tik_topk_ceil_div(self.region_size_inbyte * 4, 32),
                                src_stride=0,
                                dst_stride=0)

    def _tik_topk_ms_with_repeat(self, ub_buffer, dest_pos_ub, src_pos_ub, region_info_list,
                                 new_region_info_list):
        region_info = region_info_list[0]
        if region_info["repeat"] <= 0:
            raise RuntimeError("repeat <= 0")

        n_repeat = (region_info["repeat"] // self.merge_sort_factor)
        n_remainder = (region_info_list[0]["repeat"] % self.merge_sort_factor)

        offset = region_info["offset"]
        dst = ub_buffer[0, dest_pos_ub + offset, 0]
        src_list = []
        src_list_lengths = []

        for i in range(self.merge_sort_factor):
            src_list.append(ub_buffer[0, src_pos_ub + offset, 0])
            src_list_lengths.append(region_info["length"])
            offset += region_info["length"]

        self.tik_inst.vmrgsort4(dst, src_list, src_list_lengths,
                                if_exhausted_suspension=False,
                                valid_bit="1111", repeat_times=n_repeat)

        new_region_info_list.append(
            {"offset": region_info["offset"],
             "length": region_info["length"] * self.merge_sort_factor,
             "repeat": n_repeat})

        if n_remainder > 0:
            region_info_list[0]["offset"] += region_info[
                                                 "length"] * self.merge_sort_factor * n_repeat
            region_info_list[0]["repeat"] = n_remainder
        else:
            region_info_list = region_info_list[1:]
        return region_info_list

    def _get_merge_sort_factor(self, region_info):
        merge_sort_factor = self.merge_sort_factor
        while region_info["length"] * merge_sort_factor >= self.max_input_list_length:
            merge_sort_factor -= 1
        if merge_sort_factor < 2:
            raise RuntimeError("merge_sort_factor < 2")
        return merge_sort_factor

    def _get_vms4_info(self, merge_sort_factor, region_info, ub_buffer, src_pos_ub):
        offset = region_info["offset"]
        src_list = [ub_buffer[0, 0, 0] for i in range(self.merge_sort_factor)]
        src_list_lengths = [0 for i in range(self.merge_sort_factor)]
        valid_bit = 0

        for i in range(merge_sort_factor):
            src_list[i] = ub_buffer[0, src_pos_ub + offset, 0]
            src_list_lengths[i] = region_info["length"]
            offset += region_info["length"]
            valid_bit += 2 ** i
        return src_list, src_list_lengths, valid_bit

    def _tik_topk_ms_downgrade(self, ub_buffer, dest_pos_ub, src_pos_ub,
                               region_info_list, new_region_info_list):
        region_info = region_info_list[0]
        if region_info["repeat"] <= 0 or region_info["repeat"] < self.merge_sort_factor \
                or region_info["length"] >= self.max_input_list_length \
                or region_info["length"] * self.merge_sort_factor < self.max_input_list_length:
            raise RuntimeError("region info illegal!")

        merge_sort_factor = self._get_merge_sort_factor(region_info)

        dst = ub_buffer[0, dest_pos_ub + region_info["offset"], 0]
        src_list, src_list_lengths, valid_bit = self._get_vms4_info(merge_sort_factor, region_info, ub_buffer,
                                                                    src_pos_ub)

        self.tik_inst.vmrgsort4(dst, src_list, src_list_lengths,
                                if_exhausted_suspension=False,
                                valid_bit=valid_bit, repeat_times=1)
        new_region_info_list.append(
            {"offset": region_info["offset"],
             "length": region_info["length"] * merge_sort_factor,
             "repeat": 1})

        if region_info_list[0]["repeat"] <= merge_sort_factor:
            raise RuntimeError("repeat <= merge_sort_factor")

        region_info_list[0]["offset"] += region_info["length"] * merge_sort_factor
        region_info_list[0]["repeat"] -= merge_sort_factor

        return region_info_list

    def _tik_topk_ms_across_area(self, ub_buffer, dest_pos_ub, src_pos_ub,
                                 region_info_list, new_region_info_list):
        region_info = region_info_list[0]

        if region_info["repeat"] <= 0 or region_info["repeat"] >= self.merge_sort_factor:
            raise RuntimeError("repeat <= 0 or repeat >= self.merge_sort_factor")
        if region_info["length"] >= self.max_input_list_length:
            raise RuntimeError("length >= self.max_input_list_length")

        new_region_info = {"offset": region_info["offset"],
                           "length": 0,
                           "repeat": 1}
        offset = region_info["offset"]
        dst = ub_buffer[0, dest_pos_ub + offset, 0]
        src_list = [ub_buffer[0, 0, 0] for i in range(self.merge_sort_factor)]
        src_list_lengths = [0 for i in range(self.merge_sort_factor)]
        valid_bit = 0
        for i in range(self.merge_sort_factor):
            if len(region_info_list) == 0:
                break

            region_info = region_info_list[0]
            src_list[i] = ub_buffer[0, src_pos_ub + region_info["offset"], 0]
            src_list_lengths[i] = region_info["length"]
            valid_bit += 2 ** i

            new_region_info["length"] += region_info["length"]
            region_info_list[0]["repeat"] -= 1
            region_info_list[0]["offset"] += region_info["length"]
            if region_info_list[0]["repeat"] == 0:
                region_info_list = region_info_list[1:]

        if valid_bit > 1:
            self.tik_inst.vmrgsort4(dst=dst,
                                    src_list=src_list,
                                    element_lengths=src_list_lengths,
                                    if_exhausted_suspension=False,
                                    valid_bit=valid_bit,
                                    repeat_times=1)
        else:
            self.tik_inst.data_move(dst, src_list[0], sid=0,
                                    nburst=src_list_lengths[0] // 4,
                                    burst=self._tik_topk_ceil_div(4 * self.region_size_inbyte, 32),
                                    src_stride=0,
                                    dst_stride=0)

        new_region_info_list.append(new_region_info)
        return region_info_list

    def _fill(self, slot_idx, list_idx, variable_temp_, slot_capacity, src_list_rem_, ms_src_list_len_,
              n_burst_, ms_valid_bit_, ms_src_list, regions_orig, src_pos_):
        with self.tik_inst.if_scope(slot_idx == variable_temp_):
            self._tik_topk_min(slot_capacity,
                               src_list_rem_[list_idx],
                               ms_src_list_len_[slot_idx])
            with self.tik_inst.if_scope(
                    (ms_src_list_len_[slot_idx] % 4) > 0):
                n_burst_.set_as(
                    ms_src_list_len_[slot_idx] // 4 + 1)
            with self.tik_inst.else_scope():
                n_burst_.set_as(ms_src_list_len_[slot_idx] // 4)
            # record how many ms_src_list valid
            ms_valid_bit_.set_as(ms_valid_bit_ + 2 ** slot_idx)

            self.tik_inst.data_move(ms_src_list[slot_idx],
                                    regions_orig[
                                        0, src_pos_[list_idx], 0],
                                    sid=0,
                                    nburst=n_burst_,
                                    burst=self._tik_topk_ceil_div(
                                        self.region_size_inbyte * 4, 32),
                                    src_stride=0, dst_stride=0)

    def _fill_to_ub(self, n_src_list, src_list_rem_, list_slot_map_, variable_temp_, slot_capacity,
                    ms_src_list_len_, n_burst_, ms_valid_bit_, ms_src_list, regions_orig, src_pos_):
        for list_idx in range(n_src_list):
            with self.tik_inst.if_scope(src_list_rem_[list_idx] > 0):
                list_slot_map_[list_idx].set_as(variable_temp_)
                for slot_idx in range(n_src_list):
                    self._fill(slot_idx, list_idx, variable_temp_, slot_capacity, src_list_rem_, ms_src_list_len_,
                               n_burst_, ms_valid_bit_, ms_src_list, regions_orig, src_pos_)
                variable_temp_.set_as(variable_temp_ + 1)

    def _update_src_pos_list_rem(self, n_src_list, slot_idx, list_slot_map_, num_exhausted_, src_pos_, src_list_rem_):
        for list_idx in range(n_src_list):
            with self.tik_inst.if_scope(
                    list_slot_map_[list_idx] == slot_idx):
                src_pos_[list_idx].set_as(
                    src_pos_[list_idx] + num_exhausted_[
                        slot_idx])
                src_list_rem_[list_idx].set_as(
                    src_list_rem_[list_idx] - num_exhausted_[
                        slot_idx])

    def _merge_vms4_set_select(self, n_src_list, ms_valid_bit_, n_selected_, num_exhausted_, list_slot_map_,
                               src_pos_, src_list_rem_):
        for slot_idx in range(n_src_list):
            # make sure every list valid
            with self.tik_inst.if_scope(ms_valid_bit_ & (0x01 << slot_idx)):
                n_selected_.set_as(
                    n_selected_ + num_exhausted_[slot_idx])
                self._update_src_pos_list_rem(n_src_list, slot_idx, list_slot_map_, num_exhausted_, src_pos_,
                                              src_list_rem_)

    def _merge_vms4_post_update(self, n_total_selected_, n_selected_, n_total_rem_):
        n_total_selected_.set_as(n_total_selected_ + n_selected_)

        with self.tik_inst.if_scope(n_total_rem_ > n_selected_):
            n_total_rem_.set_as(n_total_rem_ - n_selected_)
        with self.tik_inst.else_scope():
            n_total_rem_.set_as(0)

    def _merge_vms4(self, ms_valid_bit_, ms_dest, ms_src_list, ms_src_list_len_, n_src_list, num_exhausted_,
                    n_selected_, list_slot_map_, src_pos_, src_list_rem_, n_total_rem_, n_burst_, dest_pos,
                    n_total_selected_, regions_sorted, dest_pos_):
        with self.tik_inst.if_scope(ms_valid_bit_ > 0):
            self.tik_inst.vmrgsort4(ms_dest, ms_src_list,
                                    element_lengths=ms_src_list_len_,
                                    if_exhausted_suspension=False,
                                    valid_bit=ms_valid_bit_,
                                    repeat_times=1)
            for i in range(n_src_list):
                num_exhausted_[i].set_as(ms_src_list_len_[i])

            n_selected_.set_as(0)
            self._merge_vms4_set_select(n_src_list, ms_valid_bit_, n_selected_, num_exhausted_, list_slot_map_,
                                        src_pos_, src_list_rem_)
            # the last time vmrgsort is called, n_toal_selected_ may larger than
            # n_reuird. So the following correction to `n_selected' is performed to
            # avoid invalid data write to `regions_sorted'.
            self._tik_topk_min(n_selected_, n_total_rem_, n_selected_)
            n_burst_.set_as(n_selected_ / 4)
            with self.tik_inst.if_scope((n_selected_ & 0x3) > 0):
                n_burst_.set_as(n_burst_ + 1)
            dest_pos_.set_as(dest_pos + n_total_selected_)
            self.tik_inst.data_move(regions_sorted[0, dest_pos_, 0], ms_dest, 0,
                                    nburst=n_burst_,
                                    burst=self._tik_topk_ceil_div(
                                        self.region_size_inbyte * 4, 32),
                                    src_stride=0,
                                    dst_stride=0)
            # Step-4: Do post update
            self._merge_vms4_post_update(n_total_selected_, n_selected_, n_total_rem_)

    def _tik_topk_merge_sort_ext(self, ub_buffer, regions_sorted, regions_orig,
                                 src_region_num_list, n_required, src_pos_list,
                                 dest_pos, task_id):
        """Act as the leaf node in divide-and-conqure tree"""
        n_src_list = len(src_pos_list)
        slot_capacity = self.ub_size // (n_src_list * 2) // self.region_size_inbyte

        n_total_selected_ = self.tik_inst.Scalar(
            dtype="int64")  # total number of sorted regions
        n_total_rem_ = self.tik_inst.Scalar(
            dtype="int64")  # total number of unsorted regions compared to 'n_required'
        n_selected_ = self.tik_inst.Scalar(
            dtype="int64")  # number of sorted regions in each iteration
        vms4_flag_ = self.tik_inst.Scalar(dtype="int64")
        scalar_slot_capacity_ = self.tik_inst.Scalar(dtype="int64")
        src_pos_ = [self.tik_inst.Scalar(dtype="int64") for i in range(n_src_list)]
        src_list_rem_ = [self.tik_inst.Scalar(dtype="int64") for i in range(n_src_list)]
        list_slot_map_ = [self.tik_inst.Scalar(dtype="int64") for i in
                          range(n_src_list)]  # map from list_idx to slot_idx
        n_burst_ = self.tik_inst.Scalar(dtype="int64")
        variable_temp_ = self.tik_inst.Scalar(dtype="int64")
        dest_pos_ = self.tik_inst.Scalar(dtype="int64")
        num_exhausted_ = [self.tik_inst.Scalar(dtype="int64") for i in range(n_src_list)]

        n_total_selected_.set_as(0)
        n_total_rem_.set_as(n_required)
        vms4_flag_.set_as(0)
        scalar_slot_capacity_.set_as(slot_capacity)
        dest_pos_.set_as(dest_pos)
        for i in range(n_src_list):
            src_pos_[i].set_as(src_pos_list[i])
            src_list_rem_[i].set_as(src_region_num_list[i])

        ms_dest = ub_buffer[0, self.ub_size // self.region_size_inbyte // 2, 0]
        ms_src_list = [ub_buffer[0, 0, 0] for i in
                       range(self.merge_sort_factor)]  # [0, 0, 0, 0]
        ms_src_list_len_ = [self.tik_inst.Scalar(dtype="int64") for i in
                            range(
                                self.merge_sort_factor)]  # [scalar, scalar, scalar, scalar]
        for slot_idx in range(n_src_list):
            ms_src_list[slot_idx] = ub_buffer[0, slot_capacity * slot_idx, 0]
            ms_src_list_len_[slot_idx].set_as(0)
        ms_valid_bit_ = self.tik_inst.Scalar(dtype="int64")

        min_input_length = min(src_region_num_list)
        max_iteration = self._tik_topk_ceil_div(n_required,
                                                min(slot_capacity, min_input_length))
        with self.tik_inst.for_range(0, max_iteration):
            with self.tik_inst.if_scope(n_total_selected_ < n_required):
                # Step-1: Fullfill all the inputs slots on UB
                ms_valid_bit_.set_as(0)
                variable_temp_.set_as(0)
                self._fill_to_ub(n_src_list, src_list_rem_, list_slot_map_, variable_temp_, slot_capacity,
                                 ms_src_list_len_, n_burst_, ms_valid_bit_, ms_src_list, regions_orig, src_pos_)
                self._merge_vms4(ms_valid_bit_, ms_dest, ms_src_list, ms_src_list_len_, n_src_list, num_exhausted_,
                                 n_selected_, list_slot_map_, src_pos_, src_list_rem_, n_total_rem_, n_burst_, dest_pos,
                                 n_total_selected_, regions_sorted, dest_pos_)

    def _tik_topk_ceil_div(self, value, factor):
        if factor == 0:
            raise RuntimeError("Doing ceil division with divider equals to 0")
        return (value + (factor - 1)) // factor

    def _tik_topk_min(self, first_num, second_num, result_):
        with self.tik_inst.if_scope(first_num < second_num):
            result_.set_as(first_num)
        with self.tik_inst.else_scope():
            result_.set_as(second_num)

    def _param_check(self, proposals, n_proposals, proposals_sorted, n_required, mem_intermediate, mem_ub):
        if n_proposals % 16 != 0:
            raise RuntimeError("input proposals should be multiple of 16")
        proposals_shape = proposals.shape
        if len(proposals_shape) != 3 or proposals_shape[1] != n_proposals:
            raise RuntimeError("input proposals shape {} is illegal! n_proposals equal to {}"
                               .format(proposals_shape, n_proposals))

        proposals_sorted_shape = proposals_sorted.shape
        if len(proposals_sorted_shape) != 3 or proposals_sorted_shape[1] < n_required:
            raise RuntimeError("input proposals_sorted shape {} is illegal!".format(proposals_sorted_shape))

        mem_intermediate_shape = mem_intermediate.shape
        if len(mem_intermediate_shape) != 3 or mem_intermediate_shape[1] < n_proposals:
            raise RuntimeError("input mem_intermediate shape is illegal!")

        mem_ub_shape = mem_ub.shape
        if len(mem_ub_shape) != 3:
            raise RuntimeError("input mem_ub shape is illegal!")

        ub_size = mem_ub_shape[0] * mem_ub_shape[1] * mem_ub_shape[2] * 2
        if ub_size < self.min_ub_size:
            raise RuntimeError("mem_ub buffer size should not less than 32K!, mem_ub size: {}".format(ub_size))

    def tik_topk(self, proposals_sorted, proposals, n_proposals, n_required, mem_ub, mem_intermediate):
        """
        Select top K element from last dimension
        Parameter
        :param proposals_sorted: output tensor
            Shape(1, n_proposals, 8), type float16, scope ub l1 or gm
        :param proposals:
            Shape(1, n_required, 8), type float16, scope ub l1 or gm
        :param n_proposals:
            ori proposals num
        :param n_required:
            Number of largest elements to be select
        :param mem_ub:
            Shape (1, ub_size // 16, 8), type float16, scope ub_buffer
        :param mem_intermediate:
            Shape (1, >=n_proposals, 8), type float16, scope ub l1 or gm.
        :return:
        None
        """
        self._param_check(proposals, n_proposals, proposals_sorted, n_required, mem_intermediate, mem_ub)
        ub_size = mem_ub.shape[0] * mem_ub.shape[1] * mem_ub.shape[2]
        self.ub_size = ub_size * 2

        if self.ub_size // 2 // self.region_size_inbyte >= n_proposals > 16:
            self._tik_topk_merge_sort(mem_ub, proposals_sorted, proposals,
                                      [n_proposals], n_required, src_pos_list=[0, ], dest_pos=0,
                                      task_id="0")
        else:
            self._tik_topk_recursive(mem_ub, proposals_sorted,
                                     proposals, mem_intermediate,
                                     n_proposals, n_required, src_pos=0,
                                     dest_pos=0, task_id="0")
