include $(BUILD_SYSTEM)/base_rules.mk

$(LOCAL_BUILT_MODULE):
	@mkdir -p $(dir $@)
	@./cann/ops/built-in/tbe/scripts/compile_vector_core_tbe_info.sh $(product)
	@echo $@
	@echo $(HOST_OUT_INTERMEDIATES)/$(notdir $@)
	@cp $(HOST_OUT_INTERMEDIATES)/$(notdir $@) $@
	@echo $(HOST_OUT_INTERMEDIATES)
