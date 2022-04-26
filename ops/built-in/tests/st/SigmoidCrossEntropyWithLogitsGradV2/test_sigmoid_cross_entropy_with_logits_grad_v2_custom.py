from impl.dynamic.sigmoid_cross_entropy_with_logits_grad_v2 import get_cof_by_shape
def reload_check_support():
	"""
	reload_check_support to improve cov
	"""
	get_cof_by_shape([16, 16, 16, 16], "float32")

if __name__ == '__main__':
    reload_check_support()
