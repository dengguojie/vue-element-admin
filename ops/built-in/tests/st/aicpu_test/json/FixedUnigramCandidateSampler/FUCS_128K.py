import random

def fuzz_branch():
    data_num = 16 * 1024

    # input
    true_classes_value = [[x for x in range(0, data_num)]]

    # attr
    num_true_value = data_num
    num_sampled_value = 3
    range_max_value = data_num
    vocab_file_value = ""
    distortion_value = 1.0
    num_reserved_ids_value = 0
    num_shards_value = 1
    shard_value = 0
    unigrams_value = [random.random() for _ in range(data_num)]
    seed_value = 87654321
    seed2_value = 0

    return {
        "input_desc": {
            "true_classes": {"value": true_classes_value, "shape": [1, len(true_classes_value[0])]},
        },
        "attr": {
            "num_true": {"value": num_true_value},
            "num_sampled": {"value": num_sampled_value},
            "range_max": {"value": range_max_value},
            "vocab_file": {"value": vocab_file_value},
            "distortion": {"value": distortion_value},
            "num_reserved_ids": {"value": num_reserved_ids_value},
            "num_shards": {"value": num_shards_value},
            "shard": {"value": shard_value},
            "unigrams": {"value": unigrams_value},
            "seed": {"value": seed_value},
            "seed2": {"value": seed2_value}
        }
    }


fuzz_branch()