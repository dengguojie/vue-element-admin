import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx.backend.test.case.node.tfidfvectorizer import TfIdfVectorizerHelper
import numpy as np


def make_tfidf_vectorizer():
    X = helper.make_tensor_value_info("X", TensorProto.INT32, [2, 6])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 7])

    ngram_counts = np.array([0, 4]).astype(np.int64)
    ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
    pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                            5, 6, 7, 8, 6, 7]).astype(np.int64)   # bigrams

    node_helper = TfIdfVectorizerHelper(
        mode='TF',
        min_gram_length=2,
        max_gram_length=2,
        max_skip_count=0,
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_int64s=pool_int64s
    )
    node_def = node_helper.make_node_noweights()
    graph = helper.make_graph(
        [node_def],
        "test_cast",
        [X],
        [Y],
    )

    model = helper.make_model(graph, producer_name="onnx-cast_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_tfidf_vectorizer.onnx")


if __name__ == '__main__':
    make_tfidf_vectorizer()
