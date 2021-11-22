import tensorflow as tf

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    records = tf.compat.v1.placeholder(dtype="string", shape=(1,))
    record_defaults = tf.compat.v1.placeholder(dtype="string", shape=(1,))
    tf.io.decode_csv(
        records, [record_defaults], field_delim=',', use_quote_delim=True,
        na_value=',', select_cols=None, name='decodecsv'
    )
    tf.io.write_graph(sess.graph, logdir="./", name="decode_csv_case_1.pb", as_text=False)