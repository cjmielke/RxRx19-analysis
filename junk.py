import tensorflow as tf
from keras import backend as K


# doesn't seem to work
# https://gist.github.com/codertimo/f26a4005dc8dce3e96c4f7e1e69f17e9

def word_acc(y_true, y_pred):
	a = K.argmax(y_pred, axis=-1)
	b = tf.reduce_max(y_true, axis=-1)
	b = tf.cast(b, tf.int64)

	from tensorflow.python import count_nonzero
	total_index = count_nonzero(a + b, 1)
	wrong_index = count_nonzero(a - b, 1)

	total_index = tf.reduce_sum(total_index)
	wrong_index = tf.reduce_sum(wrong_index)

	correct_index = K.cast(total_index - wrong_index, K.floatx())
	total_index = K.cast(total_index, K.floatx())

	acc = tf.divide(correct_index, total_index)

	return K.cast(acc, K.floatx())