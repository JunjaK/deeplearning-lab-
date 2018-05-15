import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('height', 192, '')
tf.app.flags.DEFINE_integer('width', 192, '')
tf.app.flags.DEFINE_integer('depth', 3, '')
tf.app.flags.DEFINE_integer('num_class', 2, '')
tf.app.flags.DEFINE_integer('num_conv', 6, '')
tf.app.flags.DEFINE_integer('kernel_size', 5, '')
tf.app.flags.DEFINE_integer('pool_size', 2, '')
tf.app.flags.DEFINE_integer('num_map', 16, '')
tf.app.flags.DEFINE_integer('num_fc_layer', 4, '')
tf.app.flags.DEFINE_integer('num_fc_input', 768, '')
tf.app.flags.DEFINE_integer('max_steps', 40000, '')
tf.app.flags.DEFINE_integer('eval_steps', 1000, '')
tf.app.flags.DEFINE_integer('batch_size', 64, '')
tf.app.flags.DEFINE_integer('capacity', 4000, '')
tf.app.flags.DEFINE_float('learning_rate', 0.001, '')

tf.app.flags.DEFINE_string('train_dir', './CNN_Homework/data/train/', '')
tf.app.flags.DEFINE_string('eval_dir', './CNN_Homework/data/eval/', '')
tf.app.flags.DEFINE_string('log_dir', './CNN_Homework/logs/', '')
