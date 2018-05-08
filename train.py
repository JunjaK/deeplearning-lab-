import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import input_data
import config
import model

FLAGS = tf.app.flags.FLAGS



def run_training():
    train, train_label = input_data.get_files(FLAGS.train_dir)
    train_batch, train_label_batch = input_data.get_batch(train, train_label, FLAGS.height, FLAGS.width, FLAGS.batch_size, FLAGS.capacity)

    
    keep_prob = tf.placeholder(tf.float32)
    tf.reset_default_graph()

    hypothesis, cross_entropy, train_step = model.make_network(train_batch, train_label_batch, keep_prob)

    cost_sum = tf.summary.scalar("cost", cross_entropy)
    
    train_acc = model.evaluation(hypothesis, train_label_batch)

    with tf.Session() as sess:
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_dir, sess.graph)
        saver = tf.train.Saver()
    
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                _,tra_loss, tra_acc = sess.run([train_step, train_loss, train_acc], feed_dict={keep_prob: 0.7})
            
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    run_training()
