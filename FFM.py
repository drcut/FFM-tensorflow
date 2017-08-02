import tensorflow as tf
configure = {
    'batch_size':64,
    'learning_rate':0.001
}
data_path = './norm_test_data.txt'
field_num = 10
feature_num = 10
feature_emb_size = 64
field_emb_size = 64
class FFM:
    def __init__(self):
        self.batch_size = configure['batch_size']
        self.lr = configure['learning_rate']
        self.data_path = data_path
        self.field_num = field_num
        self.feature_num = feature_num
        with tf.name_scope('embedding_matrix'):
            self.feature_embedding = tf.get_variable(name='feature_embedding',
                                                shape=[feature_num,feature_emb_size],
                                                dtype=tf.float32)
            self.field_embedding = []
            for idx in xrange(0,self.feature_num):
                self.field_embedding.append(tf.get_variable(name='field_embedding{}'.format(idx),
                                                            shape=[field_num,field_emb_size],
                                                            dtype=tf.float32))
        with tf.name_scope('input'):
            self.label = tf.placeholder(tf.float32, shape=(self.batch_size))
            self.feature_value = tf.placeholder(tf.float32,shape=(self.batch_size,feature_num))
if __name__ == "__main__":
