
import caffe
import numpy as np
import tensorflow as tf

import caffe_models
import tf_models

VRB = False


def log(s):
    if VRB:
        print(s)


def get_tensors_by_name(graph, tensor_names):
    return \
        [graph.get_tensor_by_name(tensor_name) for tensor_name in tensor_names]


# The input_columns format expects a batch (size B) of N "columns". This is
# provided as a length N list of B-lengthed list containing entries of each
# column.
class ModelRunner(object):
    def execute(self, input_columns):
        pass


class CaffeModelRunner(ModelRunner):
    def __init__(self, model_name, batch_size, model_dict=None, gpu_id=0):
        # TF session for optional TF evaluation (e.g. fast GPU resizing).
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        if model_dict is not None:
            self.model_dict = model_dict
        else:
            self.model_dict = caffe_models.caffe_get_model_fn(
                model_name, batch_size=self.batch_size)

        caffe.set_mode_gpu()
        caffe.set_device(self.gpu_id)
        log("Caffe GPU id: {}".format(self.gpu_id))

        self.image_width, self.image_height = self.model_dict['input_dims']
        self.model_path = self.model_dict['model_prototxt_path']
        self.weights_path = self.model_dict['model_weights_path']

        self.model = caffe.Net(self.model_path, self.weights_path, caffe.TEST)
        self.model.blobs['data'].reshape(
            self.batch_size, 3, self.image_height, self.image_width)

    def execute(self, input_columns):
        log("Enter CMR execute")
        caffe.set_mode_gpu()
        caffe.set_device(self.gpu_id)
        log("Caffe GPU id: {}".format(self.gpu_id))

        inputs = \
            self.model_dict['input_preprocess_fn'](self.sess, input_columns)
        outputs = self.model_dict['inference_fn'](self.model, inputs)
        post_processed_outputs = self.model_dict['post_processing_fn'](
            input_columns, outputs)
        log("Exit CMR execute")
        return post_processed_outputs


class TensorflowModelRunner(ModelRunner):
    def __init__(self, model_name, batch_size, model_dict=None, gpus=None):
        self.batch_size = batch_size
        if model_dict is not None:
            self.model_dict = model_dict
        else:
            self.model_dict = tf_models.tf_get_model_fn(
                model_name, batch_size=self.batch_size)

        mode = self.model_dict['mode']
        checkpoint_path = self.model_dict['checkpoint_path']
        input_tensors = self.model_dict['input_tensors']
        output_tensors = self.model_dict['output_tensors']
        if gpus is not None:
            gpu_options = tf.GPUOptions(visible_device_list=gpus)
            sess_config = tf.ConfigProto(log_device_placement=True,
                                         gpu_options=gpu_options)
        else:
            sess_config = tf.ConfigProto(log_device_placement=True)

        # NOTE: Batching is only allowed in python mode because frozen graphs
        #       cannot be modified.

        if mode == 'frozen_graph':
            tf_graph = tf.Graph()
            with tf_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=tf_graph, config=sess_config)
        elif mode == 'python':
            self.sess = tf.Session(config=sess_config)
            self.model_dict['model_init_fn']()
            saver = tf.train.Saver()
            saver.restore(self.sess, checkpoint_path)
            tf_graph = self.sess.graph
        elif mode == 'keras':
            self.sess = tf.Session(config=sess_config)
            from keras import backend as K
            K.set_session(self.sess)
            self.model_dict['model_init_fn'](K)
            tf_graph = self.sess.graph
        else:
            raise Exception("Invalid model loading mode: %s" % mode)

        self.input_tensors = get_tensors_by_name(tf_graph, input_tensors)
        self.output_tensors = get_tensors_by_name(tf_graph, output_tensors)

    def execute(self, input_columns):
        feed_dict = self.model_dict['session_feed_dict_fn'](
            self.sess, self.input_tensors, input_columns)
        outputs = self.sess.run(self.output_tensors, feed_dict)
        # run_metadata = tf.RunMetadata()
        # outputs = self.sess.run(self.output_tensors, feed_dict,
        #                         options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #                         run_metadata=run_metadata)
        # tf.profiler.profile(
        #     self.sess.graph,
        #     run_meta=run_metadata,
        #     cmd='op',
        #     options=tf.profiler.ProfileOptionBuilder.time_and_memory())
        post_processed_outputs = \
            self.model_dict['post_processing_fn'](input_columns, outputs)
        return post_processed_outputs
