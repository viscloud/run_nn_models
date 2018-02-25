from model_runners import CaffeModelRunner, TensorflowModelRunner
import h5py
import numpy as np
import os
import skvideo.io
import sys

video_name = sys.argv[1]
model_name = sys.argv[2]
use_caffe = bool(sys.argv[3])
output_dir = sys.argv[4]
batch_size = 32
display_interval = 960

# Intialize readers and models.
vreader = skvideo.io.vreader(video_name)
num_frames = int(skvideo.io.ffprobe(video_name)['video']['@nb_frames'])
if use_caffe:
    model_runner = CaffeModelRunner(model_name, batch_size)
else:
    model_runner = TensorflowModelRunner(model_name, batch_size)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
hdf5_file = None

for i in range(0, num_frames, batch_size):
    start, end = i, min(i + batch_size, num_frames)
    if i % display_interval == 0:
        print(i, num_frames)
    inputs = np.array([next(vreader) for _ in range(end - start)])
    outputs = np.array([np.loads(v) for v in model_runner.execute([inputs])[0]])
    if hdf5_file is None:
        hdf5_file = h5py.File(output_dir + '/feature_vectors.npy', mode='w')
        hdf5_file.create_dataset(
            'data', [num_frames] + list(outputs.shape[1:]), 'f4')
    hdf5_file['data'][start:end] = outputs
