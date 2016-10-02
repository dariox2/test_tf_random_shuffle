
#
# test shuffle_batch w/2 file queues
#
# example with pipeline returning batch with pairs
# of matching files (e.g. color + black/white),
# using a seed for repeating the random sequence
#
# v.9bt - idem 9at, text. but printing at the end
#

import os
import numpy as np
print("Loading tensorflow...")
import tensorflow as tf


import datetime

tf.set_random_seed(1)


def create_input_pipeline_2(files1, files2, batch_size, shape, 
  crop_shape=None, crop_factor=1.0, n_threads=1, seed=None):

    producer1 = tf.train.string_input_producer(
        files1, capacity=len(files1), shuffle=False)
    producer2 = tf.train.string_input_producer(
        files2, capacity=len(files2), shuffle=False)

    # We need something which can open the files and read its contents.
    reader1 = tf.WholeFileReader()
    reader2 = tf.WholeFileReader()

    # We pass the filenames to this object which can read the file's contents.
    # This will create another queue running which dequeues the previous queue.
    keys1, vals1 = reader1.read(producer1)
    keys2, vals2 = reader2.read(producer2)

    # And then have to decode its contents as we know it is a jpeg image
    imgs1 = tf.image.decode_jpeg(vals1, channels=3)
    imgs2 = tf.image.decode_jpeg(vals2, channels=3)

    # We have to explicitly define the shape of the tensor.
    # This is because the decode_jpeg operation is still a node in the graph
    # and doesn't yet know the shape of the image.  Future operations however
    # need explicit knowledge of the image's shape in order to be created.
    imgs1.set_shape(shape)
    imgs2.set_shape(shape)

    # Next we'll centrally crop the image to the size of 100x100.
    # This operation required explicit knowledge of the image's shape.
    if shape[0] > shape[1]:
        rsz_shape = [int(shape[0] / shape[1] * crop_shape[0] / crop_factor),
                     int(crop_shape[1] / crop_factor)]
    else:
        rsz_shape = [int(crop_shape[0] / crop_factor),
                     int(shape[1] / shape[0] * crop_shape[1] / crop_factor)]

    rszs1 = tf.image.resize_images(imgs1, rsz_shape[0], rsz_shape[1])
    rszs2 = tf.image.resize_images(imgs2, rsz_shape[0], rsz_shape[1])

    crops1 = (tf.image.resize_image_with_crop_or_pad(
        rszs1, crop_shape[0], crop_shape[1])
        if crop_shape is not None
        else imgs1)
    crops2 = (tf.image.resize_image_with_crop_or_pad(
        rszs2, crop_shape[0], crop_shape[1])
        if crop_shape is not None
        else imgs2)

    min_after_dequeue = len(files1) // 5

    capacity = min_after_dequeue + (n_threads + 1) * batch_size

    batch = tf.train.shuffle_batch([crops1, crops2],
                                   enqueue_many=False,
                                   batch_size=batch_size,
                                   capacity=capacity,
                                   min_after_dequeue=min_after_dequeue,
                                   num_threads=n_threads,
                                   seed=seed)
    
    return batch



def get_some_files(path):
  fs = [os.path.join(path, f)
  for f in os.listdir(path) if f.endswith('.jpg')]
  fs=sorted(fs)
  return fs


print("Loading files...")
filesX = get_some_files("img_align_celeba/") # image set 1 (color)
filesY = get_some_files("img_align_celeba_n/") # matching set 2 (b/w)


batch_size = 8
input_shape = [218, 178, 3]
crop_shape = [64, 64, 3]
crop_factor = 0.8

#seed=15 # not really necessary 
seed=None

TID=datetime.date.today().strftime("%Y%m%d")+"_"+datetime.datetime.now().time().strftime("%H%M%S")

n_plots=2
n_bats=3

def runtest(sess, batch, idt):

  mntgs=[]

  for bat in range(n_bats):

    mntg=[]
    batres = sess.run(batch)
    batch_xs1=np.array(batres[0])
    batch_xs2=np.array(batres[1])
    for imn in range(batch_size):
      img1=batch_xs1[imn] / 255.0 # color image
      img2=batch_xs2[imn] / 255.0 # matching b/n image
      mntg.append(img1)
      mntg.append(img2)
      
    mntgs.append(mntg)

  return mntgs

batch = create_input_pipeline_2(
    files1=filesX, files2=filesY,
    batch_size=batch_size,
    crop_shape=crop_shape,
    crop_factor=crop_factor,
    shape=input_shape,
    seed=seed)

with tf.Session() as mysess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=mysess, coord=coord)

  try:
    amntgs=[]
    for p in range(n_plots):
      if coord.should_stop():
        break
      mntgs=runtest(mysess, batch, p)
      amntgs.append(mntgs)
  
  except (Exception, e):
    # Report exceptions to the coordinator.
    print("EXCEPTION", e)
    coord.request_stop(e)
  finally:
    # Terminate as usual.  It is innocuous to request stop twice.
    coord.request_stop()
    coord.join(threads)

  for p in range(n_plots):

    mntgs=amntgs[p]

    for b in range(n_bats):
    
      mntg=mntgs[b]
        
      for im in range(batch_size):
      
        img=mntg[im*2]
        
        # debug diff with text
        print("y",im,b,p,img[50][50:51])
        # img2 don't care for this purpose

# eop

