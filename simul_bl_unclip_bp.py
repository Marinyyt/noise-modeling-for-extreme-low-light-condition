# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import cv2

import math

from scipy import misc
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
input_dir = './Sony/short/test/'
gt_dir = './Sony/long/'
checkpoint_dir = './result_Sony_v6/instance_process/'
result_dir = './result_Sony_v6/instance_process/'
bl_dir = './Sony/SID_black level_v3/'
black = {}
res = {}
with open("./simul_bl_test.txt", 'r') as f:
    for line in f:
        black[line.split(' ')[0]] = float(line.split(' ')[2])

with open("./Sony_test_list.txt", 'r') as f:
    for line in f:
        res[int(line.split(' ')[0][13:18])] = line.split(' ')[2][3:]


# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def network(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out



def pack_raw(raw, bl):
    # pack Bayer image to 4 channels
    #im = raw.raw_image_visible.astype(np.float32)
    
    im = (raw - bl) / (16383 - bl)  # subtract the black level
    
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def normalization(img):
    mean = np.mean(img)
    var = np.var(img)
    img = (img - mean) / var
    return img



def psnr1(img1, img2):
   mse = np.mean((img1 - img2) ** 2 )
   if mse < 1.0e-10:
       return 100
   return 10 * math.log10(255.0**2/mse)

def cal_ssim(im1,im2):

   mu1 = im1.mean()
   mu2 = im2.mean()
   sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
   sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
   sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
   k1, k2, L = 0.01, 0.03, 255
   C1 = (k1*L) ** 2
   C2 = (k2*L) ** 2
   C3 = C2/2
   l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
   c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
   s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
   ssim = l12 * c12 * s12
   return ssim







batch_size = 4


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')
x = 0
batch_size = 4
psnr = []
ssim = []
indicator = 0
final = []
for test_id in test_ids:

    
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)

        frame_id = str(res[test_id])+'_'+str(in_fn[9:-5])+'.ARW'
        frame_path = glob.glob(bl_dir  +  frame_id)

        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)



        #frame_raw = rawpy.imread(frame_path[0])
        ##frame = frame_raw.raw_image_visible.astype(np.float32)
        #bl = np.mean(frame[500:2000][1000:3000])
        raw = rawpy.imread(in_path)
        input_image = raw.raw_image_visible.astype(np.float32)
        #for i in range(0, 2832, 250):
         #   for j in range(0, 4256, 250):
          #      input_image[i:i+250, j:j+250] = (input_image[i:i+250, j:j+250] - np.mean(frame[i:i+250, j:j+250])) / (16383 - np.mean(frame[i:i+250, j:j+250]))  
        asd = np.zeros((11, 1), dtype = np.float32)
        for k in range(11): 
            input_full = np.expand_dims(pack_raw(input_image, 512-k +5), axis=0) 
        
            input_full = np.minimum(input_full * ratio, 1.0)

       
            input_full = normalization(input_full)

            output = sess.run(out_image, feed_dict={in_image: input_full})
            output = np.minimum(np.maximum(output, 0), 1)
            output = output[0, :, :, :]
            asd[k,0] = np.mean(output[500:1000, 1500:2000, :])
        
        final.append(np.var(asd))

        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full * ratio, 1.0)

       
        input_full = normalization(input_full)

        output = sess.run(out_image, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0), 1)
        output = output[0, :, :, :]
        gt_full = gt_full[0, :, :, :]

        
b = 0.0
for x in final:
    b = b + x
print(b)

        #scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
        #scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))
