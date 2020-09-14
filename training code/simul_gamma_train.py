# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
input_dir = './Sony/long/'
gt_dir = './Sony/long/'
checkpoint_dir = './row/gamma_paratest3/'
result_dir = './row/gamma_paratest3/'
best_dir = './row/gamma_paratest3/best_ckpt/'
bl_dir = './SID_black level_v3/'

txt = []
res = {}

with open("./Sony_train_list.txt", 'r') as f:
    for line in f:
        res[int(line.split(' ')[0][13:18])] = line.split(' ')[2][3:]


long_exp = {}
with open('./noise_level_train.txt', 'r') as f:
   for line in f:
       long_exp[int(line.split(' ')[0][0:5])] = int(line.split(' ')[0][9:11])
f.close()
exp = {}
for id in long_exp:
   exp[id] = []
with open('./Sony_train_list.txt', 'r') as f:
   for line in f:
       if long_exp[int(line.split(' ')[0][13:18])] == 30 and (300 not in exp[int(line.split(' ')[0][13:18])]) :
           exp[int(line.split(' ')[0][13:18])].append(300)
       if long_exp[int(line.split(' ')[0][13:18])] == 10 and float(line.split(' ')[0][22:-5]) == 0.1 and (100 not in exp[int(line.split(' ')[0][13:18])]):
           exp[int(line.split(' ')[0][13:18])].append(100)
       if long_exp[int(line.split(' ')[0][13:18])] == 10 and float(line.split(' ')[0][22:-5]) == 0.033 and (300 not in exp[int(line.split(' ')[0][13:18])]):
           exp[int(line.split(' ')[0][13:18])].append(300)
       if long_exp[int(line.split(' ')[0][13:18])] == 10 and float(line.split(' ')[0][22:-5]) == 0.04 and (250 not in exp[int(line.split(' ')[0][13:18])]):
           exp[int(line.split(' ')[0][13:18])].append(250)

read_noise = {}
shot_noise = {}
with open('./noise_level_train.txt', 'r') as f:
    for line in f:
        shot_noise[int(line.split(' ')[0][0:5])] = float(line.split(' ')[1])
        read_noise[int(line.split(' ')[0][0:5])] = float(line.split(' ')[2])
# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512  # patch size for training
save_freq = 500

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


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

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def pack_black_raw(black_frame):
    # pack Bayer image to 4 channels
    im = black_frame.raw_image_visible.astype(np.float32)
    result = np.zeros_like(im)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    r = im[0:H:2, 0:W:2]
    g1 = im[0:H:2, 1:W:2]
    b = im[1:H:2, 1:W:2]
    g2 = im[1:H:2, 0:W:2]

    f1 = np.fft.fft2(r)
    f1[0, 0] = 0
    f1[:, 1:] = 0
    fshift1 = np.fft.fftshift(f1)
    ishift1 = np.fft.ifftshift(fshift1)
    iimg1 = np.fft.ifft2(ishift1)
    iimg1 = np.abs(iimg1)
    result[0:H:2, 0:W:2] = iimg1

    f2 = np.fft.fft2(g1)
    f2[0, 0] = 0
    f2[:, 1:] = 0
    fshift2 = np.fft.fftshift(f2)
    ishift2 = np.fft.ifftshift(fshift2)
    iimg2 = np.fft.ifft2(ishift2)
    iimg2 = np.abs(iimg2)
    result[0:H:2, 1:W:2] = iimg2

    f3 = np.fft.fft2(b)
    f3[0, 0] = 0
    f3[:, 1:] = 0
    fshift3 = np.fft.fftshift(f3)
    ishift3 = np.fft.ifftshift(fshift3)
    iimg3 = np.fft.ifft2(ishift3)
    iimg3 = np.abs(iimg3)
    result[1:H:2, 1:W:2] = iimg3

    f4 = np.fft.fft2(g2)
    f4[0, 0] = 0
    f4[:, 1:] = 0
    fshift4 = np.fft.fftshift(f4)
    ishift4 = np.fft.ifftshift(fshift4)
    iimg4 = np.fft.ifft2(ishift4)
    iimg4 = np.abs(iimg4)
    result[1:H:2, 0:W:2] = iimg4

    result = np.expand_dims(result, axis=2)
    result = result / (16383 - 512)

    out = np.concatenate((result[0:H:2, 0:W:2, :],
                          result[0:H:2, 1:W:2, :],
                          result[1:H:2, 1:W:2, :],
                          result[1:H:2, 0:W:2, :]), axis=2)
    return out


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = network(in_image)

G_loss = tf.reduce_mean(tf.square(out_image - gt_image))

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_images = [None] * 6000
input_images = [None] * len(train_ids)


g_loss = np.zeros((5000, 1))

allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4

min_stddev = 0
max_stddev = 1.2e-4

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


for epoch in range(lastepoch, 2001):
    if os.path.isdir("result/%04d" % epoch):
        continue
    cnt = 0
    epoch_loss = 0
    if epoch > 1000:
        learning_rate = 1e-5
    for ind in np.random.permutation(len(train_ids)):
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        # get the path from image id



        ratio = np.random.uniform(100, 300)
        

        st = time.time()
        cnt += 1

        if input_images[ind] is None:

            raw = rawpy.imread(in_path)
            input_images[ind] = np.expand_dims(pack_raw(raw), axis=0)
            gt_raw = rawpy.imread(in_path)
            im = gt_raw.postprocess(use_camera_wb = True, half_size = False, no_auto_bright =True, output_bps = 16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[ind].shape[1]
        W = input_images[ind].shape[2]


        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)



        input_patch = input_images[ind][:, yy:yy + ps, xx:xx + ps, :] / ratio


        read = np.random.uniform(0, read_noise[int(train_id)])
        shot = np.random.uniform(0, shot_noise[int(train_id)])
        noise = np.random.randn(input_patch.shape[1], input_patch.shape[2], input_patch.shape[3]) * np.sqrt(shot * input_patch + read)
        pert = np.random.uniform(-7, 7)
        bias = (pert)/(16383-512)

        shape = np.random.uniform(1, 3)
        scale = 0.1 + (int(res[train_id]) - 50) * 3.7142e-5

        r = np.random.gamma(shape, scale, 512)
        g1 = np.random.gamma(shape, scale, 512)
        b = np.random.gamma(shape, scale, 512)
        g2 = np.random.gamma(shape, scale, 512)
        r = np.expand_dims(r, axis=1)
        r = np.expand_dims(r, axis=2)
        g1 = np.expand_dims(g1, axis=1)
        g1 = np.expand_dims(g1, axis=2)
        b = np.expand_dims(b, axis=1)
        b = np.expand_dims(b, axis=2)
        g2 = np.expand_dims(g2, axis=1)
        g2 = np.expand_dims(g2, axis=2)
        row = np.concatenate((r, g1, b, g2), axis=2)
        row = np.repeat(row, 512, axis=1)
        row = np.squeeze(row) / (16383-512)


        input_patch = input_patch + noise + bias + row
        input_patch = np.maximum(-1, np.minimum(input_patch * ratio , 1))

        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx *2 :xx*2 + ps*2, :]
      
        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

        _, G_current, output = sess.run([G_opt, G_loss, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current
        epoch_loss = epoch_loss + G_current
        print("%d %d Loss=%.5f Time=%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))


    saver.save(sess, checkpoint_dir + 'model.ckpt')
    print("Epoch Loss=%.5f" % (epoch_loss))

