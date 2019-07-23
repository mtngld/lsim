import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from monodepth.utils.evaluate_kitti import evaluate_kitti
from monodepth.bilinear_sampler import bilinear_sampler_1d_h
from monodepth.monodepth_model_siamese import MonodepthModel, monodepth_parameters
import argparse
import os

tf.logging.set_verbosity(tf.logging.INFO)

HEIGHT = 256
WIDTH = 512
BATCH_SIZE = 4

EIGEN_CONFIG = {
    'model_dir': './models/eigen/',
    'dataset': 'kitti',
    'train_filename': './monodepth/utils/filenames/eigen_train_files.txt',
    'test_filename': './monodepth/utils/filenames/eigen_test_files.txt',
    'base_path': '/mnt/data/data/',
    'warm_start': None,
    'model_params':
        {
            'disp_gradient_loss_weight': 0.1,
            'lr_loss_weight': 1.0,
            'alpha_image_loss': 0.85,
            'learning_rate': 1e-4
        }
}

CITYSCAPES_CONFIG = {
    'model_dir': './models/cityscapes/',
    'dataset': 'cityscapes',
    'train_filename': './filenames/cityscapes_train_files.txt',
    'test_filename': './filenames/cityscapes_test_files.txt',
    'base_path': '/mnt/data/data/',
    'warm_start': None,
    'model_params':
        {
            'disp_gradient_loss_weight': 0.1,
            'lr_loss_weight': 1.0,
            'alpha_image_loss': 0.85,
            'learning_rate': 1e-4
        }
}


def read_image(path, base_path, dataset):
    full_path = base_path + path
    file_contents = tf.read_file(full_path)
    img = tf.image.decode_jpeg(file_contents, channels=3)

    # if the dataset is cityscapes, crop the bottom to remove the car hood
    if dataset == 'cityscapes':
        o_height = tf.shape(img)[0]
        crop_height = (o_height * 4) // 5
        img = img[:crop_height, :, :]

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_images(
        img, [HEIGHT, WIDTH], tf.image.ResizeMethod.AREA)
    return img


def _augment(img, brightness, contrast, saturation, hue):
    img = tf.image.adjust_brightness(img, delta=brightness)
    img = tf.image.adjust_contrast(img, contrast_factor=contrast)
    img = tf.image.adjust_saturation(img, saturation_factor=saturation)
    img = tf.image.adjust_hue(img, delta=hue)

    # The random_* ops do not necessarily clamp.
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def augment_image_pair(left_image, right_image):
    # randomly shift gamma
    random_gamma = tf.random_uniform([], 0.8, 1.2)
    left_image_aug = left_image ** random_gamma
    right_image_aug = right_image ** random_gamma

    # randomly shift brightness
    random_brightness = tf.random_uniform([], 0.5, 2.0)
    left_image_aug = left_image_aug * random_brightness
    right_image_aug = right_image_aug * random_brightness

    # randomly shift color
    random_colors = tf.random_uniform([3], 0.8, 1.2)
    white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
    left_image_aug *= color_image
    right_image_aug *= color_image

    # saturate
    left_image_aug = tf.clip_by_value(left_image_aug,  0, 1)
    right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

    return left_image_aug, right_image_aug


def _parse_function(line, base_path, dataset, mode):
    path_left, path_right = tf.decode_csv(
        records=line,
        record_defaults=[[''], ['']],
        field_delim=' ')

    # path_left = tf.Print(path_left, [path_left, path_right])
    img_left_temp = read_image(path_left, base_path=base_path, dataset=dataset)
    img_right_temp = read_image(path_right, base_path=base_path, dataset=dataset)

    if mode == tf.estimator.ModeKeys.TRAIN:
        do_flip = tf.random_uniform([], 0, 1)
        img_left = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(img_right_temp), lambda: img_left_temp)
        img_right = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(img_left_temp),  lambda: img_right_temp)

        # randomly augment images
        do_augment = tf.random_uniform([], 0, 1)
        img_left, img_right = tf.cond(do_augment > 0.5, lambda: augment_image_pair(img_left, img_right), lambda: (img_left, img_right))

    else:
        img_left = img_left_temp
        img_right = img_right_temp

    return {'left': img_left, 'right': img_right}


def train_input_fn(filename, base_path, dataset):
    dataset = tf.data.TextLineDataset(filename) \
        .shuffle(buffer_size=30000) \
        .map(lambda line:
             _parse_function(
                line, base_path=base_path,
                dataset=dataset, mode=tf.estimator.ModeKeys.TRAIN)) \
        .apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE)) \
        .prefetch(32) \
        .repeat(1)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element, None


def test_input_fn(filename, base_path, dataset):
    dataset = tf.data.TextLineDataset(filename) \
        .map(lambda line:
             _parse_function(
                line, base_path=base_path,
                dataset=dataset, mode=tf.estimator.ModeKeys.PREDICT)) \
        .batch(1)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element, None


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = tf.shape(img)
    h = s[1]
    w = s[2]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
    return scaled_imgs


def generate_image_left(img, disp):
    return bilinear_sampler_1d_h(img, -disp)


def generate_image_right(img, disp):
    return bilinear_sampler_1d_h(img, disp)


def gradient_x(img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx


def gradient_y(img):
    gy = img[:, :-1, :, :] - img[:, 1:, :, :]
    return gy


def get_disparity_smoothness(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True))
                 for g in image_gradients_x]
    weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True))
                 for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    return smoothness_x + smoothness_y


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def encoder_layer(inputs, filters, kernel_size, activation):
    initializer = tf.contrib.layers.xavier_initializer()
    conv_a = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        strides=1,
        activation=activation,
        kernel_initializer=initializer)
    conv_b = tf.layers.conv2d(
        inputs=conv_a,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        strides=2,
        activation=activation,
        kernel_initializer=initializer)
    return conv_b


def decoder_layer(inputs, filters, kernel_size, activation, skip, prev):
    initializer = tf.contrib.layers.xavier_initializer()
    upsample = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=2,
        padding="same",
        kernel_initializer=initializer)

    if prev is not None:
        upsample_prev = tf.layers.conv2d_transpose(
            inputs=prev,
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            kernel_initializer=initializer)

    if skip is not None and prev is not None:
        concat = tf.concat([upsample, skip, upsample_prev], 3)
    elif skip is None and prev is not None:
        concat = tf.concat([upsample, upsample_prev], 3)
    elif skip is not None and prev is None:
        concat = tf.concat([upsample, skip], 3)
    else:
        concat = upsample

    out = tf.layers.conv2d(
        inputs=concat,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        strides=1,
        activation=activation,
        kernel_initializer=initializer)

    return out


def get_disp(x):
    initializer = tf.contrib.layers.xavier_initializer()
    out = tf.layers.conv2d(
        inputs=x,
        filters=1,
        kernel_size=3,
        padding="same",
        strides=1,
        activation=tf.nn.sigmoid,
        kernel_initializer=initializer)
    out = 0.3*out
    return out


def build_net(inputs):
    with tf.variable_scope('encoder_decoder'):
        layers = {'input_layer': inputs}

        encoder_spec = [
            {'name': 'enc1', 'filters': 32, 'kernel_size': 7,
             'input': 'input_layer'},
            {'name': 'enc2', 'filters': 64, 'kernel_size': 5, 'input': 'enc1'},
            {'name': 'enc3', 'filters': 128, 'kernel_size': 3, 'input': 'enc2'},
            {'name': 'enc4', 'filters': 256, 'kernel_size': 3, 'input': 'enc3'},
            {'name': 'enc5', 'filters': 512, 'kernel_size': 3, 'input': 'enc4'},
            {'name': 'enc6', 'filters': 512, 'kernel_size': 3, 'input': 'enc5'},
            {'name': 'enc7', 'filters': 512, 'kernel_size': 3, 'input': 'enc6'},
        ]

        for l_spec in encoder_spec:
            with tf.variable_scope(l_spec['name']):
                layers[l_spec['name']] = encoder_layer(
                    inputs=layers[l_spec['input']],
                    filters=l_spec['filters'],
                    kernel_size=l_spec['kernel_size'],
                    activation=tf.nn.elu)

        decoder_spec = [
            {'name': 'dec7', 'filters': 512, 'kernel_size': 3,
             'input': 'enc7', 'skip': 'enc6', 'prev': None},
            {'name': 'dec6', 'filters': 512, 'kernel_size': 3,
             'input': 'dec7', 'skip': 'enc5', 'prev': None},
            {'name': 'dec5', 'filters': 256, 'kernel_size': 3,
             'input': 'dec6', 'skip': 'enc4', 'prev': None},
            {'name': 'dec4', 'filters': 128, 'kernel_size': 3,
             'input': 'dec5', 'skip': 'enc3', 'prev': None},
            {'name': 'dec3', 'filters': 64, 'kernel_size': 3,
             'input': 'dec4', 'skip': 'enc2', 'prev': 'disp4'},
            {'name': 'dec2', 'filters': 32, 'kernel_size': 3,
             'input': 'dec3', 'skip': 'enc1', 'prev': 'disp3'},
            {'name': 'dec1', 'filters': 16, 'kernel_size': 3,
             'input': 'dec2', 'skip': None, 'prev': 'disp2'}
        ]

        for l_spec in decoder_spec:
            with tf.variable_scope(l_spec['name']):
                layers[l_spec['name']] = decoder_layer(
                    inputs=layers[l_spec['input']],
                    filters=l_spec['filters'],
                    kernel_size=l_spec['kernel_size'],
                    activation=tf.nn.elu,
                    skip=layers[l_spec['skip']
                                ] if l_spec['skip'] is not None else None,
                    prev=layers[l_spec['prev']] if l_spec['prev'] is not None else None)

                if l_spec['name'] in ['dec1', 'dec2', 'dec3', 'dec4']:
                    l_name = l_spec['name'].replace('dec', 'disp')
                    layers[l_name] = get_disp(layers[l_spec['name']])

        return layers


def monodepth_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    mono_params = monodepth_parameters(
        encoder='vgg',
        height=256,
        width=512,
        batch_size=8,
        num_threads=8,
        num_epochs=50,
        do_stereo=False,
        wrap_mode='border',
        use_deconv=False,
        alpha_image_loss=params["alpha_image_loss"],
        disp_gradient_loss_weight=params["disp_gradient_loss_weight"],
        lr_loss_weight=1.0,
        full_summary=True)

    if mode == tf.estimator.ModeKeys.TRAIN:
        monodepth_mode = 'train'
    else:
        monodepth_mode = 'test'

    model = MonodepthModel(mono_params, monodepth_mode,
                           features['left'], features['right'])

    start_learning_rate = params['learning_rate']
    total_steps = params['total_steps']
    global_step = tf.train.get_global_step()

    boundaries = [np.int32((3/5) * total_steps), np.int32((4/5) * total_steps)]
    values = [start_learning_rate, start_learning_rate / 2, start_learning_rate / 4]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    tf.summary.scalar('learning_rate', learning_rate)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=model.total_loss,
            global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=model.total_loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            'left_disp_est': model.disp_left_est[0],
            'right_disp_est': model.disp_right_est[0]
            }
        return tf.estimator.EstimatorSpec(
            mode=monodepth_mode, predictions=predictions)


def main():
    # args = parse_args()
    config = EIGEN_CONFIG

    chkp_state = tf.train.get_checkpoint_state(config['model_dir'])
    try:
        current_step = int(os.path.basename(chkp_state.model_checkpoint_path).split('-')[1])
    except AttributeError:
        current_step = 0

    train_size = sum(1 for line in open(config['train_filename']))
    current_epoch = current_step // (train_size // BATCH_SIZE)
    total_epochs = 50
    print('Current epoch: {}, current step: {}, total train size: {}'.format(
        current_epoch, current_step, train_size))

    total_steps = (train_size // BATCH_SIZE) * total_epochs
    config['model_params']['total_steps'] = total_steps

    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])
    eval_log_file = open(os.path.join(config['model_dir'], 'eval.log'), 'a+')
    for epoch in range(current_epoch, total_epochs):
        print("########### Starting epoch {} ###############".format(epoch))
        if epoch == 0 and \
                not os.path.exists(os.path.join(config['model_dir'], 'checkpoint')):
            ws = config['warm_start']
        else:
            ws = None
        est = tf.estimator.Estimator(
            model_fn=monodepth_model_fn,
            model_dir=config['model_dir'],
            warm_start_from=ws,
            params=config['model_params'])

        est.train(input_fn=lambda: train_input_fn(
            config['train_filename'],
            config['base_path'],
            config['dataset']))

        disparities_left = []
        for item in est.predict(input_fn=lambda: test_input_fn(
                config['test_filename'],
                config['base_path'],
                config['dataset'])):
            disparities_left.append(item['left_disp_est'].squeeze())

        disparities_left = np.stack(disparities_left)
        print(disparities_left.shape)
        np.save('./disparities_left.npy', disparities_left)

        str_result = evaluate_kitti(
            split='eigen',
            predicted_disp_path='./disparities_left.npy',
            gt_path=os.path.expanduser('~/data/'),
            garg_crop=True)

        eval_log_file.write('\n############\n\nEpoch: {}'.format(epoch))
        eval_log_file.write(str_result)
        print(str_result)

if __name__ == '__main__':
    main()