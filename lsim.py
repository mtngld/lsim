import tensorflow as tf
import numpy as np
from monodepth.utils.evaluate_kitti import evaluate_kitti
from lsim_model import LsimModel, monodepth_parameters
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
        img_left = tf.cond(
            do_flip > 0.5,
            lambda: tf.image.flip_left_right(img_right_temp),
            lambda: img_left_temp)
        img_right = tf.cond(
            do_flip > 0.5,
            lambda: tf.image.flip_left_right(img_left_temp),
            lambda: img_right_temp)

        # randomly augment images
        do_augment = tf.random_uniform([], 0, 1)
        img_left, img_right = tf.cond(
            do_augment > 0.5,
            lambda: augment_image_pair(img_left, img_right),
            lambda: (img_left, img_right))

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


def monodepth_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    _params = monodepth_parameters(
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
        _mode = 'train'
    else:
        _mode = 'test'

    model = LsimModel(_params, _mode,
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
            mode=_mode, predictions=predictions)


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


if __name__ == '__main__':
    main()