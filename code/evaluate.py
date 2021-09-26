import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *
from absl import app, flags
from metrics import print_metrics

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_string('dataset', 'taxi-bj/p1', 'Directory of dataset.')
flags.DEFINE_integer('n_channels', 128, 'Number of channels.')
flags.DEFINE_integer('height', 32, 'Height of the flow map.')
flags.DEFINE_integer('width', 32, 'Width of the flow map.')
flags.DEFINE_integer('scale_factor', 4, 'Upscaling factor.')
flags.DEFINE_integer('use_eif', 1, 'External influence factors.')
flags.DEFINE_string('model', 'stcf', 'Loaded model name.')


# gpu
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def main(argv):
    n_channels = FLAGS.n_channels
    scale_factor = FLAGS.scale_factor
    b_size = FLAGS.batch_size
    use_eif = FLAGS.use_eif

    # load map
    fine_grained_flow_maps = np.load('./data/' + FLAGS.dataset + '/map.npy')
    # load eif
    external_influence_factors = np.load('./data/' + FLAGS.dataset + '/eif.npy')
    eif_dim = len(external_influence_factors[0])
    _, _, test_ds = load_map(fine_grained_flow_maps, external_influence_factors, scale_factor=scale_factor,
                             batch_size=b_size)

    # define model
    # input: coarse-grained flow map
    cm = tf.keras.Input(shape=(FLAGS.height, FLAGS.width, 1), name='c-map')
    eif = tf.keras.Input(shape=eif_dim, name='eif')

    # Gaussian noise
    cms = tf.keras.layers.Concatenate(axis=-1)([cm] * (n_channels // 2))
    cms = tf.keras.layers.GaussianNoise(1)(cms)

    # STCF EIF aggregation module
    if use_eif:
        eif_maps = tf.keras.layers.Reshape(target_shape=(1, 1, eif_dim))(eif)
        eif_maps = tf.keras.layers.UpSampling2D(size=(FLAGS.height, FLAGS.width))(eif_maps)
        eif_maps = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(n_channels, activation='relu'))(eif_maps)
        eif_maps = tf.keras.layers.Dropout(.25)(eif_maps)
        eif_maps = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(n_channels // 2, activation='relu'))(eif_maps)
        cm_eif = tf.keras.layers.Concatenate(axis=-1)([cms, eif_maps])
    else:
        cm_eif = cm

    # spatial-contrasting coarse-grained encoder
    sc_x = tf.keras.layers.Conv2D(n_channels, 9, padding='same', activation='relu')(cm)
    sc_h = tf.keras.layers.Conv2D(n_channels, 3, padding='same', activation='relu')(sc_x)

    # temporal-contrasting coarse-grained encoder
    tc_x = tf.keras.layers.Conv2D(n_channels, 9, padding='same', activation='relu')(cm)
    tc_h = tf.keras.layers.Conv2D(n_channels, 3, padding='same', activation='relu')(tc_x)

    # fine-tuned encoder
    ft_x = tf.keras.layers.Conv2D(n_channels, 9, padding='same', activation='relu')(cm_eif)
    ft_h = tf.keras.layers.Conv2D(n_channels, 3, padding='same', activation='relu')(ft_x)

    x = tf.keras.layers.Concatenate(axis=-1)([sc_h, tc_h, ft_h])

    # decoder
    x = tf.keras.layers.Conv2D(n_channels, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Dropout(.5)(x)
    x = tf.keras.layers.Conv2D(scale_factor ** 2 * n_channels, 3, padding='same', activation='relu')(x)
    x = tf.nn.depth_to_space(x, scale_factor)
    fm = tf.keras.layers.Conv2D(1, 3, padding='same', activation='relu')(x)

    # s^2-normalization
    w = tf.keras.layers.AveragePooling2D(scale_factor)(fm) * scale_factor ** 2
    w = tf.keras.layers.UpSampling2D(scale_factor)(w)
    w = tf.divide(fm, w + 1e-7)
    up_c = tf.keras.layers.UpSampling2D(scale_factor)(cm)
    fm = tf.multiply(w, up_c)

    fine_tuned_model = tf.keras.Model([cm, eif], fm, name='fine-tuned-model')

    fine_tuned_model.load_weights('./saved_model/' + FLAGS.dataset + '/ft/' + FLAGS.model + '/fine-tune')
    pred = fine_tuned_model.predict(((cm, eif), fm) for cm, fm, eif in test_ds)

    print('*' * 64)
    print_metrics(pred, fine_grained_flow_maps[int(len(fine_grained_flow_maps)*.75):])
    print('*' * 64)


if __name__ == '__main__':
    app.run(main)
