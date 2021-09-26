import time

from absl import app, flags

from utils import *

# flags
FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_string('dataset', 'taxi-bj/p1', 'Directory of dataset.')
flags.DEFINE_integer('fraction', 100, 'Fraction of labeled data.')
flags.DEFINE_integer('n_channels', 128, 'Number of channels.')
flags.DEFINE_integer('height', 32, 'Height of the flow map.')
flags.DEFINE_integer('width', 32, 'Width of the flow map.')
flags.DEFINE_integer('scale_factor', 4, 'Upscaling factor.')
flags.DEFINE_integer('epochs', 1000, 'Max number of epochs.')
flags.DEFINE_integer('patience', 100, 'Early stopping patience.')
flags.DEFINE_integer('use_eif', 1, 'External influence factors.')
flags.DEFINE_string('sc', 'test', 'SC model name.')
flags.DEFINE_string('tc', 'test', 'TC model name.')
flags.DEFINE_string('model', 'test', 'Saved model name.')
flags.DEFINE_integer('gpu', 0, 'Which gpu to use.')


# gpu
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def main(argv):
    start_time = time.time()
    n_channels = FLAGS.n_channels
    scale_factor = FLAGS.scale_factor
    b_size = FLAGS.batch_size
    use_eif = FLAGS.use_eif

    # load map
    fine_grained_flow_maps = np.load('./data/' + FLAGS.dataset + '/map.npy')
    # load eif
    external_influence_factors = np.load('./data/' + FLAGS.dataset + '/eif.npy')
    eif_dim = len(external_influence_factors[0])
    train_ds, val_ds, _ = load_map(fine_grained_flow_maps, external_influence_factors, scale_factor=scale_factor,
                                         fraction=FLAGS.fraction, batch_size=b_size,)

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
    sc_enc = tf.keras.Model(cm, sc_h, name='sc-enc')

    # temporal-contrasting coarse-grained encoder
    tc_x = tf.keras.layers.Conv2D(n_channels, 9, padding='same', activation='relu')(cm)
    tc_h = tf.keras.layers.Conv2D(n_channels, 3, padding='same', activation='relu')(tc_x)
    tc_enc = tf.keras.Model(cm, tc_h, name='tc-enc')

    # fine-tuning encoder
    ft_x = tf.keras.layers.Conv2D(n_channels, 9, padding='same', activation='relu')(cm_eif)
    ft_h = tf.keras.layers.Conv2D(n_channels, 3, padding='same', activation='relu')(ft_x)

    # load weights
    sc_enc.load_weights('./saved_model/' + FLAGS.dataset + '/pt/' + FLAGS.sc + '/pre-trained-model')
    tc_enc.load_weights('./saved_model/' + FLAGS.dataset + '/pt/' + FLAGS.tc + '/pre-trained-model')

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

    fine_tuned_model = tf.keras.Model([cm, eif], fm, name=FLAGS.model)
    fine_tuned_model.summary()

    # optimizer
    optimizer = tf.keras.optimizers.Adam(FLAGS.lr)

    # criterion
    criterion = tf.keras.losses.MeanSquaredError()
    mse_loss = tf.keras.losses.MeanSquaredError()

    # loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_mse = tf.keras.metrics.Mean(name='train_mse')
    val_mse = tf.keras.metrics.Mean(name='val_mse')

    @tf.function
    def train_step(c_map, f_map, eif):
        with tf.GradientTape() as tape:
            pred_f_map = fine_tuned_model([c_map, eif], training=True)
            loss = criterion(pred_f_map, f_map)

        gradients = tape.gradient(loss, fine_tuned_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, fine_tuned_model.trainable_variables))

        train_loss(loss)
        train_mse(mse_loss(pred_f_map, f_map))

    @tf.function
    def val_step(c_map, f_map, eif):
        pred_f_map = fine_tuned_model([c_map, eif], training=False)
        v_loss = mse_loss(pred_f_map, f_map)

        val_mse(v_loss)

    # start fine-tuning
    best_val_loss = float('inf')
    patience = FLAGS.patience

    for epoch in range(FLAGS.epochs):
        epoch_start_time = time.time()
        train_loss.reset_states()
        train_mse.reset_states()
        val_mse.reset_states()

        for train_coarse_maps, train_fine_maps, train_eif in train_ds:
            train_step(train_coarse_maps, train_fine_maps, train_eif)

        for val_coarse_maps, val_fine_maps, val_eif in val_ds:
            val_step(val_coarse_maps, val_fine_maps, val_eif)

        if val_mse.result() < best_val_loss:
            best_val_loss = float(val_mse.result())
            patience = FLAGS.patience

            # save model
            fine_tuned_model.save_weights('./saved_model/' + FLAGS.dataset + '/ft/' + FLAGS.model + '/fine-tune')

        else:
            patience -= 1

        if patience == 0:
            break

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result():.6f}, '
            f'Val Loss: {val_mse.result():.3f}, '
            f'Time Cost: {time.time() - epoch_start_time:.2f}s'
        )

    print('*' * 64)
    print(f'Total running time: {(time.time()-start_time)//60:.0f}mins {(time.time()-start_time)%60:.0f}s')
    print('*' * 64)


if __name__ == '__main__':
    app.run(main)
