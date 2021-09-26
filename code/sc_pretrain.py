import time

from absl import app, flags

from utils import *

# flags
FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_string('dataset', 'taxi-bj/p1', 'Directory of dataset.')
flags.DEFINE_integer('fraction', 100, 'Fraction of labeled data.')
flags.DEFINE_integer('n_channels', 128, 'Number of channels.')
flags.DEFINE_integer('scale_factor', 4, 'Upscaling factor.')
flags.DEFINE_integer('height', 32, 'Height of the flow map.')
flags.DEFINE_integer('width', 32, 'Width of the flow map.')
flags.DEFINE_integer('epochs', 200, 'Max number of epochs.')
flags.DEFINE_integer('patience', 100, 'Early stopping patience.')
flags.DEFINE_float('temperature', .1, 'Temperature parameter for InfoNCE loss.')
flags.DEFINE_string('model', 'sc', 'Saved model name.')


# gpu
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def main(argv):
    start_time = time.time()
    n_channels = FLAGS.n_channels
    scale_factor = FLAGS.scale_factor
    b_size = FLAGS.batch_size

    # load train data
    fine_grained_flow_maps = np.load('./data/' + FLAGS.dataset + '/map.npy')
    # load eif
    external_influence_factors = np.load('./data/' + FLAGS.dataset + '/eif.npy')

    # return dataset
    train_ds = load_map(fine_grained_flow_maps, external_influence_factors, scale_factor=scale_factor,
                        fraction=FLAGS.fraction, batch_size=b_size, pre_train=True)

    # define model
    # encoder coarse
    cm = tf.keras.Input(shape=(FLAGS.height, FLAGS.width, 1), name='c-map')
    x = tf.keras.layers.Conv2D(n_channels, 9, padding='same', activation='relu')(cm)
    ch = tf.keras.layers.Conv2D(n_channels, 3, padding='same', activation='relu')(x)
    c_enc = tf.keras.Model(cm, ch, name='c-enc')
    x = tf.keras.layers.BatchNormalization()(ch)
    x = tf.keras.layers.Conv2D(n_channels * scale_factor, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    cz = tf.keras.layers.Dense(n_channels * scale_factor, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    # encoder fine
    fm = tf.keras.Input(shape=(FLAGS.height*scale_factor, FLAGS.width*scale_factor, 1), name='f-map')
    x = tf.keras.layers.Conv2D(n_channels, 9, padding='same', activation='relu')(fm)
    fh = tf.keras.layers.Conv2D(n_channels, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(fh)
    x = tf.keras.layers.Conv2D(n_channels * scale_factor, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    fz = tf.keras.layers.Dense(n_channels * scale_factor, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    pretrained_model = tf.keras.Model(inputs=[cm, fm], outputs=[cz, fz], name=FLAGS.model)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(FLAGS.lr)

    # criterion
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    # loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(c_map, f_map):
        with tf.GradientTape() as tape:
            train_cz, train_fz = pretrained_model((c_map, f_map))
            loss = loss_fn(train_cz, train_fz, criterion, b_size, FLAGS.temperature)

        gradients = tape.gradient(loss, pretrained_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pretrained_model.trainable_variables))

        train_loss(loss)

    # start pre-training
    best_pretraining_loss = float('inf')
    patience = FLAGS.patience

    for epoch in range(FLAGS.epochs):
        epoch_start_time = time.time()
        train_loss.reset_states()

        for train_coarse_maps, train_fine_maps, train_eif in train_ds:
            train_step(train_coarse_maps, train_fine_maps)

        if train_loss.result() < best_pretraining_loss:
            best_pretraining_loss = train_loss.result()
            patience = FLAGS.patience

            # save model
            c_enc.save_weights('./saved_model/' + FLAGS.dataset + '/pt/' + FLAGS.model + '/pre-trained-model')

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result():.6f}, '
                f'Time Cost: {time.time() - epoch_start_time:.2f}s'
            )
        else:
            patience -= 1

        if patience == 0:
            break

    print(f'Total running time: {(time.time()-start_time)//60:.0f}mins {(time.time()-start_time)%60:.0f}s')


if __name__ == '__main__':
    app.run(main)
