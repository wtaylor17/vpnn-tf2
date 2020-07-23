import tensorflow as tf
import argparse

from vpnn import vpnn


parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, default=1, help='number of vpnn layers')
parser.add_argument('--rotations', type=int, default=2, help='1/2 the number of total rotations')
parser.add_argument('--theta_initializer', type=str, default='uniform', help='initializer for angles')
parser.add_argument('--t_initializer', type=str, default='uniform', help='initializer for t params')
parser.add_argument('--bias_initializer', type=str, default='uniform', help='initializer for biases')
parser.add_argument('--dense', action='store_true', help='if present, use a dense output not SVD')
parser.add_argument('--nobias', action='store_true', help='if present no bias is used')
parser.add_argument('--noperm', action='store_true', help='if present no permutation is used')
parser.add_argument('--nodiag', action='store_true', help='if present no diagonal is used')
parser.add_argument('--diagonal_fn', type=str, default=None, help='a diagonal function (from tf.keras.activations)')
parser.add_argument('--hidden_activation', type=str, default='relu', help='activation for hidden layers')
parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer for training')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
parser.add_argument('--tensorboard', action='store_true', help='to use tensorboard')
parser.add_argument('--name', type=str, default='mnistvpnn', help='name of model (checkpoint and logs)')
parser.add_argument('--save_checkpoints', action='store_true', help='if present save the model per epoch')
parser.add_argument('--trainable_M', action='store_true', help='if present make cheby activations trainable')
parser.add_argument('--cheby_M', type=float, default=1.3, help='initial cheby M (if not trainable)')
parser.add_argument('--momentum', type=float, default=0.0, help='optimizer momentum if applicable')

args = parser.parse_args()


class TFSaveAndTestCallback(tf.keras.callbacks.Callback):
    def __init__(self, xdata, ydata, metrics=None):
        super().__init__()
        self.metrics = metrics or []
        self.data = [xdata, ydata]

    def on_epoch_end(self, epoch, logs=None):
        path = f'models/mnist/{args.name}-epoch{epoch + 1}-of-{args.epochs}'
        tf.saved_model.save(self.model, path)
        print(f'\n\n\n**************Saved to {path}************')
        print(f'******************metrics****************')
        loaded = tf.saved_model.load(path)
        infer = loaded.signatures['serving_default']
        pred = infer(tf.constant(self.data[0], dtype=tf.float32))
        h = pred[list(pred.keys())[0]]
        for metric in self.metrics:
            val = tf.reduce_mean(metric(self.data[1], h))
            print(f'{metric.__name__}: {val}')
        print('***************************************\n\n')


if __name__ == '__main__':
    model = vpnn(28*28,
                 n_layers=args.layers,
                 n_rotations=args.rotations,
                 theta_initializer=args.theta_initializer,
                 t_initializer=args.t_initializer,
                 bias_initializer=args.bias_initializer,
                 output_dim=None if args.dense else 10,
                 use_bias=not args.nobias,
                 use_permutations=not args.noperm,
                 use_diagonals=not args.nodiag,
                 diagonal_fn=args.diagonal_fn,
                 hidden_activation=args.hidden_activation,
                 output_activation=args.hidden_activation if args.dense else 'softmax',
                 trainable_M=args.trainable_M,
                 M_init=args.cheby_M)
    if args.optimizer == 'rmsprop':
        optimizer = tf.optimizers.RMSprop(momentum=args.momentum)
    elif args.optimizer == 'sgd':
        optimizer = tf.optimizers.SGD(momentum=args.momentum)
    else:
        optimizer = args.optimizer
    if args.dense:
        output = tf.keras.layers.Dense(10, activation='softmax')(model.output)
    else:
        output = model.output
    model = tf.keras.Model(model.input, output)
    model.compile(optimizer=args.optimizer, loss='categorical_crossentropy', metrics='accuracy')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255
    x_test = x_test.reshape(-1, 28*28) / 255
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    callbacks = []

    if args.tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{args.name}'))
    if args.save_checkpoints:
        callbacks.append(TFSaveAndTestCallback(metrics=[tf.keras.losses.categorical_crossentropy,
                                                        tf.keras.metrics.categorical_accuracy],
                                               xdata=x_test, ydata=y_test))

    model.summary()
    print('optimizer =', optimizer)
    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              callbacks=callbacks)
    tf.saved_model.save(model, f'models/mnist/{args.name}-final')
    print('All done :)')
