from VaDER.tensorflow2.vader.vader import VADER
import numpy as np
import os

# Default values:
# (X_train, W_train=None, y_train=None, n_hidden=[12, 2], k=3, groups=None,
# output_activation=None,
# batch_size = 32, learning_rate=1e-3, alpha=1.0, phi=None, cell_type="LSTM",
# cell_params=None, recurrent=True,
# save_path=None, eps=1e-10, seed=None, n_thread=0)

# X_train = np.load('X.npy')
# W_train = np.load('W.npy')
# save_path = os.path.join('out_vader', 'vader.ckpt_cell_type_LSTM_recurrent_True')

def main():

    print('loading X_granular.npy')
    X_train = np.load('X_granular.npy')
    print('loading W_granular.npy')
    W_train = np.load('W_granular.npy')

    save_path = os.path.join('out_vader', 'vader.ckpt_cell_type_LSTM_recurrent_True_granular')

    # exclude years with missingness greater than 40%
    W_train = W_train[:,1999-1940:2016-1940+1,:]
    X_train = X_train[:,1999-1940:2016-1940+1,:]

    # simple(save_path, X_train, W_train)

    # https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
    k_cross_validation = 3
    hyperparameter_optimization(
        save_path, X_train, W_train,
        k_cross_validation,
        )

    return


def simple(save_path, X_train, W_train):

    for k in range(2,6):
     if k != 3: continue

     if os.path.isfile('c_granular.{}.npy'.format(k)):
      continue

     vader = VADER(
        k=k,
        X_train=X_train,
        W_train=W_train,
        save_path=save_path,
        # n_hidden=[12, 2],
        # k=4,
        # learning_rate=1e-3,
        # output_activation=None,
        # recurrent=True,
        # batch_size=64,
        # recurrent=False,
        # batch_size=16,

        # cell_type="LSTM",
        # recurrent=True,

        #           # learning_rate=1e-3, output_activation=None, recurrent=True, cell_type="Transformer", batch_size=64,
        #           # cell_params={'d_model': 4, 'num_layers': 1, 'num_heads': 1, 'dff': 16, 'rate': 0.0},
        )

     vader.pre_fit(n_epoch=50, verbose=True)

     vader.fit(n_epoch=50, verbose=True)

     print('_reconstruction_loss', vader._reconstruction_loss)
     print('_latent_loss', vader._latent_loss)
     print('reconstruction_loss', vader.reconstruction_loss)
     print('latent_loss', vader.latent_loss)
     print('get_loss', vader.get_loss(X_train, W_train))
     print('loss', vader.loss)

     c = vader.cluster(X_train)
     with open('c_granular.{}.npy'.format(k), 'wb') as f:
        np.save(f, c)

     p = vader.predict(X_train)
     with open('p_granular.{}.npy'.format(k), 'wb') as f:
        np.save(f, p)

    print(save_path)
    print("model = keras.models.load_model('{}') ; dir(model)".format(save_path))

    exit()

    return


def hyperparameter_optimization(
    save_path, X, W,
    k_cross_validation,
    ):

    sample_size = X.shape[0] // k_cross_validation
    for i_cross_validation in range(k_cross_validation):
        print('i_cross_validation', i_cross_validation)
        i1 = (i_cross_validation + 0) * sample_size
        i2 = (i_cross_validation + 1) * sample_size
        X_train = X[i1:i2,:,:]
        W_train = W[i1:i2,:,:]
        # k: Number of mixture components. (default: 3)
        for k in range(1, 12 + 1):
            print('k', k)
            # for num_layers in (1, 2):
            # Batch size used for training. (default: 32)
            for batch_size in (16, 32, 64, 128):
                # Learning rate for training. (default: 1e-3)
                for learning_rate in (1e-4, 1e-3, 1e-2):
                    # The hidden layers. List of length >= 1. Specification of the number of nodes in the hidden layers.
                    for n_hidden in ([12,2], [36,4], [36,8]):

                        vader = VADER(
                            X_train=X_train,
                            W_train=W_train,
                            save_path=save_path,
                            # n_hidden=[12, 2],
                            # num_layers=num_layers,
                            k=k,
                            learning_rate=learning_rate,
                            # output_activation=None,
                            # recurrent=True,
                            batch_size=batch_size,
                            # recurrent=False,
                            # batch_size=16,

                            # cell_type="LSTM",
                            # recurrent=True,

                            #           # learning_rate=1e-3, output_activation=None, recurrent=True, cell_type="Transformer", batch_size=64,
                            #           # cell_params={'d_model': 4, 'num_layers': 1, 'num_heads': 1, 'dff': 16, 'rate': 0.0},
                            )

                        vader.pre_fit(n_epoch=50, verbose=True)

                        vader.fit(n_epoch=50, verbose=True)

                        with open('get_loss.txt', 'a') as f:
                            d = vader.get_loss(X_train, W_train)
                            print(
                                k_cross_validation,
                                k, batch_size, learning_rate,
                                n_hidden[0], n_hidden[1],
                                d['reconstruction_loss'], d['latent_loss'],
                                sep='\t', file=f,
                                )

                        c = vader.cluster(X_train)
                        with open(os.path.join(save_path, 'c.npy'), 'wb') as f:
                            np.save(f, c)

                        p = vader.predict(X_train)
                        with open(os.path.join(save_path, 'p.npy'), 'wb') as f:
                            np.save(f, p)

    return


if __name__ == '__main__':
    main()