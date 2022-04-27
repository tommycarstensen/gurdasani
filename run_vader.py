from VaDER.tensorflow2.vader.vader import VADER
import numpy as np
import os
import re

k_range = list(range(1, 12 + 1))
k_cross_validation = 20

def main():

    # axis 0: individuals; approximately 16k
    # axis 1: years; approximately 1940-2016
    # axis 2: ICD codes; approximately 60
    print('loading X_granular.npy')
    X_train = np.load('X_granular.npy')
    print('loading W_granular.npy')
    W_train = np.load('W_granular.npy')

    save_path = os.path.join(
        'out_vader', 'vader.ckpt_cell_type_LSTM_recurrent_True_granular')

    # Exclude years with missingness greater than 40%; i.e. keep 1999-2016.
    W_train = W_train[:, 1999-1940:2016-1940+1, :]
    X_train = X_train[:, 1999-1940:2016-1940+1, :]

    # simple(save_path, X_train, W_train)

    # https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
    # hyperparameter_optimization(
    #     X_train, W_train,
    #     k_cross_validation,
    #     )

    prediction_strength(X_train, W_train)

    return


def prediction_strength(X, W):

    # Calculate prediction strength to choose number of clusters.

    # Steps here:
    # https://academic.oup.com/gigascience/article/8/11/giz134/5626377?login=false

    # from plot_violin.py - values that give lowest reconstruction loss
    batch_size = 16
    learning_rate = 0.0001
    n_hidden = [12, 2]

    # split samples into two
    samples = X.shape[0]
    sample_size = X.shape[0] // 2
    X1 = np.delete(X, list(range(0, sample_size + 1)), 0)
    X2 = np.delete(X, list(range(sample_size + 1, samples)), 0)
    W1 = np.delete(W, list(range(0, sample_size + 1)), 0)
    W2 = np.delete(W, list(range(sample_size + 1, samples)), 0)

    for k in k_range:

        if os.path.isfile(f'k{k}.txt'):
            continue
        with open(f'k{k}.txt', 'w') as f:
            print('', file=f)

        save_path = f'out/k{k}'

        # 1) Train VaDER on the training data (the training data model).
        vader1 = VADER(
            X_train=X1,
            W_train=W1,
            save_path=save_path,
            n_hidden=n_hidden,
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
            )

        vader1.pre_fit(n_epoch=50, verbose=True)

        vader1.fit(n_epoch=50, verbose=True)

        # with open('get_loss.txt', 'a') as f:
        #     d = vader.get_loss(X_train, W_train)
        #     print(
        #         i_cross_validation,
        #         k, batch_size, learning_rate,
        #         n_hidden[0], n_hidden[1],
        #         re.search(
        #             'numpy=([0-9.]*)',
        #             str(d['reconstruction_loss']),
        #             )[1],
        #         re.search(
        #             'numpy=([0-9.]*)',
        #             str(d['latent_loss']),
        #             )[1],
        #         sep='\t', file=f,
        #         )

        # 2) Assign clusters to the test data using the training data model.
        # returns one dimensional array of clusters
        c1 = vader1.cluster(X2)
        with open(os.path.join(save_path, 'c1.npy'), 'wb') as f:
            np.save(f, c1)

        # 3) Train VaDER on the test data (the test data model).
        vader2 = VADER(
            X_train=X2,
            W_train=W2,
            save_path=save_path,
            n_hidden=n_hidden,
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
            )

        # 4) Assign clusters to the test data using the test data model.
        # returns one dimensional array of clusters
        c2 = vader2.cluster(X2)
        with open(os.path.join(save_path, 'c2.npy'), 'wb') as f:
            np.save(f, c2)

        # 5) Compare the resulting 2 clusterings: for each cluster of the test data model, compute the fraction of pairs of samples in that cluster that are also assigned to the same cluster by the training data model.
        # Prediction strength is defined as the minimum proportion across all clusters of the test data model [43].
        print(c1, c2)
        ps = np.sum(c1 == c2) / len(c1)
        print(ps)
        for p1, p2 in zip(c1, c2):
            if p1 == p2:
                ps += 1
        print(ps)
        with open(f'k{k}.txt', 'w') as f:
            print(ps, file=f)

    return


def simple(save_path, X_train, W_train):

    for k in range(2, 6):

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
        print(f"model = keras.models.load_model('{save_path}') ; dir(model)")

        exit()

    return


def hyperparameter_optimization(X, W,):

    sample_size = X.shape[0] // k_cross_validation
    if not os.path.isdir('touch'):
        os.mkdir('touch')

    combinations_run = set()
    if os.path.isfile('get_loss.txt'):
        with open('get_loss.txt') as f:
            for line in f:
                combinations_run.add(tuple(line.split()[:-2]))

    for i_cross_validation in range(k_cross_validation):
        i1 = (i_cross_validation + 0) * sample_size
        i2 = (i_cross_validation + 1) * sample_size
        obj = range(i1, i2 + 1)
        X_train = np.delete(X, list(range(i1, i2 + 1)), 0)
        W_train = np.delete(W, list(range(i1, i2 + 1)), 0)
        # k: Number of mixture components. (default: 3)
        for k in k_range:
            # for num_layers in (1, 2):
            # Batch size used for training. (default: 32)
            for batch_size in (16, 32, 64, 128):
                # Learning rate for training. (default: 1e-3)
                for learning_rate in (1e-4, 1e-3, 1e-2):
                    # The hidden layers. List of length >= 1.
                    # Specification of the number of nodes.
                    for n_hidden in ([12, 2], [36, 4], [36, 8]):

                        t = (
                            f'i_cross_validation_{i_cross_validation}',
                            f'k_{k}',
                            f'batch_size_{batch_size}',
                            f'learning_rate_{learning_rate}',
                            f'n_hidden_{n_hidden[0]}_{n_hidden[1]}',
                            )

                        t2 = tuple(map(str, [
                            i_cross_validation,
                            k,
                            batch_size,
                            learning_rate,
                            n_hidden[0],
                            n_hidden[1],
                            ]))

                        save_path = os.path.join(
                            'out_vader',
                            'vader.ckpt_cell_type_LSTM_recurrent_True_granular',
                            '_'.join(t),
                            )

                        if os.path.isfile('touch/{}'.format('_'.join(t))):
                            if t2 in combinations_run:
                                continue
                            with open('touch/{}'.format('_'.join(t))) as f:
                                if f.read() == 'initiated\n':
                                    continue
                        with open('touch/{}'.format('_'.join(t)), 'w') as f:
                            f.write('initiated\n')

                        # Be verbose.
                        print(
                            'i_cross_validation', i_cross_validation,
                            'k', k,
                            'batch_size', batch_size,
                            'learning_rate', learning_rate,
                            'n_hidden', n_hidden,
                            )

                        vader = VADER(
                            X_train=X_train,
                            W_train=W_train,
                            save_path=save_path,
                            n_hidden=n_hidden,
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
                            )

                        vader.pre_fit(n_epoch=50, verbose=True)

                        vader.fit(n_epoch=50, verbose=True)

                        with open('get_loss.txt', 'a') as f:
                            d = vader.get_loss(X_train, W_train)
                            print(
                                i_cross_validation,
                                k, batch_size, learning_rate,
                                n_hidden[0], n_hidden[1],
                                re.search(
                                    'numpy=([0-9.]*)',
                                    str(d['reconstruction_loss']),
                                    )[1],
                                re.search(
                                    'numpy=([0-9.]*)',
                                    str(d['latent_loss']),
                                    )[1],
                                sep='\t', file=f,
                                )

                        # returns one dimensional array of clusters
                        c = vader.cluster(X_train)
                        with open(os.path.join(save_path, 'c.npy'), 'wb') as f:
                            np.save(f, c)

                        # returns predicted array with same dimensions as X_train                        
                        p = vader.predict(X_train)
                        with open(os.path.join(save_path, 'p.npy'), 'wb') as f:
                            np.save(f, p)

                        with open('touch/{}'.format('_'.join(t)), 'w') as f:
                            f.write('finished\n')

                        exit()

    return


if __name__ == '__main__':
    main()
