import numpy as np
from sklearn.model_selection import KFold

from data import load_grape_data

from .model import build_model, shape


def train():
    print('Loading data')
    X, Y = get_data()

    model = build_model()
    print('Data loaded, start training')
    kf = KFold(n_splits=5)
    i = 0
    for train, test in kf.split(X):
        print('Training split %s' % i)
        x = model.fit(X[train], Y[train], batch_size=32, epochs=1,
                      validation_data=(X[test], Y[test]), verbose=2)
        print(x)
        i += 1


def get_data():
    try:

        X = np.load('simple_x.npy')
        Y = np.load('simple_y.npy')
        return X, Y
    except IOError:
        print('persisted data not found, rebuilding...')
        X, _, Y, _ = load_grape_data(1, shape[0])
        X.resize((X.shape[0], *shape))
        np.save('simple_x.npy', X)
        np.save('simple_y.npy', Y)
        return X, Y
