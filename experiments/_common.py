"""Общая загрузка данных и базовые величины сигнала (чтобы не повторять в каждом скрипте)."""
import os
import numpy as np
import pandas as pd

SEED = 42


def load():
    """Считать файл, инвертировать сигнал, рассчитать baseline, амплитуду, пик, SNR."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'data', 'Run200_Wave_0_1.txt')
    raw = pd.read_csv(path, sep=' ', header=None, skipinitialspace=True)
    X = raw.drop([0, 1, 2, 3, 504], axis=1).to_numpy(float)
    Y = (2**14) - X                       # инверсия: импульс становится положительным
    baseline = Y[:, :120].mean(1)
    noise = Y[:, :120].std(1)
    Yc = Y - baseline[:, None]
    amplitude = Yc.max(1)
    peak_index = Yc.argmax(1)
    snr = amplitude / noise
    return dict(X=X, Yc=Yc, baseline=baseline, noise=noise,
                amplitude=amplitude, peak_index=peak_index, snr=snr)


def cumsum(Yc):
    """Префикс-суммы: площадь окна [a:b] = C[:, b] - C[:, a]."""
    return np.concatenate([np.zeros((len(Yc), 1)), np.cumsum(Yc, axis=1)], axis=1)
