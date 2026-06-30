"""Улучшение границы гамма/нейтрон и финальная модель.
Хотел вытащить тип частицы в зоне слияния ветвей."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from _common import load, cumsum, SEED

d = load()
X, Yc, amplitude, snr = d['X'], d['Yc'], d['amplitude'], d['snr']
C = cumsum(Yc)
area = lambda a, b: C[:, b] - C[:, a]
tot = area(145, 350)
psd = area(160, 350) / np.where(np.abs(tot) < 1, np.nan, tot)
psd2 = area(170, 350) / np.where(np.abs(tot) < 1, np.nan, tot)
aoa = tot / np.where(amplitude < 1, np.nan, amplitude)
log_amp = np.log10(np.clip(amplitude, 1, None))
psd_med = np.nanmedian(psd)
psd_f = np.nan_to_num(psd, nan=psd_med)

# простые признаки формы
POST = Yc[:, 150:270] / np.where(amplitude < 1, np.nan, amplitude)[:, None]
def fall(fr):
    b = POST < fr
    t = b.argmax(1).astype(float)
    t[~b.any(1)] = POST.shape[1]
    return t
fall20, fall10 = fall(.2), fall(.1)

# выпрямляю PSD от тренда по амплитуде для устойчивых псевдометок
fit = snr > np.quantile(snr, 0.10)
k_, b_ = np.polyfit(log_amp[fit & np.isfinite(psd)], psd[fit & np.isfinite(psd)], 1)
resid = psd_f - (k_ * log_amp + b_)

F = StandardScaler().fit_transform(np.column_stack([
    log_amp, psd_f, np.nan_to_num(psd2, nan=np.nanmedian(psd2)),
    np.nan_to_num(aoa, nan=np.nanmedian(aoa)), fall20, fall10, resid]))

# выпрямленный PSD как отдельная модель отбросил: на Kaggle 0.834 что хуже базы 0.838
# self-training: seed из ярких -> итеративно добавляю p>0.98 -> переобучаю
seed = (snr > np.quantile(snr, 0.50)) & (np.abs(resid) > 0.04)
y = np.full(len(F), -1)
y[seed] = (resid[seed] < 0).astype(int)
labeled = y >= 0
for _ in range(15):
    clf = LogisticRegression(max_iter=1000).fit(F[labeled], y[labeled])
    rest = ~labeled
    if rest.sum() == 0:
        break
    p = clf.predict_proba(F[rest])
    add = p.max(1) > 0.98
    if add.sum() == 0:
        break
    ri = np.where(rest)[0]
    y[ri[add]] = clf.classes_[p.argmax(1)][add]
    labeled = y >= 0
clf = LogisticRegression(max_iter=1000).fit(F[labeled], y[labeled])
main = y.copy()
main[y < 0] = clf.predict(F[y < 0])
main = main.astype(int)
if np.nanmean(psd[main == 1]) > np.nanmean(psd[main == 0]):
    main = 1 - main
print('размечено уверенно:', int(labeled.sum()), 'из', len(F))
# self-training (2 класса) на Kaggle = 0.84377

# финал: + крошечный кластер 2 (насыщение + крайние выбросы по правдоподобию)
g = GaussianMixture(2, random_state=SEED, n_init=5).fit(F[fit, :2])
loglik = g.score_samples(F[:, :2])
anom = np.zeros(len(F), bool)
anom[np.argsort(loglik)[:int(0.003 * len(F))]] = True
anom |= (X.min(1) <= 0)
labels = main.copy()
labels[anom] = 2
print('итог:', pd.Series(labels).value_counts().sort_index().to_dict())
# финальный сабмит на Kaggle = 0.84381 (3 кластера, 5 баллов)
