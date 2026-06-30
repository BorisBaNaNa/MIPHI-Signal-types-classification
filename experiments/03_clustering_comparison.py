"""Сравнение подходов к кластеризации.
Хотел проверить, можно ли получить 3 кластера в лоб - оказалось что нет."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from _common import load, cumsum, SEED

d = load()
X, Yc, amplitude, snr = d['X'], d['Yc'], d['amplitude'], d['snr']
C = cumsum(Yc)
area = lambda a, b: C[:, b] - C[:, a]
tot = area(145, 350)
psd = area(160, 350) / np.where(np.abs(tot) < 1, np.nan, tot)
log_amp = np.log10(np.clip(amplitude, 1, None))
aoa = tot / np.where(amplitude < 1, np.nan, amplitude)
rng = np.random.default_rng(SEED)


def metrics(Z, lab):
    return (silhouette_score(Z[:5000], lab[:5000]),
            calinski_harabasz_score(Z, lab), davies_bouldin_score(Z, lab))


# 1) наивно: PCA по сырым отсчётам -> KMeans
pca = PCA(20, random_state=SEED).fit_transform(StandardScaler().fit_transform(X))
for k in (2, 3):
    lab = KMeans(k, random_state=SEED, n_init=10).fit_predict(pca)
    print('raw PCA k=%d:' % k, '%.3f %.0f %.3f' % metrics(pca, lab))
# на Kaggle ~0.51: PCA ловит амплитуду, режет на большие/маленькие, а не на гамму/нейтрон

# 2) признаки формы, прямое k=3
psd_f = np.nan_to_num(psd, nan=np.nanmedian(psd))
Z = StandardScaler().fit_transform(
    np.column_stack([log_amp, psd_f, np.nan_to_num(aoa, nan=np.nanmedian(aoa))]))
for name, m in [('KMeans3', KMeans(3, random_state=SEED, n_init=10)),
                ('GMM3', GaussianMixture(3, random_state=SEED, n_init=5))]:
    lab = m.fit_predict(Z)
    print(name, '%.3f %.0f %.3f' % metrics(Z, lab))
# тоже плохо: третий кластер не выделяется, разбиение уходит в амплитуду

# 3) двухэтапно: GMM2 на надёжных + слабые в кластер 2
Z2 = StandardScaler().fit_transform(np.column_stack([log_amp, psd_f]))
rel = snr > np.quantile(snr, 0.10)
g = GaussianMixture(2, random_state=SEED, n_init=5).fit(Z2[rel])
lab = np.full(len(X), 2)
lab[rel] = g.predict(Z2[rel])
sel = rng.choice(len(X), 9000, replace=False)
plt.figure(figsize=(8, 6))
plt.scatter(amplitude[sel], np.nan_to_num(psd[sel], nan=-.02), c=lab[sel], s=6, alpha=.5, cmap='tab10')
plt.xscale('log'); plt.ylim(-.05, .45)
plt.title('двухэтапно: две ветви + слабые в кластер 2'); plt.show()
# вывод: только двухэтапная схема даёт физичное разбиение. кластер 2 нельзя получать вместе с двумя основными
