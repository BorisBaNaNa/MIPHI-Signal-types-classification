"""Признаки формы и подбор окна PSD.
Хотел: получить PSD и убедиться, что он разделяет гамму/нейтроны как в методичке."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from _common import load, cumsum, SEED

d = load()
Yc, amplitude = d['Yc'], d['amplitude']
rng = np.random.default_rng(SEED)
C = cumsum(Yc)
area = lambda a, b: C[:, b] - C[:, a]
band = (amplitude >= 1500) & (amplitude <= 8000)   # чистая полоса для оценки


def psd_for(delay, end):
    tot = area(145, end)
    return area(150 + delay, end) / np.where(np.abs(tot) < 1, np.nan, tot)


def dprime(psd):
    # насколько разъехались два пика (расстояние/разброс по 1D-GMM)
    v = psd[band]; v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [.5, 99.5]); v = v[(v >= lo) & (v <= hi)].reshape(-1, 1)
    g = GaussianMixture(2, random_state=SEED, n_init=2).fit(v)
    m, var = g.means_.ravel(), g.covariances_.ravel()
    return abs(m[1] - m[0]) / np.sqrt(var.sum() / 2)


# перебор окон: хотел самое чёткое разделение двух пиков
rows = [(de, en, dprime(psd_for(de, en))) for de in [6, 10, 15, 20, 30] for en in [300, 350, 400]]
grid = pd.DataFrame(rows, columns=['delay', 'end', 'dprime']).sort_values('dprime', ascending=False)
print(grid.head())
de, en = int(grid.iloc[0].delay), int(grid.iloc[0].end)
print('лучшее окно: delay=%d end=%d' % (de, en))   # вышло 10/350

psd = psd_for(de, en)
area_total = area(145, en)
sel = rng.choice(np.where(np.isfinite(psd) & (amplitude > 0))[0], 8000, replace=False)

# две ветви: amplitude×area (цвет=PSD) и PSD×amplitude
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sc = ax[0].scatter(amplitude[sel], area_total[sel], c=psd[sel], s=5, alpha=.4, cmap='coolwarm',
                   vmin=np.nanpercentile(psd, 2), vmax=np.nanpercentile(psd, 98))
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_title('амплитуда×площадь (цвет=PSD)')
fig.colorbar(sc, ax=ax[0])
ax[1].scatter(amplitude[sel], psd[sel], s=5, alpha=.25)
ax[1].set_xscale('log'); ax[1].set_ylim(-.05, .45); ax[1].set_title('PSD×амплитуда')
plt.tight_layout(); plt.show()

# два пика PSD = два типа частиц
plt.figure(figsize=(10, 4))
for lo, hi in [(1000, 4000), (4000, 12000)]:
    s = (amplitude >= lo) & (amplitude < hi) & np.isfinite(psd)
    plt.hist(psd[s], bins=120, range=(-.05, .45), histtype='step', density=True, label='%d-%d' % (lo, hi))
plt.legend(); plt.title('распределение PSD: два пика'); plt.show()
# вывод: две чёткие ветви, на малых амплитудах сливаются. там будущие аномалии?,
# граница слегка наклонена следовательно простой порог по PSD не оптимален
