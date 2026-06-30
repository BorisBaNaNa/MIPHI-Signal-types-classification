"""EDA: хотел понять, как устроен сигнал, где baseline и пик, есть ли аномалии."""
import numpy as np
import matplotlib.pyplot as plt
from _common import load, SEED

d = load()
X, Yc = d['X'], d['Yc']
amplitude, peak_index = d['amplitude'], d['peak_index']
rng = np.random.default_rng(SEED)

print('сигналов:', X.shape)
print('baseline медиана %.0f, шум %.2f' % (np.median(d['baseline']), np.median(d['noise'])))
print('пик: медиана %.0f (1%%=%.0f, 99%%=%.0f)' %
      (np.median(peak_index), np.percentile(peak_index, 1), np.percentile(peak_index, 99)))
# вывод: baseline стабилен, пик всегда ~150 -> сигналы синхронизированы, выравнивать не надо

# средняя форма импульса
bright = amplitude > np.percentile(amplitude, 90)
plt.figure(figsize=(11, 4))
plt.plot(Yc.mean(0), label='все')
plt.plot(Yc[bright].mean(0), label='яркие (топ-10%)')
plt.title('средняя форма импульса'); plt.legend(); plt.show()

# случайные сигналы. видно, что хвост у всех разный
idx = rng.choice(len(X), 6, replace=False)
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, i in zip(axes.ravel(), idx):
    ax.plot(Yc[i], lw=.8); ax.set_title('#%d amp=%.0f' % (i, amplitude[i]))
plt.tight_layout(); plt.show()

# экстремальные по амплитуде: сверху самые большие, снизу шумовые
big, small = np.argsort(amplitude)[-3:], np.argsort(amplitude)[:3]
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for ax, i in zip(axes[0], big):
    ax.plot(Yc[i], color='r'); ax.set_title('big amp=%.0f' % amplitude[i])
for ax, i in zip(axes[1], small):
    ax.plot(Yc[i], color='b'); ax.set_title('small amp=%.0f' % amplitude[i])
plt.tight_layout(); plt.show()

# распределения и насыщение
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(amplitude, bins=200); axes[0].set_yscale('log'); axes[0].set_title('амплитуда')
axes[1].hist(d['noise'], bins=100); axes[1].set_title('шум')
axes[2].hist(peak_index, bins=60); axes[2].set_title('пик')
plt.tight_layout(); plt.show()
print('насыщенных (срез АЦП):', int((X.min(1) <= 0).sum()))
# вывод: на краю амплитуды отдельный всплеск = насыщение АЦП (9 шт) это кандидаты в аномалии
