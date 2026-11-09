"""Генерирует синтетические данные в формате Kaggle Rossmann (для проверки, что код работает).
Реальные результаты для резюме считайте ТОЛЬКО на настоящих данных."""
import numpy as np, pandas as pd, os

rng = np.random.default_rng(42)
os.makedirs("data", exist_ok=True)
dates = pd.date_range("2013-01-01", "2015-07-31")
stores = np.arange(1, 41)
rows = []
for s in stores:
    base = rng.uniform(3000, 9000)
    dow_eff = rng.uniform(0.8, 1.2, 7)
    for d in dates:
        open_ = 0 if d.dayofweek == 6 and rng.random() < 0.95 else 1
        promo = int(d.dayofweek < 5 and (d.isocalendar().week % 2 == 0))
        season = 1 + 0.15 * np.sin(2 * np.pi * d.dayofyear / 365)
        sales = base * dow_eff[d.dayofweek] * season * (1 + 0.25 * promo) * rng.normal(1, 0.08) if open_ else 0
        rows.append((s, d.dayofweek + 1, d, max(int(sales), 0), open_, promo,
                     "0", int(rng.random() < 0.05)))
df = pd.DataFrame(rows, columns=["Store", "DayOfWeek", "Date", "Sales", "Open", "Promo",
                                 "StateHoliday", "SchoolHoliday"])
df.to_csv("data/train.csv", index=False)
pd.DataFrame({"Store": stores,
              "StoreType": rng.choice(list("abcd"), len(stores)),
              "Assortment": rng.choice(list("abc"), len(stores)),
              "CompetitionDistance": rng.integers(100, 20000, len(stores))}
             ).to_csv("data/store.csv", index=False)
print("saved data/train.csv, data/store.csv", df.shape)
