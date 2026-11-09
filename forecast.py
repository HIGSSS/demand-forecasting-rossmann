"""Прогноз ежедневных продаж магазинов на горизонт 14 дней (Rossmann Store Sales).

Запуск:
    python forecast.py            # данные в data/train.csv и data/store.csv
    python forecast.py --tune     # + подбор гиперпараметров по TimeSeriesSplit (дольше)
Результаты: results.md, forecast_vs_actual.png, feature_importance.png, error_analysis.png
"""
import argparse
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler

HORIZON = 14          # прогнозируем на 14 дней вперёд -> все лаги >= 14, утечки нет
TEST_DAYS = 6 * 7     # последние 6 недель = hold-out

# ---------- метрики ----------
def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def wape(y, p): return float(np.abs(y - p).sum() / np.abs(y).sum())
def report(y, p): return {"MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE": wape(y, p)}

# ---------- данные и признаки ----------
def load():
    df = pd.read_csv("data/train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv("data/store.csv")
    df = df.merge(store[["Store", "StoreType", "Assortment", "CompetitionDistance"]], on="Store", how="left")
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)

def make_features(df):
    df = df.copy()
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    for c in ["StateHoliday", "StoreType", "Assortment"]:
        df[c] = df[c].astype("category")
    df["year"], df["month"] = df.Date.dt.year, df.Date.dt.month
    df["day"], df["weekofyear"] = df.Date.dt.day, df.Date.dt.isocalendar().week.astype(int)
    g = df.groupby("Store")["Sales"]
    # лаги и скользящие средние сдвинуты минимум на HORIZON дней
    for lag in (14, 21, 28, 364):
        df[f"lag_{lag}"] = g.shift(lag)
    for w in (7, 28):
        df[f"roll_mean_{w}"] = g.transform(lambda s: s.shift(HORIZON).rolling(w, min_periods=3).mean())
    df["roll_std_28"] = g.transform(lambda s: s.shift(HORIZON).rolling(28, min_periods=3).std())
    return df

FEATURES = ["Store", "DayOfWeek", "Promo", "SchoolHoliday", "StateHoliday", "StoreType", "Assortment",
            "CompetitionDistance", "year", "month", "day", "weekofyear",
            "lag_14", "lag_21", "lag_28", "lag_364", "roll_mean_7", "roll_mean_28", "roll_std_28"]

PARAMS = dict(objective="regression_l1", n_estimators=600, learning_rate=0.05,
              num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=42, verbose=-1)

NO_LAG_FEATURES = [f for f in FEATURES if not f.startswith(("lag_", "roll_"))]

def tune(train, n_iter=8):
    """Случайный поиск гиперпараметров с валидацией по времени (только на train)."""
    space = {"num_leaves": [31, 63, 127, 255], "learning_rate": [0.03, 0.05, 0.1],
             "min_child_samples": [20, 50, 100], "colsample_bytree": [0.6, 0.8, 1.0],
             "subsample": [0.6, 0.8, 1.0], "n_estimators": [300, 600, 900]}
    tscv, best = TimeSeriesSplit(n_splits=3), (9, None)
    for params in ParameterSampler(space, n_iter=n_iter, random_state=42):
        w = []
        for tr_idx, va_idx in tscv.split(train):
            m = lgb.LGBMRegressor(**{**PARAMS, **params}).fit(train.loc[tr_idx, FEATURES], train.loc[tr_idx, "Sales"])
            w.append(wape(train.loc[va_idx, "Sales"].values, m.predict(train.loc[va_idx, FEATURES])))
        print(f"  {params} -> WAPE {np.mean(w):.4f}")
        if np.mean(w) < best[0]:
            best = (np.mean(w), params)
    print("Лучшие параметры:", best[1], f"(CV WAPE {best[0]:.4f})")
    return {**PARAMS, **best[1]}, best

def error_analysis(test):
    """Где модель ошибается сильнее: по дню недели, промо, праздникам, магазинам."""
    def by(col):
        g = test.groupby(col, observed=True).apply(
            lambda d: pd.Series({"WAPE": wape(d.Sales.values, d.pred.values), "rows": len(d)}), include_groups=False)
        return g.round(4)
    tables = {"День недели": by("DayOfWeek"), "Промо (Promo)": by("Promo"),
              "Школьные каникулы": by("SchoolHoliday"), "Государственный праздник": by("StateHoliday")}
    store = test.groupby("Store").apply(lambda d: wape(d.Sales.values, d.pred.values), include_groups=False)
    worst = store.sort_values(ascending=False).head(10).round(4).rename("WAPE").to_frame()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes, ["День недели", "Промо (Promo)", "Государственный праздник"]):
        tables[key]["WAPE"].plot.bar(ax=ax, title=f"WAPE: {key}"); ax.set_xlabel("")
    plt.tight_layout(); plt.savefig("error_analysis.png", dpi=130); plt.close()
    return tables, worst, store

def main(do_tune=False):
    df = make_features(load())
    df = df[(df.Open == 1) & (df.Sales > 0)].dropna(subset=["lag_14", "roll_mean_7"])
    cutoff = df.Date.max() - pd.Timedelta(days=TEST_DAYS)
    train, test = df[df.Date <= cutoff], df[df.Date > cutoff]
    train = train.sort_values("Date").reset_index(drop=True)
    print(f"train: {len(train):,} rows up to {cutoff.date()} | test: {len(test):,} rows")

    results = {}
    results["Baseline 1: Sales 28 дней назад (seasonal naive)"] = report(test.Sales, test["lag_28"].fillna(test.Sales.mean()))
    mean_store_dow = train.groupby(["Store", "DayOfWeek"]).Sales.mean().rename("p").reset_index()
    pred = test.merge(mean_store_dow, on=["Store", "DayOfWeek"], how="left")["p"].fillna(train.Sales.mean())
    results["Baseline 2: среднее по магазину и дню недели"] = report(test.Sales.values, pred.values)

    # абляция: что даёт история продаж (лаги)?
    m0 = lgb.LGBMRegressor(**PARAMS).fit(train[NO_LAG_FEATURES], train.Sales)
    results["LightGBM без лагов (только календарь и промо)"] = report(test.Sales.values, m0.predict(test[NO_LAG_FEATURES]))

    params, tuned_info = (PARAMS, None)
    if do_tune:
        params, tuned_info = tune(train)

    tscv = TimeSeriesSplit(n_splits=4)
    cv_wape = []
    for tr_idx, va_idx in tscv.split(train):
        m = lgb.LGBMRegressor(**{**params, "n_estimators": min(params["n_estimators"], 300)})
        m.fit(train.loc[tr_idx, FEATURES], train.loc[tr_idx, "Sales"])
        cv_wape.append(wape(train.loc[va_idx, "Sales"].values, m.predict(train.loc[va_idx, FEATURES])))
    print("TimeSeriesSplit WAPE по фолдам:", np.round(cv_wape, 4))

    model = lgb.LGBMRegressor(**params).fit(train[FEATURES], train.Sales)
    test = test.assign(pred=model.predict(test[FEATURES]))
    name = "LightGBM (лаги + календарь + промо)" + (", подобраны гиперпараметры" if do_tune else "")
    results[name] = report(test.Sales.values, test.pred.values)

    res = pd.DataFrame(results).T.round(4)
    tables, worst, store_wape = error_analysis(test)
    md = ["# Результаты (hold-out: последние 6 недель, горизонт 14 дней)\n", res.to_markdown(),
          f"\nTimeSeriesSplit (4 фолда), WAPE LightGBM: {np.round(cv_wape, 4).tolist()} (среднее {np.mean(cv_wape):.4f})\n"]
    if tuned_info:
        md.append(f"Подобранные гиперпараметры: `{tuned_info[1]}` (CV WAPE {tuned_info[0]:.4f})\n")
    md.append("\n## Анализ ошибок (hold-out)\n")
    for k, t in tables.items():
        md.append(f"**{k}**\n\n{t.to_markdown()}\n")
    md.append(f"**10 магазинов с наибольшей ошибкой**\n\n{worst.to_markdown()}\n")
    md.append(f"Медиана WAPE по магазинам: {store_wape.median():.4f}, максимум: {store_wape.max():.4f}\n")
    open("results.md", "w", encoding="utf-8").write("\n".join(md))
    print(res)

    day = test.groupby("Date")[["Sales", "pred"]].sum()
    ax = day.plot(figsize=(10, 4), title="Суммарные продажи: факт vs прогноз (hold-out)")
    ax.set_ylabel("Sales"); plt.tight_layout(); plt.savefig("forecast_vs_actual.png", dpi=130); plt.close()
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(figsize=(7, 6), title="Важность признаков (LightGBM)")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=130); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tune", action="store_true")
    main(ap.parse_args().tune)
