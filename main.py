# %%
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# %%
df = sns.load_dataset("taxis")
# %%
df = df.assign(dropoff=pd.to_datetime(df.dropoff), pickup=pd.to_datetime(df.pickup))
# %%
df = df.assign(
    duration=df.dropoff - df.pickup,
    duration_min=lambda x: x.duration.dt.total_seconds() / 60,
    unit_fare=lambda x: x.fare / (x.distance + 1e-6),
)
# %%
df = df.dropna(
    subset=["total", "unit_fare", "distance", "tip", "tolls", "duration_min"]
)
# %%
formula = "total ~ unit_fare + distance + tip + tolls + duration_min"
results = smf.ols(formula, data=df).fit()
print(results.summary())
# %%
formula = "total ~ np.power(distance, 2) + distance + tip + tolls"
results = smf.ols(formula, data=df).fit()
print(results.summary())

# %%
from pathlib import Path

Path("plots").mkdir(parents=True, exist_ok=True)
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 150
sns.set_theme()
scatter_mlr = sns.relplot(
    data=df,
    x="distance",
    y="total",
    style=None,
    palette="deep",
    markers=None,
    col="color",
    kind="scatter",
    hue="pickup_borough",
)
scatter_mlr.savefig("plots/scatter_mlr.png")
plt.clf()
# %%
df_logit = sns.load_dataset("exercise")
df_logit = df_logit.assign(
    fat=(df_logit.diet == "no fat").astype(int), pulse=df_logit.pulse.astype(float)
)
# %%
scatter_logit = sns.relplot(
    data=df_logit,
    x="pulse",
    y="fat",
    style=None,
    palette="deep",
    markers=None,
    row="kind",
    kind="scatter",
    hue="time",
)
scatter_logit.savefig("plots/scatter_logit.png")
plt.clf()
# %%
formula = "fat ~ pulse + C(kind) + C(time)"
model_logit = smf.logit(formula, data=df_logit)
results_logit = model_logit.fit()
results_logit.summary()
# %%
subject = pd.DataFrame([[120, "rest", "15 min"]], columns=["pulse", "kind", "time"])
y = results_logit.predict(subject)
print(f"The chances of this person being on a no fat diet are {y[0]}")
# %%
