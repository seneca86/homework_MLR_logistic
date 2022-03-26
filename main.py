# %%
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# %%
df = sns.load_dataset("taxis")
# %%
df = df.assign(
    dropoff=pd.to_datetime(df.dropoff),
    pickup=pd.to_datetime(df.pickup)
)
# %%
df = df.assign(
    duration=df.dropoff - df.pickup,
    duration_min=lambda x: x.duration.dt.total_seconds() / 60,
    unit_fare = lambda x: x.fare / (x.distance + 1e-6),
)
# %%
df = df.dropna(subset=['total', 'unit_fare', 'distance', 'tip', 'tolls', 'duration_min'])
# %%
formula = 'total ~ unit_fare + distance + tip + tolls + duration_min'
results = smf.ols(formula, data=df).fit()
print(results.summary())
# %%
formula = 'total ~ distance + tip + tolls + duration_min'
results = smf.ols(formula, data=df).fit()
print(results.summary())
# %%
