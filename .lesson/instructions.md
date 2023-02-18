# Homework MLR and logistic regression

## MLR

(1) Import `seaborn`, `numpy`, `pandas`, and `statsmodels.formula.api`

(2) Load the dataset "taxis" by using `sns.load_dataset("taxis")``

(3) Use the `assign` command and the `pd.to_datetime` command to transform the columns `dropoff` and `pickup` from strings to timestamps

(4) Create a column called duration which is the difference between dropoff and pickup time, and another one called `duration_min` which is the duration in minutes; hint:

```python
df = df.assign(
    duration=df.dropoff - df.pickup,
    duration_min=lambda x: x.duration.dt.total_seconds() / 60,
    unit_fare=lambda x: x.fare / (x.distance + 1e-6),
)
```

(5) Create a column called `unit_fare` which is the fare divided by the distance; when dividing, add a small number such as 0.001 to the denominator to avoid numerical problems

(6) Drop `nan`s from columns `total`, `unit_fare`, `distance`, `tip`, `tolls`, and `duration_min`

(7) Fit a multiple linear regression with those variables, and print the results

(8) Do the same but including the distance squared; hint:

```python
formula = "total ~ np.power(distance, 2) + distance + tip + tolls"
```

(9) Plot the dataset (you do not need to plot the model); hint:

```python
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
```

(10) Reflect on the results: which variables are important? What do you think of the p-values and the R squared? Which model works best? Is it intuitive? Write your reflections as comments in the code (place a `#` at the beginning of the line); be concise.


## Logistic regression

(1) Similarly to before, load the dataset "exercise"

(2) Assign a boolean variables called "no fat" that is one for no fat diets and zero for low fat diets; also, convert `pulse` to `float`; hint:

```python
df_logit = df_logit.assign(
    fat=(df_logit.diet == "no fat").astype(int),
    pulse=df_logit.pulse.astype(float)
)
```

(3) Plot the dataset using the command from the exercise above; you have some freedom, but you should use "fat" as variable `y` and "pulse" as variable `x`; there is no "right" setting here for `hue`, `kind`, etc.

(4) Fit a `logit` model with the following formula and print the summary of the results:

```python
formula = "fat ~ pulse + C(kind) + C(time)"
```

(5) Define a certain subject with a pulse of 120 is resting state after 15 minutes. How likely it is that this subject is on a zero fat diet? Hint:

```python
subject = pd.DataFrame([[120, "rest", "15 min"]], columns=['pulse', 'kind', 'time'])
y = results_logit.predict(subject)
```

(6) Reflect on the results: which variables are important? What do you think of the p-values and the R squared? Which model works best? Is it intuitive?

Write your reflections as comments in the code (place a `#` at the beginning of the line); be concise.

Do not worry about the underlying biology; maybe low fat is better for you than zero fat, but this is not a biology class :-)