import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from category_encoders import TargetEncoder as taren
from sklearn.model_selection import RandomizedSearchCV

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

# Загружаем датасет
df = pd.read_csv("vehicles.csv")#Please read the README.md file before running the code.

print(df.sample(10))
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.nunique())

#убираем ненужные колонки и строки, где нет координат или описания
df = df.drop(columns=["url", "id", "region_url","image_url","county","size"])
df = df.dropna(subset=["lat","long","description"])
# print(df.info())

# Заменяем отрицательный пробег на пропуск

print((df["year"] <= 0).sum())
print((df["odometer"] <= 0).sum())
df.loc[df["odometer"] < 0, "odometer"] = np.nan

# Убираем слишком маленькие и слишком большие цены, которые могут быть ошибками

print((df["price"] > 1000000).sum())
df = df[df["price"] <= 900000]
df = df[df["price"] > 500]

# Разделяем данные на признаки и цену, после чего делим их на train и test

x = df.drop(columns=["price","VIN","cylinders"])
y = df["price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

df_corr = x_train.corr(numeric_only=True)
sns.heatmap(df_corr, annot=True, cmap="YlGnBu")
plt.show()

# Функция для заполнения пропусков с групировкой для большей точности используя данные из train

def fil_med(df,colum,train,method,group_1,group_2 = None):
    groups = [group_1] if group_2 is None else [group_1, group_2]
    if method == "num":
        med_mf_md = train.groupby(groups)[colum].median().reset_index()
        med_mf_md = med_mf_md.rename(columns={colum: "med_1"})
        med_mf = train.groupby(group_1)[colum].median().reset_index()
        med_mf = med_mf.rename(columns={colum: "med_2"})
        med = train[colum].median()
    if method == "cat":
        med_mf_md = train.groupby(groups)[colum].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
        med_mf_md = med_mf_md.rename(columns={colum: "med_1"})
        med_mf = train.groupby(group_1)[colum].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
        med_mf = med_mf.rename(columns={colum: "med_2"})
        med = train[colum].mode().iloc[0]

    df = df.merge(med_mf_md, on=groups, how="left")
    df[colum] = df[colum].fillna(df["med_1"])
    if df[colum].isnull().sum() != 0:
        df = df.merge(med_mf, on=group_1, how="left")
        df[colum] = df[colum].fillna(df["med_2"])
        if df[colum].isnull().sum() != 0:
            df[colum] = df[colum].fillna(med)
    df = df.drop(columns=["med_1", "med_2"],errors="ignore")
    return df

x_train = fil_med(x_train,"year", x_train, "num","manufacturer","model")
x_test = fil_med(x_test,"year", x_train, "num","manufacturer","model")

x_train = fil_med(x_train,"odometer", x_train, "num","manufacturer","model")
x_test = fil_med(x_test,"odometer", x_train, "num","manufacturer","model")

x_train = fil_med(x_train,"drive", x_train, "cat","manufacturer","model")
x_test = fil_med(x_test,"drive", x_train, "cat","manufacturer","model")

x_train = fil_med(x_train,"type", x_train, "cat","manufacturer","model")
x_test = fil_med(x_test,"type", x_train, "cat","manufacturer","model")

x_train = fil_med(x_train,"transmission", x_train, "cat","manufacturer","model")
x_test = fil_med(x_test,"transmission", x_train, "cat","manufacturer","model")

x_train = fil_med(x_train,"manufacturer", x_train, "cat","model","year")
x_test = fil_med(x_test,"manufacturer", x_train, "cat","model","year")

x_train = fil_med(x_train,"fuel", x_train, "cat","model","transmission")
x_test = fil_med(x_test,"fuel", x_train, "cat","model","transmission")

print(x_train["year"].isnull().sum())#0
print(x_test["year"].isnull().sum())#0
print(x_train["odometer"].isnull().sum())#0
print(x_test["odometer"].isnull().sum())#0

# проверка качества заполнения пропусков колонки manufacturer

test = x_train.groupby(["model", "year"])["manufacturer"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
test = test.rename(columns={"manufacturer": "med_1"})
sravne_1 = x_train.merge(test, on=["model", "year"], how="left")
print(((sravne_1["manufacturer"] == sravne_1["med_1"]).sum() / (sravne_1["med_1"].notnull().sum())) * 100) #99.6

# проверка качества заполнения пропусков колонки model

test = x_train.groupby(["type", "drive"])["model"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
test = test.rename(columns={"model": "med_1"})
sravne_2 = x_train.merge(test, on=["type", "drive"], how="left")
print(((sravne_2["model"] == sravne_2["med_1"]).sum() / (sravne_2["med_1"].notnull().sum())) * 100) #manufacturer + year = 16.0 / manufacturer + type = 23.1 / manufacturer + drive = 15.7 / manufacturer + transmission — 13.6
print(x_train.sample(10))
print(x_train.info())

# Пропуски в model заменяем на unknown

x_train["model"] = x_train["model"].fillna("unknown")
x_test["model"] = x_test["model"].fillna("unknown")

# проверка качества заполнения пропусков колонки fuel

test = x_train.groupby(["model", "drive"])["fuel"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
test = test.rename(columns={"fuel": "med_1"})
sravne_2 = x_train.merge(test, on=["model", "drive"], how="left")
print(((sravne_2["fuel"] == sravne_2["med_1"]).sum() / (sravne_2["med_1"].notnull().sum())) * 100) #manufacturer + model = 91.2 / model + transmission = 91.6 / model + drive = 91.4

# проверка столбца title_status

print(df["title_status"].unique())
print(df[df["title_status"] == "missing"].describe()) #2.619
print(df[df["title_status"].isna()].describe()) #11998.0

# Пропуски в title_status заменяем на unknown

x_train["title_status"] = x_train["title_status"].fillna("unknown")
x_test["title_status"] = x_test["title_status"].fillna("unknown")

# Создаём новый признак возраст машины на момент объявления

print(x_train["posting_date"].dtype)

x_train["age"] = (x_train["posting_date"].str[:4]).astype(int) - x_train["year"]
x_test["age"] = (x_test["posting_date"].str[:4]).astype(int) - x_test["year"]

print(x_train["age"].describe()) # -1
print((x_train["age"] < 0).sum())
print((x_test["age"] < 0).sum())
print(x_train[x_train["age"] < 0])
print(x_test[x_test["age"] < 0])

# Если получился отрицательный возраст заменяем его на 0

x_train.loc[x_train["age"] < 0, "age"] = 0
x_test.loc[x_test["age"] < 0, "age"] = 0

# Пропуски в paint_color заменяем на unknown

x_train["paint_color"] = x_train["paint_color"].fillna("unknown")
x_test["paint_color"] = x_test["paint_color"].fillna("unknown")

# убираем так как есть новый признак

x_train = x_train.drop(columns=["posting_date"])
x_test = x_test.drop(columns=["posting_date"])

# проверка качества заполнения пропусков колонки condition

test = x_train.groupby(["year"])["condition"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
test = test.rename(columns={"condition": "med_1"})
sravne_2 = x_train.merge(test, on=["year"], how="left")
print(((sravne_2["condition"] == sravne_2["med_1"]).sum() / (sravne_2["med_1"].notnull().sum())) * 100) #age = 33.5 / year = 33.5 / age + odometer = 79.2

print(x_train.describe())

# # Создаём новый признак група пробега

# x = [0, 20000, 50000, 100000, 150000, 200000, 300000, 500000, 1000000, np.inf]
# x_train["odometer_group"] = pd.cut(x_train["odometer"], x)
# x_test["odometer_group"] = pd.cut(x_test["odometer"], x)

# проверка качества заполнения пропусков колонки condition с новым признаком

test = x_train.groupby(["manufacturer","age"])["condition"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
test = test.rename(columns={"condition": "med_1"})
sravne_2 = x_train.merge(test, on=["manufacturer","age"], how="left")
print(((sravne_2["condition"] == sravne_2["med_1"]).sum() / (sravne_2["med_1"].notnull().sum())) * 100) #age + odometer_group = 36.8 / year odometer_group = 36.8 / manufacturer + age = 35.2 / manufacturer + odometer_group = 36.3

# Пропуски в condition заменяем на unknown

x_train["condition"] = x_train["condition"].fillna("unknown")
x_test["condition"] = x_test["condition"].fillna("unknown")

# убираем колонку с описанием так как пока не работаем с текстом

x_train = x_train.drop(columns=["description"])
x_test = x_test.drop(columns=["description"])

# # Создаём новый признак растояние в год

# x_train["dis_per_year"] = x_train["odometer"] / (x_train["age"] + 1)
# x_test["dis_per_year"] = x_test["odometer"] / (x_test["age"] + 1)



print(x_train.sample(10))
print(x_train.nunique())
print(x_train.info())
print(x_train.nunique())

# Убираем model, так как в этой колонке много разных и некорректных значений

x_train = x_train.drop(columns=["model"])
x_test = x_test.drop(columns=["model"])

# Сбрасываем индексы после удаления строк

x_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

print(x_train.shape[0] == y_train.shape[0])
print(x_test.shape[0] == y_test.shape[0])

# Разделяем признаки на числовые и категориальные

num = x_train.select_dtypes(include="number").columns
cat_one = x_train.select_dtypes(include="object").drop(columns=["region"]).columns # - "model",
cat_targ = x_train[["region"]].columns # - "model",

# Создаём обработку для разных типов признаков

prep = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(), cat_one),
    ("targ", taren(), cat_targ)])

# Функция для проверки моделей

def compare_models(X_train, y_train, X_test, y_test,prep):
    results = []
    models = [
    sk.linear_model.LinearRegression(),
    sk.linear_model.LassoCV(n_jobs=-1),
    sk.linear_model.RidgeCV(),
    sk.neighbors.KNeighborsRegressor(n_neighbors=16),
    sk.tree.DecisionTreeRegressor(max_depth=10, random_state=42),
    RandomForestRegressor(random_state=42,n_jobs=-1),
    GradientBoostingRegressor(),
    BaggingRegressor(random_state=42),
    xgb.XGBRegressor(objective="reg:squarederror", random_state=42,n_jobs=-1),]
    for model in models:
        pipe = Pipeline(steps=[
            ("preprocessor", prep),
            ("model", model)])
        print(type(model).__name__)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = sk.metrics.r2_score(y_test, preds)
        mae = sk.metrics.mean_absolute_error(y_test, preds)
        rmse = np.sqrt(sk.metrics.mean_squared_error(y_test, preds))

        results.append({"model": type(model).__name__,
                        "r2": r2,
                        "mae": mae,
                        "rmse": rmse,})
    results = pd.DataFrame(results)
    return results

res = compare_models(x_train, y_train, x_test, y_test,prep)
res = res.sort_values(by="mae", ascending=False)
print(res)

# 3               XGBRegressor  0.717106  3892.837646   8899.588305
# 2           BaggingRegressor  0.784065  2596.488838   7775.356686
# 1      RandomForestRegressor  0.789851  2436.723195   7670.476522


# проверка модели на логарифмированой цене

y_train_log = np.log1p(y_train)

pipe = Pipeline(steps=[
            ("preprocessor", prep),
            ("model", RandomForestRegressor(random_state=42,n_jobs=-1))])
pipe.fit(x_train, y_train_log)
preds_log = pipe.predict(x_test)
preds_log = np.expm1(preds_log)

r2 = sk.metrics.r2_score(y_test, preds_log)
mae = sk.metrics.mean_absolute_error(y_test, preds_log)
rmse = np.sqrt(sk.metrics.mean_squared_error(y_test, preds_log))
print(r2)
print(mae)
print(rmse)

# Находим объявления, где модель ошиблась сильнее всего

df2 = x_test.copy()
df2["eror"] = preds_log
df2["price"] = y_test.values
df2["dif"] = abs(df2["eror"] - df2["price"])
df2.sort_values(by="dif", ascending=False, inplace=True)
print(df2.head(50))

# 0.7168291747129448
# 2680.7534077120017
# 8903.948828663068

#Создаём базовую модель Random Forest и оцениваем её качество.
#После сравнения с другими моделями именно эта конфигурация показала лучший результат, поэтому далее используем её как основную модель

# Создаём обработку для разных типов признаков

prep = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_one),
    ("targ", taren(), cat_targ)])

# Объединяем обработку данных и Random Forest в один Pipeline

pipe = Pipeline(steps=[
            ("preprocessor", prep),
            ("model", RandomForestRegressor(random_state=42,n_jobs=-1))])

# Обучаем модель и делаем предсказания

pipe.fit(x_train, y_train)
preds = pipe.predict(x_test)

# Считаем основные метрики модели

r2 = sk.metrics.r2_score(y_test, preds)
mae = sk.metrics.mean_absolute_error(y_test, preds)
rmse = np.sqrt(sk.metrics.mean_squared_error(y_test, preds))
print(r2)
print(mae)
print(rmse)

# Находим объявления, где модель ошиблась сильнее всего

df2 = x_test.copy()
df2["eror"] = preds
df2["price"] = y_test.values
df2["dif"] = abs(df2["eror"] - df2["price"])
df2.sort_values(by="dif", ascending=False, inplace=True)
print(df2.head(50))

# Смотрим, как меняется ошибка модели в разных ценовых диапазонах

bins = [0, 25000, 50000, 100000, 150000, np.inf]
labels = ["0–25k", "25–50k", "50–100k", "100–150k", "150k+"]

error_df = pd.DataFrame({"real": y_test.values, "pred": preds})
error_df["error"] = abs(error_df["real"] - error_df["pred"])
error_df["price_group"] = pd.cut(error_df["real"], bins=bins, labels=labels)
result = error_df.groupby("price_group", observed=True).agg(MAE=("error", "mean"),count=("error", "size"))

print(result)

# проверка влияния признаков на модель

importance = permutation_importance(pipe, x_test, y_test,scoring="r2",random_state=42,n_repeats=5, n_jobs=-1)
importance_df = pd.DataFrame({
    "feature": x_test.columns,
    "importance": importance.importances_mean})
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False)
print(importance_df)

# проверка колони model = ford так как в ней я увидел очень много ошибочных данных

model_count = df[df["manufacturer"] == "ford"]["model"].value_counts().reset_index()
model_count.columns = ["model", "count"]
print(model_count)

# подбор параметров

param = { "model__n_estimators": [200, 300, 500, 700, 900, 1200],
                        "model__max_depth": [None, 10, 15, 20, 25, 30, 40],
                        "model__min_samples_split": [2, 5, 10, 20],
                        "model__min_samples_leaf": [1, 2, 4, 8, 12],
                        "model__max_features": [0.3, 0.5, 0.7, 1.0, "sqrt"],
                        "model__bootstrap": [True, False]}

# Проверяем получится ли улучшить Random Forest подбором параметров

search = RandomizedSearchCV(pipe,param_distributions=param,n_iter=3,scoring="neg_mean_absolute_error",cv=3,random_state=42,n_jobs=1,verbose=2)

search.fit(x_train, y_train)
preds_2 = search.predict(x_test)

r2 = sk.metrics.r2_score(y_test, preds_2)
mae = sk.metrics.mean_absolute_error(y_test, preds_2)
rmse = np.sqrt(sk.metrics.mean_squared_error(y_test, preds_2))
print(r2)
print(mae)
print(rmse)
print(search.best_score_)
print(search.best_params_)

#RandomizedSearchCV не дал улучшения: лучшая комбинация показала R² ≈ 0.80 и MAE ≈ $2889,поэтому оставляем предыдущую модель с R² ≈ 0.83 и MAE ≈ $2400.































