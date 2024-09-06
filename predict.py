import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
def generate_lag(data, months, lag_column):
    '''Функция добавляет лаги в целевой признак. Принимает на вход датасет, величину сдвига и сам целевой признак'''
    for month in months:
        data_shift = data[['date_block_num', 'shop_id', 'item_id', lag_column]].copy()
        data_shift.columns = ['date_block_num', 'shop_id', 'item_id', lag_column + '_lag_' + str(month)]
        data_shift['date_block_num'] += month
        data = pd.merge(data, data_shift, on=['date_block_num', 'shop_id', 'item_id'], how='left')
    return data

new_sales = pd.read_csv('new_sales.csv')
print(new_sales.head())

#целевой лаг
new_sales = generate_lag(new_sales, [1,2,3,6,12], 'item_cnt_month')

#cдвиг среднемесячных продаж по товарам
group = new_sales.groupby(['date_block_num', 'item_id'])['item_cnt_month'].mean().rename('item_month_mean').reset_index()
new_sales = pd.merge(new_sales, group, on=['date_block_num', 'item_id'], how='left')
new_sales = generate_lag(new_sales, [1,2,3,6,12], 'item_month_mean')
new_sales.drop(['item_month_mean'], axis=1, inplace=True)

#cдвиг среднемесячных продаж по магазинам
group = new_sales.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().rename('shop_month_mean').reset_index()
new_sales = pd.merge(new_sales, group, on=['date_block_num', 'shop_id'], how='left')
new_sales = generate_lag(new_sales, [1,2,3,6,12], 'shop_month_mean')
new_sales.drop(['shop_month_mean'], axis=1, inplace=True)

#cдвиг среднемесячных продаж по категориям
group = new_sales.groupby(['date_block_num', 'cat_code'])['item_cnt_month'].mean().rename('cat_code_month_mean').reset_index()
new_sales = pd.merge(new_sales, group, on=['date_block_num', 'cat_code'], how='left')
new_sales = generate_lag(new_sales, [1,2,3,6,12], 'cat_code_month_mean')
new_sales.drop(['cat_code_month_mean'], axis=1, inplace=True)

#cдвиг среднемесячных продаж по магазинам/категориям
group = new_sales.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].mean().rename('shop_category_month_mean').reset_index()
new_sales = pd.merge(new_sales, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
new_sales = generate_lag(new_sales, [1,2,3,6,12], 'shop_category_month_mean')
new_sales.drop(['shop_category_month_mean'], axis=1, inplace=True)

#заполнение нулями значений None
new_sales.fillna(0, inplace=True)

#тренировочная выборка
x_train = new_sales[new_sales.date_block_num < 33].drop(['item_cnt_month'], axis=1)
y_train = new_sales[new_sales.date_block_num < 33]['item_cnt_month']

#валидационная выборка
x_valid = new_sales[new_sales.date_block_num == 33].drop(['item_cnt_month'], axis=1)
y_valid = new_sales[new_sales.date_block_num == 33]['item_cnt_month']

#тестовая выборка
x_test = new_sales[new_sales.date_block_num == 34].drop(['item_cnt_month'], axis=1)

model = xgb.XGBRegressor(
    n_estimators = 1000,
    learning_rate = 0.1,
    max_depth = 10,
    subsample = 0.5,
    colsample_bytree = 0.5)

model.fit(
    x_train,
    y_train,
    eval_metric='rmse',
    eval_set=[(x_train, y_train),
               (x_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=10,
          )

#прогноз для оценочного набора данных
predictions = model.predict(x_test).clip(0,20)

print("Accuracy:", model.best_score)

plt.figure(figsize=(25,10))

#прогноз для валидационного набора данных
pred_val = model.predict(x_valid)

#построение графика
sns.lineplot(x=x_valid.index, y=pred_val, color="#00ffbf", label='Predicted')
sns.lineplot(x=x_valid.index, y=y_valid, alpha=0.6, color="#8f17ff", label='Actual')
#добавляем легенду для различения линий
plt.legend()

#отображение графика
plt.show()

fig, ax = plt.subplots(1,1,figsize=(10,14))
xgb.plot_importance(model, ax=ax)

#изменение цвета столбцов
for patch in ax.patches:
    patch.set_color('#8f17ff')

plt.show()

#сохранение результата прогноза в файл
test = pd.read_csv('test.csv')
result = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": predictions
})
result.to_csv('result.csv', index=False)