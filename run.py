import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# 加载数据
df = pd.read_csv('/Users/daydream/Desktop/data of RFSI.csv')

# 准备数据 - 假设 'x' 和 'y' 是特征，'UrineIod_1' 是你想要预测的目标变量
X = df[['x', 'y']]  # 特征
y = df['UrineIod_1']  # 目标变量

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建随机森林回归器对象
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# 使用 X_train 和 y_train 训练回归器
regressor.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = regressor.predict(X_test)

# 计算均方根误差
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'均方根误差: {rmse}')

# 假设你有一个包含缺失值的DataFrame，名为 df_missing
# 加载包含缺失值的数据
df_missing = pd.read_csv('/Users/daydream/Desktop/missing value.csv')

# 准备缺失值数据
missing_values = df_missing[['x', 'y']]

# 使用回归器预测缺失值
predicted_values = regressor.predict(missing_values)

# 查看df_missing的前几行确保数据加载正确
print(df_missing.head())

# 打印预测值
print(predicted_values)

print(df_missing[['x', 'y', 'UrineIod_1']].head())


# 回填预测值
df_missing['UrineIod_1'] = predicted_values

# 查看回填后的df_missing的前几行
print(df_missing.head())

# 保存df_missing到CSV文件
df_missing.to_csv('/Users/daydream/Desktop/missing_values_filled.csv', index=False)



