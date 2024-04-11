import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from numpy import mean, std

# 加载数据
df = pd.read_csv('/Users/daydream/Desktop/data of RFSI.csv')

# 准备数据 - 假设 'x', 'y', 和 'ThyroidV_1' 是特征
X = df[['x', 'y', 'ThyroidV_1']]  # 更新特征集
y = df['UrineIod_1']  # 目标变量

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建随机森林回归器对象
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# 使用 X_train 和 y_train 训练回归器
regressor.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = regressor.predict(X_test)

# 计算均方根误差 (RMSE) 和 R² 分数
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'均方根误差: {rmse}')
print(f'R² 分数: {r2}')

# 使用交叉验证来评估模型
cv_rmse = cross_val_score(regressor, X, y, scoring='neg_root_mean_squared_error', cv=5)
cv_r2 = cross_val_score(regressor, X, y, scoring='r2', cv=5)

print(f'交叉验证 RMSE: 均值 = {-mean(cv_rmse)}, 标准差 = {std(cv_rmse)}')
print(f'交叉验证 R²: 均值 = {mean(cv_r2)}, 标准差 = {std(cv_r2)}')
