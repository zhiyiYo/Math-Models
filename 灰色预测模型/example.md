# 某地区年平均降雨量数据如下表所示：
![表格](resource\某地区年平均降雨量数据.jpg)

规定 $\xi$=320，并认为 $x^{(0)}(i)\le \xi$ 为旱灾，请预测下一次旱灾发生的时间

```python
import numpy as np
import matplotlib.pyplot as plt
from gray_forecast_model import GrayForecastModel

plt.style.use('matlab')
plt.rcParams['xtick.minor.visible'] = False

# 旱灾发生时间
X_0 = [3, 8, 10, 14, 17]
GM = GrayForecastModel(X_0)
print(f'下一次旱灾发生的时间在{(GM.forecast(1, False)[-1] - X_0[-1]):.2f}年后')
```

# 菌落生长曲线预测
```python
origin_data = [0.025, 0.023, 0.029, 0.044, 0.084, 0.164, 0.332, 0.521, 0.97, 1.6,
                2.45, 3.11, 3.57, 3.76, 3.96, 4, 4.46, 4.4, 4.49, 4.76, 5.01]
hour = np.arange(0, 21)
hour_ = np.arange(0, 31)
GM.setModel(origin_data, GM.VERHULST)
GM.forecast(10, True, hour, hour_, xlabel='Hour/h', ylabel='Hour/h',ylim=(0,5.2),xlim=(0,31))
print('后面的预测值为：', GM.forecast_series[len(origin_data):])
plt.legend(['actual num', 'predict num'])
_ = plt.xticks(hour_, [str(i) for i in hour_])
```

# GM(1, N) 预测
```python
org_data = [[560823, 542386, 604834, 591248, 583031, 640636,
             575688, 689637, 570790, 519574, 614677],
            [104, 101.8, 105.8, 111.5, 115.97, 120.03,
             113.3, 116.4, 105.1, 83.4, 73.3],
            [135.6, 140.2, 140.1, 146.9, 144, 143,
             133.3, 135.7, 125.8, 98.5, 99.8],
            [131.6, 135.5, 142.6, 143.2, 142.2,
             138.4, 138.4, 135, 122.5, 87.2, 96.5],
            [54.2, 54.9, 54.8, 56.3, 54.5, 54.6,
             54.9, 54.8, 49.3, 41.5, 48.9]]
GM.setModel(org_data, GM.GM_1_N)
GM.forecast(2, True, np.arange(0, 11), np.arange(
        0, 13), xlabel='Hour/h', ylabel='Hour/h')
```