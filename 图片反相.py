import os
from PIL import Image


path = '灰色预测模型\\resource\\某地区年平均降雨量数据.png'
img = Image.open(path) #type:Image.Image
img = img.convert('RGB').point(lambda x: 255 - x)
img.save(os.path.splitext(path)[0]+'.jpg')