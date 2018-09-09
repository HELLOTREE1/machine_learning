#-*-coding:utf-8-*-

import pandas as pd

before = '/home/szu/PycharmProjects/WN_process_20180508/屈光手术前0529.xls'
after='/home/szu/PycharmProjects/WN_process_20180508/屈光手术后0529.xls'
dataBefore = pd.DataFrame(pd.read_excel(before))
B1_IOP=pd.DataFrame(dataBefore['CST_IOP(mm/hg)'])
print(B1_IOP)
dataAfter=pd.DataFrame(pd.read_excel(after))
B2_IOP = pd.DataFrame(dataAfter['CST_IOP(mm/hg)'])

import matplotlib.pyplot as plt #导入图像库
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

plt.figure() #建立图像
plt.subplot(2,2,1)
p = B1_IOP.boxplot(return_type='dict')
x = p['fliers'][0].get_xdata() # 'flies'即为异常值的标签
y = p['fliers'][0].get_ydata()
y.sort() #从小到大排序，该方法直接改变原对象
plt.subplot(2,2,2)#画箱线图，直接使用DataFrame的方法
q=B2_IOP.boxplot(return_type='dict')
x1 = q['fliers'][0].get_xdata() # 'flies'即为异常值的标签
y1 = q['fliers'][0].get_ydata()
y1.sort() #从小到大排序，该方法直接改变原对象

#以下参数都是经过调试的，需要具体问题具体调试。
for i in range(len(x)):
  if i>0:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
  else:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))

plt.show()