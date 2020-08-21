# -*- coding: utf-8 -*-  
# 热力图
import random
import csv
import sys
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/Library/Fonts/Songti.ttc')
# https://blog.csdn.net/coder_Gray/article/details/81867639
# https://blog.csdn.net/weixin_39541558/article/details/79813936
def draw(trace, xLabel, yLabel, data):
    #定义热图的横纵坐标
    # xLabel = ['A','B','C','D','E']
    # yLabel = ['1','2','3','4','5']

    # #准备数据阶段，利用random生成二维数据（5*5）
    # data = []
    # for i in range(5):
    #     temp = []
    #     for j in range(5):
    #         k = random.randint(0,100)
    #         temp.append(k)
    #     data.append(temp)

    #作图阶段
    fig = plt.figure()
    #定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    #定义横纵坐标的刻度
    ax.set_yticks(range(len(yLabel)))
    ax.set_ylabel('size rate', fontproperties=font, fontsize=20)
    ax.set_yticklabels(yLabel, fontproperties=font, fontsize=18)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel, fontproperties=font, fontsize=18)
    ax.set_xlabel('p', fontproperties=font, fontsize=20)
    #作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(data, cmap=plt.cm.bone_r)
    #增加右侧的颜色刻度条
    plt.colorbar(im)
    #增加标题
    plt.title(trace, fontproperties=font, fontsize=20)
    #show
    fig.subplots_adjust(bottom=0.15)
    # plt.show()
    # fig.set_size_inches(1, 4) 
    plt.savefig(trace + '.png', dpi=300)

def generate_data(xpLabel, ysizeLabel, mydict):
    
    data = []
    for size in ysizeLabel:
        l = []
        for p in xpLabel:
            l.append(mydict[(p,size)])
        data.append(l)
    print xpLabel
    print ysizeLabel
    print data
    return data

# load file
d = {}
with open('test.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        print row
        trace = row[0]
        if trace in d:
            (xpLabel, ysizeLabel, mydict) = d[trace]
        else:
            xpLabel = [] 
            ysizeLabel = []
            mydict = {}
        xp = float(row[1])
        ysize = float(row[2])
        # value = int(1000*float(row[4]))
        value = float(row[4])
        if xp not in xpLabel:
            xpLabel.append(xp)
        if ysize not in ysizeLabel:
            ysizeLabel.append(ysize)
        # vhitLabel.append(row[4])
        mydict[(xp, ysize)] = value
        d[trace] = (xpLabel, ysizeLabel, mydict)
        # print(d)

csvfile.close()
for trace in d:
    (xpLabel, ysizeLabel, mydict) = d[trace]
    print(mydict)
    xpLabel.sort()
    ysizeLabel.sort(reverse=True)
    data = generate_data(xpLabel, ysizeLabel, mydict)
    draw(trace, xpLabel, ysizeLabel, data)
    # sys.exit(-1)





