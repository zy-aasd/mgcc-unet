import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['ConvStem-Encoder', 'MGFEM',
           'ConvStem-Decoder', 'MS-UPB','MGIIM']
sizes = [10.6, 0.754, 5.292, 8.433, 0.205]
colors = ['#FF6B6B', '#FFA726', '#FFCA28', '#42A5F5', '#66BB6A']

# 自定义显示函数
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = pct * total / 100.0
        return f'{val:.2f}M\n({pct:.1f}%)'
    return my_autopct

# 创建环形图
fig, ax = plt.subplots(figsize=(11, 8))


# 外环
wedges, texts, autotexts = ax.pie(sizes,
                                  labels=labels,
                                  colors=colors,
                                  autopct=make_autopct(sizes),
                                  startangle=90,
                                  pctdistance=0.7,  # 调整百分比文字更靠近中心
                                  labeldistance=1.05,  # 调整标签文字位置
                                  textprops={'fontsize': 16})


# 内环（白色圆形）
centre_circle = plt.Circle((0,0),0.45,fc='white')
fig.gca().add_artist(centre_circle)

# 美化
for autotext in autotexts:
    autotext.set_color('black')


ax.axis('equal')
plt.tight_layout()
#plt.show()
plt.savefig('pie_params.png')