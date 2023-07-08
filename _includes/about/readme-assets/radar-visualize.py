# -*- coding: utf-8 -*-
import pandas as pd
import pyecharts
import pyecharts.charts as pyc
import pyecharts.options as opts
import pyecharts.globals as glbs
from pyecharts.commons.utils import JsCode

df = pd.DataFrame(
    {
        'Filed': [
            '业务思维\nBusiness Thinking', 
            '理论知识\nTheoretical Knowledge', 
            '分析技能\nTechnical Ability', 
            '团队合作\nTeam Working', 
            '项目经验\nProject Experience', 
            '学习能力\nLearning Capacity'
        ],
        'Score': [7.8, 6.9, 8.2, 8.0, 5.9, 8.9]
    }
)

radar = pyc.Radar(
    init_opts=opts.InitOpts(width='100%', height='360px', theme=glbs.ThemeType.DARK, bg_color='#1a1c1d')
).add_schema(
    schema=[{'name': filed, 'max': 10, 'min': 0} for filed in df.Filed],
    # angleaxis_opts=opts.AngleAxisOpts(start_angle=30)
    splitline_opt=opts.SplitLineOpts(
        is_show=True,
        linestyle_opts=opts.LineStyleOpts(width=0.6)
    )
).add(
    series_name='评分',
    data=[{'name': df.Score.name, 'value': [float(i) for i in df.Score.values]}],
    areastyle_opts=opts.AreaStyleOpts(opacity=0.2),
    linestyle_opts=opts.LineStyleOpts(width=2),
    color='#3d7afd'
).set_series_opts(
    label_opts=opts.LabelOpts(is_show=False)
).set_global_opts(
    legend_opts=opts.LegendOpts(is_show=False)
)
radar.render()

# js_0= 'https://cdn.jsdelivr.net/npm/echarts@latest/dist/'
# js_1 = 'https://assets.pyecharts.org/assets/'
# with open('render.html', 'r', encoding='utf-8') as f:
#     html = f.read().replace(js_0, js_1)
# with open('render.html', 'w', encoding='utf-8') as f:
#     f.write(html)
