import pandas as pd
from pyecharts import charts as pyc
from pyecharts import options as opts
from pyecharts import globals as glbs

data = pd.read_csv('data.csv', sep='\t')
line = pyc.Line(init_opts=opts.InitOpts(theme=glbs.ThemeType.DARK, width='100%')
).add_xaxis(
    list(data.iloc[:,0])
).add_yaxis(
    'Beijing AQI', list(data.iloc[:,1])
).set_series_opts(
    label_opts=opts.LabelOpts(is_show=False), 
    markline_opts=opts.MarkLineOpts(
        label_opts=opts.LabelOpts(position='end'),
        linestyle_opts=opts.LineStyleOpts(color='#333'),
        data=[{'yAxis': y} for y in [50, 100, 150, 200, 300]]
    )
).set_global_opts(
    title_opts=opts.TitleOpts(title='Beijing AQI', pos_left='1%'), 
    legend_opts=opts.LegendOpts(is_show=False),
    visualmap_opts=opts.VisualMapOpts(
        is_piecewise=True, pos_top=50, pos_right=10,
        pieces=[
            {'min':0, 'max': 50, 'color': '#93CE07'},
            {'min':50, 'max': 100, 'color': '#FBDB0F'},
            {'min':100, 'max': 150, 'color': '#FC7D02'},
            {'min':150, 'max': 200, 'color': '#FD0100'},
            {'min':200, 'max': 300, 'color': '#AA069F'},
            {'min':300, 'color': '#AC3B2A'}
        ]
    ), 
    datazoom_opts=[
        opts.DataZoomOpts(
            start_value='2014-06-01', range_start=None, range_end=None
        ), 
        opts.DataZoomOpts(type_='inside')
    ], 
    tooltip_opts=opts.TooltipOpts(trigger='axis'),
    toolbox_opts=opts.ToolboxOpts(
        pos_left=None, pos_right=10,
        feature={'dataZoom': {}, 'restore': {}, 'saveAsImage': {}}
    ),
    yaxis_opts=opts.AxisOpts(
        splitline_opts=opts.SplitLineOpts(is_show=True)
    )
)

line.options['grid'] = {'left': '6%', 'right': '15%', 'bottom': '12%'}
line.render()
