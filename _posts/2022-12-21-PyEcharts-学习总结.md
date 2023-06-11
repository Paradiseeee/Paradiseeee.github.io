---
layout:     post
title:      "PyEcharts 学习总结"
subtitle:   "如何用 pyecharts 绘制一切你想象得到的交互式图表？"
date:       2022-12-21 12:00:00
author:     "Paradise"
header-img: "post-assets/20221221/header.jpg"
header-mask: 0.6
tags:
    - Python
    - 数据可视化
    - 总结
    - RCM
---

> pyecharts 分为 `v0.5.x` 和 `v1.0.0+` 两个版本，这篇总结针对新版本。

> 本文不包含基础例子的演示，主要记录一些细节方面的理解。简介中推荐的网站有相关示例。

## 一、简介 -- Echarts & PyEcharts

[Echarts](https://github.com/apache/echarts) 是一个基于 JavaScript 的图表库，用于在 HTML 中生成可交互的图表。关于更多 js 图表库可以参考 CSDN 上的 [这篇博客](https://blog.csdn.net/Pokemogo/article/details/78864664)。[PyEcharts](https://github.com/pyecharts/pyecharts) 旨在提供在 Python 中使用 Echarts 的 API，以便将数据可视化的流程整合到 Python 数据处理的流程当中。

**以下是四个文档和示例网站，各有特点。利用好这几个网站基本上就可以解决所有可能遇到的问题，而不需要在搜索引擎中漫无目的地翻查。**（喜欢这篇文章的新手朋友，建议先撸一遍下面 gallery.pyecharts.org 里面的代码再来看，因为文章写得不好，入门者很可能一头雾水）

- **Echarts**
    - <https://www.w3cschool.cn/echarts_tutorial>   （配置项的文档非常详尽）
    - <https://echarts.apache.org/zh/index.html>    （[示例](https://echarts.apache.org/examples/zh/index.html)中有很多出色的可视化例子）
- **PyEcharts**
    - <https://pyecharts.org/#/zh-cn/intro>         （PyEcharts 的 API 文档）
    - <https://gallery.pyecharts.org/#/README>      （PyEcharts 绘图的示例和代码）

### Echarts

首先通过一个简单的例子了解 Echarts 的工作方式，注意到使用 Echarts 需要 HTML 语言基础，进阶则需要 js 和 css 的基础。（如果只是在 Python 中进行可视化，对前面三件套的要求不高，基本上看看例子就能摸索出来写法，编程语言都是共通的。）

```html
<!DOCTYPE html>
<html>
<head>
  <title>MyCharts</title>
  <!-- 引入echarts.js 很多cdn源都可以引用 视情况使用 -->
  <script src="https://cdn.jsdelivr.net/npm/echarts@latest/dist/echarts.min.js">
  </script>
</head>
<body>
  <!-- 图表容器--整个网页就是一个div 内容都是通过echarts.min.js渲染出来的 -->
  <div id="main" style="width: 600px;height:400px;"></div>
  <!-- 图表渲染的js代码 -->
  <script type="text/javascript">
      // 初始化echarts实例
      var myChart = echarts.init(document.getElementById('main'));
      // 指定图表的配置项和数据
      var option = {
          // option变量的内容是一个js对象类型(Object) 类似python中的字典类型
          // option对象的属性(Attribute)可以赋值为各种js基本类型 包括对象类型
          title: {text: 'ECharts 入门示例'}, // 图表标题选项
          tooltip: {},                      // 工具提示选项
          legend: {data:['销量']},          // 图例选项
          xAxis: {data: ["衬衫","羊毛衫","雪纺衫","裤子","高跟鞋","袜子"]},
          yAxis: {},  // 缺省时没有标注
          series: [{  // series就是图表主体部分用到的数据
              name: '销量',   // name参数用于区分多个series
              type: 'bar',    // 图表类型 可以是line,scatter等
              data: [5, 20, 36, 10, 10, 20] // 图表主体的数据 这里就是bar的长度
          }]
      };
      // 使用刚指定的配置项和数据显示图表。
      myChart.setOption(option);
  </script>
</body>
</html>
```

<img src="/post-assets/20221221/basic.png" style="filter: brightness(80%);">

将上述代码保存为 html 文件在浏览器打开就可以看到图表了（左图），如果直接将 `<head>` 中引用的 js 文件保存到本地，将引用链接改为本地文件链接，那么无网络的情况下也可以显示。清楚了原理很容易就可以画出自己想要的图表，比如将上述代码中 `option.series[0].type` 的值改为 `line`，条形图就变成折线图了（右图）。将 `option.series[0].data` 中的数据改为自己的数据，就可以可视化自己的数据了。

### PyEcharts

但是在实际的数据可视化中，我们的数据可能要成千上万行，绘制的图表也有很多个，不可能手动复制粘贴去修改，这时候就需要用到 pyecharts 了。首先通过 `pip install pyecharts --upgrade` 安装或更新到最新版。绘制以上同样的图表，只需要以下的代码：

```python
# 引入子模块
from pyecharts import charts as pyc
from pyecharts import options as opts
# 这里链式调用，也可以先实例化一个 charts 对象再一步步设置
bar = (
    # 实例化：
    pyc.Bar()
    # 对应上面html文档中的 option.XAxis.data：
    .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    # 对应上面html文档中的 option.series：
    # 注意到 option.series 是js数组类型，也就是可以添加多个序列
    .add_yaxis("销量", [5, 20, 36, 10, 10, 20]) 
    # global_opts 对应上面html文档中的 option 的属性
    # 例如 option.title 通过 title_opts 参数设置
    .set_global_opts(title_opts=opts.TitleOpts(title="Echarts 入门示例"))
)
# 在当前目录生成默认名称为 render.html 的文档
bar.render()
```

通过这个例子就可以理解，pyecharts 就是通过 python 代码设置 js 代码里面的变量，然后生成 html 文档。生成的 html 文档不同的是 pyecharts 会把缺省的参数自动填充默认值，所以生成的 html 代码会比上面自己写的长很多。从这里也可以明白为什么说使用 pyecharts 要使用开始介绍的两个 echarts 的网站，因为参数都是对应的。**pyecharts 文档中没查到的参数设置，可以到 echarts 的教程上面查找。**

## 二、常用图表

pyecharts 整个项目结构见下图。绘图时用到的所有类都在 `pyecharts.charts` 子模块中，`pyecharts.charts` 中所有的图表类都直接或间接继承于 `pycharts.Base` 类。由于图表类型不同，中间又有不同图表类型的子类，如直角坐标系图表类、 3D 图表类、极坐标图表类，等。相同图表类型的图表绘图方法高度一致，在文章开始介绍的两个 pyecharts 网站中可以找到 API 文档和图表示例代码，比较简单，这里就不做示例了。

![](/post-assets/20221221/tree.png)


## 三、选项配置

选项配置主要是 set_global_opts 和 set_series_opts 两个方法，顾名思义前者设置的是对应的 js 代码中 `option` 变量的属性，后者设置的对应的 js 代码中 `option.series` 的属性。

### 常用 options

常用的 options 分为四类，对应一般绘图时的四个步骤，注意不是所有图表类的对应方法都通用，在 pyecharts.org 中参考对应方法的文档使用，或者在 ipython 中使用魔术命令： `Bar.set_global_opts?`、`opts.ItemStyleOpts?` 等查看对应参数和用法。另外这些 options 是层层嵌套的，比如在 `opts.AxisOpts` 里面，又有用到 `opts.LabelOpts` 来定义坐标轴标签的样式，这里就不深挖了，只列举第一层的常用选项。

```python
# 图表类的 init_opts，设置图表整体风格，大小等
from pyecharts import globals as glbs
Bar(init_opts=opts.InitOpts(
    glbs.ThemeType.DARK, bg_color='#1a1c1d', width='100%', height='400px'))
```

```python
# 使用 add 方法（泛指 add_yaxis, add_schema 等）时的 options
itemstyle_opts = opts.ItemStyleOpts()           # 元素样式，如散点图的点
emphasis_itemstyle_opts = opts.ItemStyleOpts()  # 元素高亮样式
linestyle_opts = opts.LineStyleOpts()           # 用于 Line.add_yaxis 里面
```

```python
# set_series_opts 中的 options
tooltip_opts = opts.TooltipOpts()       # 工具提示
itemstyle_opts = opts.ItemStyleOpts()   # 元素样式
label_opts = opts.LabelOpts()           # 标签样式，例如条形图顶部显示的数值
areastyle_opts = opts.AreaStyleOpts()   # 区域样式，例如线形图与坐标轴围成的区域
```

```python
# set_global_opts 中的 options
title_opts = opts.TitleOpts()           # 标题、副标题及其字体颜色等样式
xaxis_opts = opts.AxisOpts()            # 坐标轴的样式，如轴线样式，文字旋转等
yaxis_opts = opts.AxisOpts()            # 同上
datazoom_opts = opts.DataZoomOpts()     # 数据过多时可以使用类似滚动条的插件
legend_opts = opts.LegendOpts()         # 图例是否显示、样式及其位置等
visualmap_opts = opts.VisualMapOpts()   # 视觉映射，例如按系列值的大小给元素着色
toolbox_opts = opts.ToolboxOpts()       # 在图中加入放大缩小，导出图片等工具按钮
graphic_opts = [opts.GraphicGroup]      # 加入自定义的图形，如多边形，水印等内容
```

### 经验总结

要注意到 `pyecharts.options` 这个类，上述的两个方法的参数都是赋值为 `pyecharts.options` 的子类，例如 `pyecharts.options.AxisOpts`、`pyecharts.options.LabelOpts`，等。除了使用 options 的子类，还可以直接使用字典（如前文所述，Python 中的字典就相当于 js 的对象，把对应的字典传给对应的属性对象就可以了）。例如

```python
Line().add_xaxis(...).add_yaxis(...).set_global_opts(
    legend_opts=opts.LegendOpts(is_show=True)
)
```

可以写为：

```python
Line().add_xaxis(...).add_yaxis(...).set_global_opts(
    legend_opts={'show': True}
)
```

注意到这里也不是严格对应的，比如在 options 类的参数中是 `is_show`，在字典中的键是 `show`。个人觉得这是 pyecharts 的一个不足的地方，主要是因为这是一个 Python 的项目，它得默认使用者不懂 js，于是只能弄一堆的 options 对象（pyecharts.options 里面有大约 100 个 `***Opts` 对象）和一堆的 init 参数，虽然是很好地 “Python 化” 了，但是也搞到 API 十分的复杂，难记。如果在某些地方适当地使用 js 里面的内容，那整个逻辑会简化很多。可以看到文章一开始使用 echarts 绘图的时候就是一个很简单的逻辑。

另外这些 options 并不是全在上述两个方法中使用，比如 `add_yaxis` 中也有很多 options 参数，感觉有点混乱。而且也不是所有图表通用的，只能说是在大部分图表中通用。例如

```python
Scatter().add_xaxis(...).add_yaxis(...).set_series_opts(
    emphasis_itemstyle_opts=opts.ItemStyleOpts(color='red')
)
```

这段代码设置了散点图当鼠标悬停在某个点上时，该点的高亮颜色。在 Line、Bar 等很多图表中都适用，但是在地理图 Geo 的 set_series_opts 方法中没有这个参数。

**并且**，尝试得多了，你可能会发现：

- 在 echarts 中支持的属性，在 pyecharts 的 API 中不支持
- 在 pyecharts 中支持的参数，写到 html 中时被直接无视，也不报错
- 在 pyecharts 中不支持的参数，你给它直接写到 html 文档的 js 代码中时，它是有效的

等等各种奇怪的问题，这也不知道是 pyecharts 不成熟的地方，还是我自己没理解透彻。**总之说明了一个道理，如果想要精通 pyecharts，随心所欲地绘制各种图表，还是要从学习 echarts 入手。**

## 四、进阶技巧

基础的绘图例子都可以在文章开始的网站中找到示例代码，下面介绍一下一些比较有用的进阶技巧。

### 更换 echarts.min.js 的引用源

pyecharts 默认的源是 `https://assets.pyecharts.org/assets/`，一般国内网络加载较慢，可以通过以下代码采用 CDN 加速。当然，也可以像简介部分提到的，将 echarts.min.js 保存到任意位置然后引用，比如 localhost。如果这样则要注意，有些图不仅仅会用到 echarts.min.js，还需要用到其他的 js 资源，例如地图、词云图等。

```python
cdn = "https://cdn.jsdelivr.net/npm/echarts@latest/dist/"
pyecharts.globals.CurrentConfig.ONLINE_HOST = cdn
```

### 参数数据类型

一般我们会使用 pandas 处理数据，然后绘图。这时候要注意 pandas 里面的数值型使用的不是内置类型，而是 numpy 的数据类型。而 echarts 是不认识 numpy 数据类型的，这时候会画一个空的没有数据的图，让人一脸懵。可以如下显式更改数据类型：

```python
Scatter().add_xaxis(
    [int(i) for i in df.index]
).add_yaxis(
    [float(n) for n in df['col']]
)
# 注意，以下写法是不行的
# 虽然传递的参数是内置类型float，但是pandas还是会自作主张给你改成numpy.float64
df['col'].astype(float)
```

### 遇到不支持或无效的参数怎么办

遇到上一小节最后提到的那些问题，比如某些属性在 echarts 中有效，但是在 pyecharts 中没有对应参数或者无效，这种情况我们最直接的思路就是手动去改 html 里面 js 代码。还是以简介部分那个图为例，比如我想微调一下主图在整个画布中的位置（可能是出于想用 overlap 方法在主图上叠加其他图表等原因），在 echarts 中可以如下实现：

![](/post-assets/20221221/grid-html.png)

通过 grid 属性定义主图<u>左上角的坐标</u>和<u>右下角的坐标</u>，坐标可以是相对值，也可以是数值表示的绝对值，单位是 css 中的 px。为了好理解，这里故意将位置调得夸张一点，实际上我们经常需要微调主图的位置来适应布局。但是在 pyecharts 的常用图表类里是不能直接设置这个 grid 属性的（反正我没找到）：

![](/post-assets/20221221/grid-error.png)

可以看到 global options 没有该参数。series options 里面倒是有这个参数，但是设置的不是同一个东西了。唯一的可能就是使用 `pyecharts.charts.Grid` 这个图表类时才允许加入 grid 参数，如下图。但是 Grid 类和其他常用图表类（例如 Bar）直接继承的类是不一样的，所以用 Grid 类打包图表之后会导致很多原本支持的设置失效。

![](/post-assets/20221221/grid.png)

对此一开始我想到一个很笨拙的办法：强行在 js 中写入属性，替代手动修改属性的过程。首先按照简介中的代码，生成一个 render.html，然后如下写入属性：

```python
def write_js(file, locate, write):
    with open(file, 'r') as f:
        html = f.read()
    with open(file, 'w') as f:
        f.write(html.replace(locate, write)) 

# locate 用于定位修改位置，这里建议使用 option 里面参数来定位
# 比如这里准备写到 "title" 属性的前面：
locate = '"title": [\n' # 尽可能多的字符，否则可能定位错
# 需要写入的属性
write = '"grid": {"x": "20%", "y": "30%", "x2": "30%", "y2": "20%"}, \n'
write_js('render.html', locate, write+locate)
```

![](/post-assets/20221221/render.png)

好笨的方法，千万别学。后来经过进一步对 pycharts 图表类属性的研究，发现可以直接这样就搞定：

```python
# 首先绘图
bar = Bar().add_xaxis(...).add_yaxis(...).set_global_opts(...)
# 然后还缺什么属性，直接往 options 里面添加，然后 render
bar.options['grid'] = {"x": "20%", "y": "30%", "x2": "30%", "y2": "20%"}
bar.render()
# 注意到 bar.options 是一个字典
# 它的内容就是生成的 html 文档的 js 代码里面的 option 变量的内容
```


### 运用 css 和 js 控制视觉效果

如果想要随心所欲地自定义视觉效果，需要了解一些 css 的基础知识。如果仅仅使用 Python 代码，我们定义一个散点图中的散点的颜色时，可以这样写：

```python
Scatter().add_xaxis(index_data).add_yaxis(
    series_name, value_data, itemstyle_opts=opts.ItemStyleOpts(color='red')
)
```

运用 css 代码，我们可以将 ItemStyleOpts 写成：

```python
opts.ItemStyleOpts(color='#ff0000')
opts.ItemStyleOpts(color='rgba(255,0,0,1)')
```

或者运用图表库提供的函数：

```python
from pyecharts.commons.utils import JsCode

opts.ItemStyleOpts(color=JsCode(
    # 引入echarts.min.js后就可以使用echarts库的方法
    ''' new echarts.graphic.RadialGradient(
        0.5, 0.5, 0.5,  
        [{offset: 0, color: 'rgba(255,0,0,1)'},
        {offset: 1, color: 'rgba(0,0,255, 1)'}]
    ) '''
))
```

直接在浏览器 console 中更改 option（还是简介中那个图），结果如下：

![](/post-assets/20221221/css.png)

或者对于一些在 API 中为数值常量的参数，可以通过 js 变成变量。例如在散点图中定义点的大小，可以通过 js 使每个点显示不同大小：

```python
Scatter().add_xaxis(...).add_yaxis(
    series_name, data_pair,
    # 固定大小：symbol_size=5,
    # 由数据值决定大小：
    symbol_size=JsCode('function (data) {return data[1]*10;}')
)
```

这里用到的 js 代码其实并不复杂，Python 混 Java 的感觉，学过上述两种的凭感觉都会写。感兴趣的可以学一下。

### 运用 js 控制交互效果

这里经常用于改变工具提示（tooltip），就是鼠标悬停某个数据点时显示出来的东西。以之前做的一个图表为例，主要涉及的 option 如下：

![](/post-assets/20221221/example-option.png)

默认情况下图表的工具提示格式是：`series_name \n x轴标签 y轴的值`，但是如果需要自定义格式，就要使用 js 回调函数。这个在 pyecharts.org 上也提到，但不是很详细。具体代码如下，参考上图给出的 option 就可以理解了。

```python
from pyecharts.commons.utils import JsCode
tooltipJS = '''
    function (param) { 
        // param参数接收的就是上图中的option.series.data这个数组中对应的元素
        var line1 = 'name: ' + param.data[0] + '<br/>';
        var line2 = 'value1: ' + param.data[1] + '<br/>';
        var line3 = 'value2: ' + Math.ceil(10**param.data[2]); // 引用js内置函数
        return line1 + line2 + line3;   // return的内容就是到时工具提示显示的内容
    }
'''
# 然后通过 JsCode 传给 TooltipOpts 的 formatter
Scatter().set_global_opts(
    tooltip_opts=opts.TooltipOpts(formatter=JsCode(tooltipJS))
)
```

那么问题来了，上图的数据部分是绘图时传给 series 的参数，但是如果我还想在 tooltip 里说点别的东西怎么办？可以在 `add_yaxis` 时**强行**传进去再使用 js 回调函数。以下是一个完整的绘图例子：

```python
from pyecharts import charts as pyc
from pyecharts import options as opts
from pyecharts import globals as glbs
bar = pyc.Bar(
    init_opts=opts.InitOpts(theme=glbs.ThemeType.DARK, bg_color='#1a1c1d')
).add_xaxis(['A', 'B', 'C']).add_yaxis(
    'test charts', [1, 2, 3]
)
bar.render()
```

![](/post-assets/20221221/tooltipjs.png)

如上图，这时是默认的工具提示格式，但是如果我想让它不显示为数值 `[1, 2, 3]`，而是描述性的信息，例如 `['低', '中', '高']` 或者更丰富的信息，那可以这样写：

```python
bar = pyc.Bar(
    init_opts=opts.InitOpts(theme=glbs.ThemeType.DARK, bg_color='#1a1c1d')
).add_xaxis(['A', 'B', 'C']).add_yaxis(
    'test charts', 
    [(0,1,'低','更多信息'), (1,2,'中','更多信息'), (2,3,'高','更多信息')]
    # 通过元组列表强行传进更多的信息
    # 元组第一位是对应 xaxis 的序号，第二位是序列的值 这里就是bar的长度
    # 剩下的爱加多少加多少，每个点写一篇800字作文都行
).set_global_opts(
    tooltip_opts=opts.TooltipOpts(formatter=JsCode(
        '''
        function (param) {
            var line1 = 'index: ' + param.data[0] + '<br/>';
            var line2 = 'value: ' + param.data[1] + '<br/>';
            var line3 = 'info1: ' + param.data[2] + '<br/>';
            var line4 = 'info2: ' + param.data[3];
            return line1 + line2 + line3 + line4;
        }
        '''
    ))
)
bar.render()
```

结果如下。如果没看懂这段代码，可以将 html 源码 render 出来，对应着看，自然就理解了。

![](/post-assets/20221221/tooltipjs2.png)

**再进一步深入**，以上方法也不是万能的。例如在特别的图表中，根本就没有 `add_yaxis` 方法。比如 `Geo` 图表类型，是通过 `add` 方法传入 series 数据，于是传参的格式跟上面方法又不同。这时可以通过以下代码传入主图以外的数据，以作 tooltip 等其他用途：

```python
Geo().add_schema(...).add_coordinate(...).add(
    series_name,
    [('A', [1,2,3]), ('B', [4,5,6]), ('C', [7,8,9])]
    # 这里传入的数据也是一个元组列表，但是它仅限两位的元组
    # 第一位是地图中区域的名称，也就是到时会在地图中 A, B, C 三个地点标上散点
    # 第一位在 js 中通过 param.data.name 引用
    # 第二位是包含额外数据的列表，通过 param.data.value(一个js数组) 引用
).set_global_opts(
    tooltip_opts=opts.TooltipOpts(formatter=JsCode(
        '''
        function (param) {
            return param.data.name + ' | ' +
                'info1:' + param.data.value[0] + 
                ', info2:' + param.data.value[1] + 
                ', info3:' + param.data.value[2];
        }
        '''
        # 当鼠标悬停在 B 点时，返回“B | info1: 4, info2: 5, info3: 6”
    ))
)
```

**更复杂的情况是**，加上额外数据后格式就是不对，或者会影响图表的其他组件。比如在上例的 `set_global_opts` 中加入了 visualmap 组件时，直接传入额外数据会导致 visualmap 显示不正常。这里我没有找到好的办法，所以想了一个*投机取巧*的方法。就是由于 python 字典跟 js 对象的格式是一样的，可以直接生成个字典的字符串写入 js ，然后再引用。于是上例可以写成：

```python
data_pair = [('A', 10), ('B', 100), ('C', 1000)] 
# 上面的 10, 100, 1000 就是用在 visualmap 中的数据值
# 下面的则是用于 tooltip 的额外数据
extend_data = {'A': [1,2,3], 'B': [4,5,6], 'c': [7,8,9]}
tooltip_data = [{'name': k, 'value': v} for k, v in extend_data.items()]
# 返回 tooltips 的函数
tooltipjs = r'''
    function (param) {
        var tooltip_data = __MARKER__; // 到时在 __MARKER__ 这里写入要传的数据
        for (int i=0; i<tooltip_data.length; i++) {         // 遍历
            if (param.data.name == tooltip_data[i].name) {  // 匹配
                var values = tooltip_data[i].value;         // 提取额外数据
            }
        }
        // 返回给 tooltips
        return param.data.name + ' | ' +
            'info1:'+values[0]+', info2:'+values[1]+', info3:'+values[2];
    }
'''.replace('__MARKER__', str(tooltip_data))    # 在这里加入了额外的数据
# 绘图
Geo().add_schema(...).add_coordinate(...).add(
    series_name, data_pair
).set_global_opts(
    tooltip_opts=opts.TooltipOpts(formatter=Jscode(tooltipjs)),
    visualmap_opts=opts.VisualMapOpts()
)
# 可以看到在调用 pyecharts 绘图时没有用到额外的数据，只有 data_pair
# 用于 tooltip 的数据是写在 js 字符串里偷偷传进去的
# 这时 tooltip 的效果跟上例一样，但是上例的写法不支持 visualmap
```

到这基本就到头了，因为上面的方法虽然很繁琐，但是基本可以所向披靡，对任何的图都通用。并且除了 tooltips，其他的数据交互的问题也可以按类似的思路来解决。

## 五、总结

### 整体思路

回到开始的问题：如何用 pyecharts 绘制一切你想象得到的交互式图表？根据前文的内容可以总结以下的流程：

- 1） 根据需求，确定图表类型、基本内容、风格样式等
- 2） 粗略画个草图，如果不会画就上 gallery.pyechart.org 上抄个类似的
- 3） 通过各种 options 修改图中的细节部分，详细的文档在 pyecharts.org 上
- 4） 实在找不到的设置，在简介提到的 Echarts 网站中找配置，再写进 options 字典里
- 5） 跟据前面提到的技巧及其思路，修改还不符合要求的地方

### 实例验证

最后用一个实例验证一下上述流程是否真的有用。**第一步**，先找一个属于常用图表类型的，并且相对较复杂的图。这里到 Apache 官网的示例上去找，就用如下[这个图](https://echarts.apache.org/examples/zh/editor.html?c=line-aqi)，看起来花里胡俏挺唬人的。~~这个图印象中 gallery.echarts.org 好像有个类似的例子，这里我们假装不知道先来自己做一下。~~

![](/post-assets/20221221/example.png)

首先把 option 的内容折叠起来以免自己偷看作弊。它上面的数据没有直接写入 option 变量，而是用 jquery 加载的，我没找到接口在哪，所以就在 option 里面加了个 dataView 工具，然后运行，在数据视图中将数据 copy 下来，保存为 `data.csv`（文末可以下载）。

**第二步**，粗略地画个草图。就是一个主题为 `pyecharts.globals.ThemeType.DARK` 的折线图，横轴为时间，纵轴为 AQI 数值。页面上加了标题、工具箱、图例、数据缩放工具四个组件。

```python
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
)
line.render()
```

首先画个最简单的图，结果长下面那样。对比一下，缺了上面提到的三个组件。图例默认有了，但是不是我们想要的样子。工具提示显示格式一样了，但是触发方式不一样，下图中要悬停到具体的点上才能触发。折线上多了数据标签，主图背景少了些分割线，等等。

![](/post-assets/20221221/1.png)


然后先简单地使用默认参数，把完整的 options 加上去。可以看到加上组件了，已经离目标接近很多了。

```python
line.set_series_opts(
    label_opts=opts.LabelOpts(is_show=False)
).set_global_opts(
    title_opts=opts.TitleOpts(title='Beijing AQI'), 
    legend_opts=opts.LegendOpts(),
    visualmap_opts=opts.VisualMapOpts(), 
    datazoom_opts=opts.DataZoomOpts(), 
    tooltip_opts=opts.TooltipOpts(),
    toolbox_opts=opts.ToolboxOpts()
)
line.render()
```

![](/post-assets/20221221/2.png)

**第三、四、五步**其实就是一个循环迭代的过程，用这个思路来修改每一个 options。接下来根据上图与原图不同的地方，具体定义组件的 options 参数：

Title 这里位置太靠边了，原图离最左边有点空隙：

```python
line.set_global_opts(
    title_opts=opts.TitleOpts(title='Beijing AQI', pos_left='1%'), 
)
```

Legend 这里想了很久，不知道怎么改成原图那样。后来跳过了，做下面的内容才发现原图那个不是 Legend，而是 Visual Map 产生的图例。所以这里就简单把 Legend 禁用了就行了：

```python
line.set_global_opts(
    legend_opts=opts.LegendOpts(is_show=False),
)
```

Viusal Map 这里明显的区别是，原图那个是离散的，这里目前是连续的，所以先找个参数将它离散化。然后需要定义一下分割的区间。这里我先是凭感觉写了个参数，发现不对，便上 [官网](https://pyecharts.org/#/zh-cn/global_options?id=visualmapopts%ef%bc%9a%e8%a7%86%e8%a7%89%e6%98%a0%e5%b0%84%e9%85%8d%e7%bd%ae%e9%a1%b9) 查看了一下文档如下。至于每个区间的颜色，可以用取色器在原图获取：

```
# 自定义的每一段的范围，以及每一段的文字，以及每一段的特别的样式。例如：
# pieces: [
#   {"min": 1500}, // 不指定 max，表示 max 为无限大（Infinity）。
#   {"min": 900, "max": 1500},
#   {"min": 310, "max": 1000},
#   {"min": 200, "max": 300},
#   {"min": 10, "max": 200, "label": '10 到 200（自定义label）'},
#   {"value": 123, "label": '123（自定义特殊颜色）', "color": 'grey'},
#   {"max": 5}     // 不指定 min，表示 min 为无限大（-Infinity）。
# ]
pieces: Optional[Sequence] = None,
```

```python
line.set_global_opts(
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
)
```

Data Zoom 组件默认显示 20%-80%，看一下文档，发现有个 start_value 参数，由于原图中就是显示最后一部分，所以可以使用这个参数搞定。但是第一次试了发现不行，原因是它已经给开始结束位置设置了默认值，所以要先将这两个值设为空值，start_value 参数才有效。另外发现原图是可以在主图里面直接拖动的，所以要改一下 type 参数。这里就出现问题了，改了 type 之后，里面可以拖动了，但是外面的滚动条不见了。这里只好查看原图的代码抄答案，原来它用数组的形式定义了两个 Data Zoom 组件，一个在里面，一个在外面：

```python
line.set_global_opts(
    datazoom_opts=[
        opts.DataZoomOpts(
            start_value='2014-06-01', range_start=None, range_end=None
        ), 
        opts.DataZoomOpts(type_='inside')
    ], 
)
```

Tooltip 组件需要更改触发方式，简单加上 trigger 参数即可：

```python
line.set_global_opts(tooltip_opts=opts.TooltipOpts(trigger='axis'))
```

Toolbox 组件多了一堆按钮，简单研究一下文档，把不必要的删掉就 OK。实际写时发现 pyecharts 的那个缺点又出现了，就是太多 options 对象。执行 `opts.ToolBoxFeatureOpts?` 就可以看到，可以说看了头皮发麻。所以这里用字典传参，只需要把 python 中的变量名用下划线分割单词的习惯，改成 js 中用首字母大写分割单词，参数基本就对应上了。然后用字典代替 options 对象就行了。然后还要改一下它的位置，原图是靠右对齐，并且离最右边留有一点空隙，这里因为也已经有默认参数，一样要先把默认参数置空：

```python
line.set_global_opts(
    toolbox_opts=opts.ToolboxOpts(
        pos_left=None, pos_right=10,
        feature={'dataZoom': {}, 'restore': {}, 'saveAsImage': {}}
    ),
)
```

然后好像还差点什么？没错，Y 轴的分割线。估计应该是在 yaxis_opts 里添加，于是试一下。发现分割线是有了，但是还差那些带箭头的虚线，查了一下文档，发现应该在 `set_series_opts` 里面通过 `opts.MarkLineOpts` 设置：

```python
line.set_series_opts(
    markline_opts=opts.MarkLineOpts(
        label_opts=opts.LabelOpts(position='end'),
        linestyle_opts=opts.LineStyleOpts(color='#333'),
        data=[{'yAxis': y} for y in [50, 100, 150, 200, 300]]
    )
)
```

最后主图的位置和原图不太一样，通过 grid 属性改一下。参考第四节中提到的关于[参数无效](#遇到不支持或无效的参数怎么办)的问题：

```python
line.options['grid'] = {'left': '6%', 'right': '15%', 'bottom': '12%'}
```

结果如下，基本与原图一样了。有细微区别是因为 Apache 网站上的图表容器跟这里的不一样，并且上面的网页截图时浏览器开了扩展 DarkReader，所以颜色也有点改变的地方。

> <a href="{{ site.baseurl }}/post-assets/20221221/render.html" target="_blank">在新标签页打开交互式图表</a> \| <a href="{{ site.baseurl }}/post-assets/20221221/code.py" target="_blank">下载完整绘图代码</a> \| <a href="{{ site.baseurl }}/post-assets/20221221/data.csv" target="_blank">下载数据</a>

![](/post-assets/20221221/3.png)

<hr>

鸣谢 & 推荐：<a href="https://blog.csdn.net/qq_27484665" target="_blank">@AwesomeTang</a>

**END**
