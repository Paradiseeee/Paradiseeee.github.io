---
layout:     post
title:      "Python 处理 PDF 文件"
subtitle:   "再也不需要 Office、WPS 或 Adobe 啦！"
date:       2021-05-26 12:00:00
author:     "Paradise"
header-img: "img/post-bg.jpg"
header-style: text
tags:
    - Python
    - 编程语言
    - 总结
---

本文记录如何使用 python 集成自己常用的 PDF 文件处理功能。虽然下面只是一些很简单的功能，但是据我所知，WPS 需要收费，Office 根本不支持（Word 对 PDF 文件的解析可以用一塌糊涂来形容）。

# 脚本

```python 
# coding = utf-8
"""
    通过PyPDF2包可以实现PDF文档的读取、合并、拆分、旋转、水印，以及加密等操作；
    类似的包：pdfrw，与上面的功能基本一样除了加密。
"""
import PyPDF2 as pp

# 1. 提取PDF中的信息
def extract_information(filepath):
    ''' 这里函数结束之后文件关闭，无法调用.extractText方法读取文本'''
    with open(filepath, 'rb') as f:
        pdf = pp.PdfFileReader(f)
        information = pdf.getDocumentInfo()
        pages = pdf.getNumPages()
        return pdf, information, pages
print(extract_information("test1.pdf"))


# 2. 合并PDF
def merge_pdfs(outputpath, *filepaths):
    '''将filepaths中的文件合并，并写入outputpath'''
    writer = pp.PdfFileWriter()
    for path in filepaths:
        pdf = pp.PdfFileReader(path)
        for page in range(pdf.getNumPages()):
            writer.addPage(pdf.getPage(page))
    with open(outputpath, 'wb') as f:
        writer.write(f)
    print('成功写出文件！')


# 3. 拆分PDF
def split_pdfs(filepath, pages):
    '''pages应该是一个列表'''
    writer = pp.PdfFileWriter()
    pdf = pp.PdfFileReader(filepath)
    for p in pages:
        if p > pdf.getNumPages():
            print(f'页数<{p}>超出文档页码范围！')
        else:
            page = pdf.getPage(p-1)
            writer.addPage(page)
    with open('splited.pdf', 'wb') as f:
        writer.write(f)
    print('成功写出文件！')

``` 

<br>

# 应用

在 python 安装目录的 `site-packages` 目录下新建一个文件夹，随便命名，例如 `my_module`。这就是一个库。再将以上脚本放进文件夹（这里命名为 `PdfProcessing`）。就可以在代码中引用。例如在本例中，如果需要在命令行中修改 PDF 文件，可以进入 ipython shell，执行以下代码：

```python
from my_module import PdfProcessing

# 查看PDF中的大致信息（不排除有解析错误的情况，只是大致看一眼）
PdfProcessing.extract_information('C:/test.pdf')

# 提取PDF某一部分页码
PdfProcessing.split_pdfs('C:/test.pdf', [4,5,6])
# 提取 4、5、6 页，输出为 splited.pdf

# 合并 PDF
PdfProcessing.merge_pdfs('output.pdf', 'C:/test.pdf', 'splited.pdf')
```

另一种方法是将以上脚本通过 `Pyinstaller` 编译成可执行文件（.exe），这样就可以不用 Python 环境直接运行。不过需要设计一下用户操作指引，自己用可以直接在命令行操作。如果想分享给别人可以设计一个用户页面，并且用 `Nuitka` 打包可以生成一个更高效的可执行文件。

跟据自己对PDF文件的高频操作，可以自己定义函数功能。这里还有很多功能没有提及，如果如要增加更多的功能可以参考 [PyPDF2 的文档](https://pythonhosted.org/PyPDF2/)。

<br>