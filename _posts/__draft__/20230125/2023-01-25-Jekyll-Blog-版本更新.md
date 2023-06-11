---
layout:         post
title:          "Paradise's Blog 版本更新"
subtitle:       "新版本说明"
date:           "2023-01-25 12:00:00"
author:         "Paradise"
header-style:   text
tags:
    - 日志
---

`热烈庆祝 Paradise's Blog 上线 V2.0 版本！[鼓掌][鼓掌][鼓掌]`

好久没逛博客，也有点懒得写。感觉主要是潜意识觉得自己技术够硬了，不用学习写笔记了？？~~罪恶感~~ Anyway，最近看到 [黄玄](https://www.github.com/huxpro/) 大神的博客模版又更新了，于是跟着更新一波。并且由于这段时间搞数据可视化的时候 *道听途说* 地学了不少前端知识，居然连 js 和 css 都会写了，于是便再重新修饰一下博客站点。

本次更新前期的更改测试都在 Gitee 上进行了，详细信息可以查看下面 repository 的前 58 个 commits：

> <https://gitee.com/paradiseeee/blog/commits/master>

模版 clone 下来的时候在这个位置：

> [huxpro.github.io-commit-b1d823b4e24a4b3b0efa4cc87c79d041e6b3de43](https://github.com/Huxpro/huxpro.github.io/commit/b1d823b4e24a4b3b0efa4cc87c79d041e6b3de43)

初步改好之后在这个位置：

> [blog-commit-387e734c7ae7f6365997e461e5e0665b61178a1d](https://gitee.com/paradiseeee/blog/commit/387e734c7ae7f6365997e461e5e0665b61178a1d)

然后就是将改好的站点合并到之前的 v1.0 上面，也就是本仓库的 commit `merge v1.0`：

> [Paradiseeee.github.io-commit-2fe4aee54c8db909485000742deca3534e842ddc](https://github.com/Paradiseeee/Paradiseeee.github.io/commit/2fe4aee54c8db909485000742deca3534e842ddc)

最后就是整理好一些文档资源的存放和引用路径等问题，然后就可以开心地写文章啦！`[鼓掌][鼓掌][鼓掌]`

### 以下是在博客模版的基础上做出的主要更改

- 弃用了一些功能，例如 PWA。因为我不懂，这样避免很多未知的 Error；
- 删除了没有用到和不打算用到的文件；
- 将 cloudflare 上的 js、css、和字体资源全部同步到 localhost，也就是不使用 CDN；
- 由于弃用了 cloudflare 还需要将 MathJax 依赖的资源也同步下来，这样加载数学公式快多了；
- 弃用 img 目录，改为 page-assets 和 post-assets 两个资源目录；
- 改为黑暗模式（/css/hux-blog.dark.css），这里是直接对着网页一点点美化的，没有将整个 css 覆盖更改，所以以后可能出现问题，到时再改，反正好玩；
- 本来想加个切换 dark \| light 的按钮，用 js 更改样式，但是除了 css 还有不少东西要跟着改的，没头绪；
- 由于部署在 Gitee Page 的时候域名后面加了个 /blog （仓库名），所以 jekyll 引擎渲染 markdown 的时候会出现问题，具体解决参考 [/post-assets/info]({{ site.fileurl }}/post-assets/info)。这样会存在问题，暂时用着。

### Version 1.0 的日志看这里

> [README.1.0.md]({{ site.fileurl }}/README.1.0.md)

<br>