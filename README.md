# Paradiseeee.github.io

<p>
    <a href="https://github.com/paradiseeee/paradiseeee.github.io" target="_blank">
        <img src="./page-assets/github.svg" style="max-width:100%;">
        &nbsp;Fork on GitHub
    </a>
    &nbsp;&nbsp;
    <a href="https://paradiseeee.github.io" target="_blank">
        <img src="./page-assets/github-mirror.svg" style="max-width:100%;">
    </a>
    &nbsp;&nbsp;
    <a href="https://gitee.com/paradiseeee/blog" target="_blank">
        <img src="./page-assets/gitee.svg" style="max-width:100%;">
        &nbsp;Fork on Gitee
    </a>
    &nbsp;&nbsp;
    <a href="https://paradiseeee.gitee.io/blog" target="_blank">
        <img src="./page-assets/gitee-mirror.svg" style="max-width:100%;">
    </a>
</p>

## 以下是在博客模版的基础上做出的主要更改

- 弃用了一些功能，例如 PWA。因为我不懂，这样避免很多未知的 Error；
- 删除了没有用到和不打算用到的文件；
- 将 cloudflare 上的 js、css、和字体资源全部同步到 localhost，也就是不使用 CDN；
- 由于弃用了 cloudflare 还需要将 MathJax 依赖的资源也同步下来，这样加载数学公式快多了；
- 弃用 img 目录，改为 page-assets 和 post-assets 两个资源目录；
- 改为黑暗模式（/css/hux-blog.dark.css），这里是直接对着网页一点点美化的，没有将整个 css 覆盖更改，所以以后可能出现问题，到时再改，反正好玩；
- 本来想加个切换 dark \| light 的按钮，用 js 更改样式，但是除了 css 还有不少东西要跟着改的，没头绪；
- 由于部署在 Gitee Page 的时候域名后面加了个 /blog （仓库名），所以 jekyll 引擎渲染 markdown 的时候会出现问题，具体解决参考 [/post-assets/info](/post-assets/info)。这样会存在问题，暂时用着。
