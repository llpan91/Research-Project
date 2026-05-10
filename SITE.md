# Pan's Research Blog — 使用文档

## 目录

- [快速开始](#快速开始)
- [本地验证](#本地验证)
- [内容管理](#内容管理)
- [笔记同步](#笔记同步)
- [远程部署与访问](#远程部署与访问)
- [自定义配置](#自定义配置)
- [故障排查](#故障排查)

---

## 快速开始

### 前置依赖

| 工具 | 用途 | 安装 |
|------|------|------|
| Docker + Docker Compose | 构建与运行 | [docs.docker.com/get-docker](https://docs.docker.com/get-docker/) |
| Git | 管理 PaperMod 子模块 | 系统自带 |

> Hugo **不需要**本地安装，Docker 镜像中已包含。如需本地调试可选装：`snap install hugo` 或从 [GitHub Releases](https://github.com/gohugoio/hugo/releases) 下载 extended 版本。

### 首次克隆

```bash
git clone --recurse-submodules <your-repo-url>
cd UMI-project/site
```

如果已 clone 但忘记 `--recurse-submodules`：

```bash
git submodule update --init --depth=1
```

---

## 本地验证

### 方式一：Docker 开发模式（推荐）

```bash
cd site
docker compose --profile dev up
```

- 访问 http://localhost:1313
- 修改任何文件后浏览器自动刷新（Hot Reload）
- `Ctrl+C` 停止

### 方式二：Docker 生产模式

```bash
cd site
docker compose up --build
```

- 访问 http://localhost:8080
- 模拟真实部署效果（nginx 托管静态文件）
- 验证 gzip、缓存、PDF 服务是否正常

### 方式三：本地 Hugo（需安装 Hugo extended）

```bash
cd site
hugo server -D --disableFastRender
```

- 访问 http://localhost:1313
- `-D` 参数会渲染 `draft: true` 的草稿

### 验证清单

| 检查项 | 方法 |
|--------|------|
| 首页渲染 | 访问 `/`，确认 homeInfoParams 显示 |
| 数学公式 | 访问 `/blog/hello-world/`，确认 KaTeX 渲染 $$公式$$ |
| 暗色模式 | 点击右上角主题切换按钮 |
| 搜索功能 | 访问 `/search/`，输入关键词 |
| 论文卡片 | 访问 `/research/diffusion-models/`，确认卡片样式和 arXiv 链接 |
| 目录导航 | 长页面右侧应出现 TOC |
| PDF 访问 | 生产模式下访问 `http://localhost:8080/papers/` |
| 移动端适配 | 浏览器 F12 切换移动视图 |

---

## 内容管理

### 目录结构说明

```
content/
├── _index.md              # 首页（使用 homeInfoParams）
├── search.md              # 搜索页
├── about/_index.md        # 关于页
├── blog/                  # 博客文章
│   ├── _index.md          # 列表页配置
│   └── *.md               # 每篇文章一个文件
└── research/              # 研究主题
    ├── _index.md          # 总览页
    ├── diffusion-models/
    ├── robot-manipulation/
    ├── autonomous-driving/
    └── 3d-vision/
```

### 新建博客文章

在 `content/blog/` 下创建 `.md` 文件：

```markdown
---
title: "文章标题"
date: 2026-05-10
draft: false
tags: ["diffusion-models", "笔记"]
summary: "一句话摘要，显示在列表页。"
math: true
---

正文内容...

行内公式：$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$

独立公式：
$$L = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$
```

#### Front Matter 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | string | 文章标题 |
| `date` | date | 发布日期，影响排序 |
| `draft` | bool | `true` 则不在生产构建中显示 |
| `tags` | list | 标签，用于分类和搜索 |
| `summary` | string | 摘要，显示在列表卡片中 |
| `math` | bool | 是否加载 KaTeX（全局已启用，可省略） |
| `ShowToc` | bool | 是否显示目录，默认 true |
| `weight` | int | 自定义排序权重（越小越前） |

### 使用论文卡片

在任何 `.md` 文件中使用 shortcode：

```
{{</* paper-card
  title="DDPM"
  authors="Ho, Jain, Abbeel"
  year="2020"
  arxiv="2006.11239"
  description="奠基性工作，建立去噪扩散概率模型"
*/>}}
```

参数说明：
- `title`（必填）：论文标题
- `authors`：作者
- `year`：年份
- `arxiv`：arXiv ID，会自动生成链接
- `venue`：发表会议/期刊
- `description`：简短描述

### 管理论文数据

论文元数据存放在 `data/papers/*.yaml`，可用于未来的模板渲染：

```yaml
- title: "Paper Title"
  authors: "Author et al."
  year: 2024
  arxiv: "2401.xxxxx"
  category: "分类标签"
```

---

## 笔记同步

### 从 topics/ 同步到网站

```bash
./scripts/sync-notes.sh
```

**工作原理：**

1. 扫描 `topics/*/notes/` 下所有 `.md` 文件
2. 跳过 `_` 开头的目录（原始素材）
3. 对没有 front matter 的文件自动注入标题、日期、标签
4. 复制到 `content/blog/<topic-slug>/`

**同步映射：**

| topics/ 目录 | → content/blog/ 子目录 |
|---|---|
| `01-diffusion-models(扩散模型)/notes/` | `diffusion-models/` |
| `02-robot-manipulation(机器人操控)/notes/` | `robot-manipulation/` |
| `03-autonomous-driving(自动驾驶)/notes/` | `autonomous-driving/` |
| `04-3d-vision(3D视觉与重建)/notes/` | `3d-vision/` |

### 同步工作流

```bash
# 1. 在 topics/ 中正常写笔记
vim topics/01-diffusion-models\(扩散模型\)/notes/理论基础/新笔记.md

# 2. 同步到网站
./scripts/sync-notes.sh

# 3. 本地预览
docker compose --profile dev up

# 4. 确认无误后提交
git add content/
git commit -m "Sync notes to site"
```

### 注意事项

- 同步是**单向覆盖**（topics → site），不要直接编辑 `content/blog/<slug>/` 中同步来的文件
- 手动创建在 `content/blog/` 根目录的文章不会被覆盖
- 如果原文件已有 `---` 开头的 front matter，会直接使用原始 front matter

---

## 远程部署与访问

### 方案 A：服务器部署（Docker）

适合有 VPS/云服务器的情况。

```bash
# 在服务器上
git clone --recurse-submodules <your-repo-url>
cd UMI-project/site

# 启动生产服务
docker compose up -d --build

# 查看状态
docker compose ps
docker compose logs -f site
```

默认监听 `8080` 端口。通过 Nginx/Caddy 反向代理绑定域名：

**Caddy 示例（自动 HTTPS）：**

```
# /etc/caddy/Caddyfile
research.example.com {
    reverse_proxy localhost:8080
}
```

**Nginx 反向代理示例：**

```nginx
server {
    listen 443 ssl;
    server_name research.example.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 方案 B：GitHub Pages（免费静态托管）

```bash
# 本地构建
cd site
docker run --rm -v $(pwd):/src hugomods/hugo:exts-0.140.0 hugo --minify

# 将 public/ 部署到 GitHub Pages
# 方法1：用 ghp-import
pip install ghp-import
ghp-import -n -p -f public

# 方法2：GitHub Actions 自动部署（见下文）
```

**GitHub Actions 自动部署：**

在仓库根目录创建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy Hugo site to GitHub Pages

on:
  push:
    branches: [main]
    paths: [site/**]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: '0.140.0'
          extended: true

      - name: Build
        working-directory: site
        run: hugo --minify

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: site/public

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 方案 C：Cloudflare Pages / Vercel / Netlify

这些平台支持自动检测 Hugo 项目：

1. 连接 GitHub 仓库
2. 设置构建配置：
   - **构建命令**：`cd site && hugo --minify`
   - **输出目录**：`site/public`
   - **环境变量**：`HUGO_VERSION=0.140.0`
3. 每次 push 自动构建部署

### 远程更新内容

```bash
# 本地修改 → 推送 → 自动部署
git add site/content/
git commit -m "Add new blog post"
git push origin main
# 等待 CI/CD 自动构建（GitHub Pages/Cloudflare 等）
```

对于 Docker 服务器部署：

```bash
# SSH 到服务器
ssh your-server

# 拉取更新并重建
cd UMI-project/site
git pull
docker compose up -d --build
```

### 方案 D：局域网临时分享

不需要公网服务器，在局域网内让他人访问：

```bash
# 查看本机 IP
ip addr | grep inet | grep -v 127.0.0.1

# 启动开发服务器绑定所有网卡
cd site
docker compose --profile dev up
# 其他设备访问 http://<你的IP>:1313
```

---

## 自定义配置

### 修改个人信息

编辑 `hugo.toml`：

```toml
# 网站基础 URL（部署时必须修改）
baseURL = "https://your-domain.com/"

# 标题
title = "Your Research Blog"

# 首页介绍
[params.homeInfoParams]
  Title = "Hi, I'm YourName"
  Content = "你的研究方向介绍..."

# 社交链接
[[params.socialIcons]]
  name = "github"
  url = "https://github.com/your-username"

[[params.socialIcons]]
  name = "email"
  url = "mailto:you@example.com"
```

### 支持的社交图标

PaperMod 内置图标：`github`, `twitter`, `linkedin`, `email`, `rss`, `googlescholar`, `orcid`, `arxiv` 等。

### 添加新的研究主题

1. 创建目录和页面：
```bash
mkdir -p content/research/new-topic
```

2. 创建 `content/research/new-topic/_index.md`：
```markdown
---
title: "新主题名"
description: "简短描述"
ShowToc: true
---

内容...
```

3. 在 `content/research/_index.md` 中添加链接

4. 创建 `data/papers/new-topic.yaml` 存放论文数据

5. 在 `scripts/sync-notes.sh` 的 `TOPIC_MAP` 中添加映射

### 自定义域名配置

修改 `hugo.toml` 中的 `baseURL`，并在 DNS 添加解析记录。

---

## 故障排查

### 常见问题

**Q: 数学公式显示为原始 LaTeX 代码**

检查：
1. `hugo.toml` 中 `passthrough` 配置是否正确
2. `layouts/partials/extend_head.html` 是否存在
3. 确认 Hugo 版本 ≥ 0.132（Docker 镜像已满足）
4. 行内公式 `$...$` 中美元符号不要紧贴中文字符（加空格）

**Q: docker compose 报错 submodule 为空**

```bash
git submodule update --init --depth=1
```

**Q: 页面 404**

- 确认文件名为 `_index.md`（带下划线 = section page）
- 确认 `baseURL` 设置正确
- 生产模式确认 `draft: false`

**Q: CSS 样式不生效**

PaperMod 通过 `assets/css/extended.css` 自动加载自定义样式（文件名必须精确匹配）。

**Q: 同步脚本报错 permission denied**

```bash
chmod +x scripts/sync-notes.sh
```

**Q: 中文文件名 PDF 无法访问**

nginx 对 URL 编码的中文路径支持良好，但建议 PDF 文件名使用英文。浏览器地址栏会自动编码中文字符。

### 有用的调试命令

```bash
# 查看 Hugo 构建输出和警告
docker run --rm -v $(pwd):/src hugomods/hugo:exts-0.140.0 hugo --minify --printPathWarnings

# 检查特定页面是否被生成
docker run --rm -v $(pwd):/src hugomods/hugo:exts-0.140.0 hugo list all

# 查看 nginx 容器日志
docker compose logs -f site

# 进入运行中的容器检查文件
docker compose exec site sh
ls /usr/share/nginx/html/
```

---

## 日常工作流总结

```
┌─────────────────────────────────────────────────────────┐
│  日常写作流程                                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 在 topics/ 写研究笔记（日常工作）                      │
│          ↓                                              │
│  2. ./scripts/sync-notes.sh（同步到网站）                 │
│          ↓                                              │
│  3. docker compose --profile dev up（本地预览）           │
│          ↓                                              │
│  4. git add & commit & push（发布）                      │
│          ↓                                              │
│  5. CI/CD 自动构建 或 服务器 docker compose up --build    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

直接写博客文章则跳过步骤 1-2，直接在 `content/blog/` 创建 `.md` 文件即可。
