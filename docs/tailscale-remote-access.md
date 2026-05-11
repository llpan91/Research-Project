# Tailscale 远程访问配置指南

通过 Tailscale 组建虚拟内网，实现家里访问办公室电脑上的 Hugo 开发服务器（或 SSH）。

---

## 前置条件

- 两台设备（办公室 + 家里）均可联网
- 一个 Google / Microsoft / GitHub 账号用于 Tailscale 登录

---

## 一、安装 Tailscale

### Linux（Ubuntu/Debian）

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

### macOS

App Store 搜索 **Tailscale** 直接安装。

### Windows

下载安装：https://tailscale.com/download/windows

---

## 二、办公室电脑配置

```bash
# 1. 启动 Tailscale（首次会弹出浏览器要求登录）
sudo tailscale up

# 2. 查看本机 Tailscale IP
tailscale ip
# 输出类似：100.64.x.x

# 3. 启动 Hugo 开发服务器（绑定所有网卡）
cd ~/Documents/Research-Project/site
hugo server -D --disableFastRender --bind 0.0.0.0
```

> 注意：必须加 `--bind 0.0.0.0`，否则 Hugo 只监听 localhost，远程无法访问。

---

## 三、家里电脑配置

```bash
# 1. 安装 Tailscale（同上）
curl -fsSL https://tailscale.com/install.sh | sh

# 2. 用相同账号登录
sudo tailscale up

# 3. 确认能看到办公室设备
tailscale status
```

---

## 四、使用方式

### 浏览器访问 Hugo 站点

```
http://<办公室Tailscale-IP>:1313
```

例如：`http://100.64.1.2:1313`

### SSH 远程登录

```bash
ssh mi@<办公室Tailscale-IP>
```

### 查看设备 IP

```bash
# 查看自己的 IP
tailscale ip

# 查看所有在线设备
tailscale status
```

---

## 五、可选优化

### 使用 MagicDNS（用设备名代替 IP）

Tailscale 默认启用 MagicDNS，可以直接用设备名访问：

```
http://mi-precision-3660:1313
ssh mi@mi-precision-3660
```

设备名在 https://login.tailscale.com/admin/machines 中查看和修改。

### 设置开机自启

Linux 安装后 Tailscale 默认以 systemd 服务运行，重启后自动连接：

```bash
# 确认服务状态
sudo systemctl status tailscaled
```

### 长期运行 Hugo（后台）

```bash
nohup hugo server -D --disableFastRender --bind 0.0.0.0 > /dev/null 2>&1 &
```

---

## 常见问题

**Q: 两台设备必须在同一网络吗？**

不需要。Tailscale 会自动穿透 NAT，两台设备在任何网络下都能互通。

**Q: 免费吗？**

个人使用免费，支持最多 100 台设备。

**Q: 速度如何？**

大多数情况下 Tailscale 建立点对点直连（不经中转），速度接近直连。

**Q: 需要公网 IP 吗？**

不需要。这正是 Tailscale 的核心优势。
