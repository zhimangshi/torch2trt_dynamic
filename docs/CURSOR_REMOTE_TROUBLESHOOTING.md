# Cursor 远程服务器连接故障排除指南

本文档提供了使用 Cursor IDE 远程连接服务器时常见问题的解决方案。

## 常见错误

### 错误：IO is still pending on closed socket

```
stderr: close - IO is still pending on closed socket. read:1, write:0
```

此错误通常表示 SSH 连接在数据传输过程中被意外关闭。

## 解决方案

### 1. 检查网络连接稳定性

确保您的网络连接稳定。不稳定的网络可能导致 SSH 隧道中断。

```bash
# 测试到服务器的网络连接
ping -c 10 your-server-ip

# 检查网络延迟和丢包率
mtr your-server-ip
```

### 2. 配置 SSH 保持连接

在本地 SSH 配置文件中添加保持连接设置：

**Windows:** `C:\Users\<用户名>\.ssh\config`
**Linux/Mac:** `~/.ssh/config`

```
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
```

### 3. 在服务器端配置 SSH

编辑服务器上的 `/etc/ssh/sshd_config`：

```
ClientAliveInterval 60
ClientAliveCountMax 3
TCPKeepAlive yes
```

然后重启 SSH 服务：

```bash
sudo systemctl restart sshd
```

### 4. 手动下载并安装 Cursor 远程服务器组件

如果自动下载失败，可以手动下载并安装：

1. 从错误日志中获取下载链接：
   ```
   https://downloads.cursor.com/production/<commit-hash>/linux/x64/cursor-reh-linux-x64.tar.gz
   ```

2. 手动下载文件到服务器：
   ```bash
   # 在服务器上执行
   cd ~
   mkdir -p .cursor-server
   cd .cursor-server
   wget https://downloads.cursor.com/production/<commit-hash>/linux/x64/cursor-reh-linux-x64.tar.gz
   tar -xzf cursor-reh-linux-x64.tar.gz
   ```

3. 确保正确的目录结构：
   ```bash
   ls -la ~/.cursor-server/bin/
   ```

### 5. 检查防火墙设置

确保防火墙没有阻止必要的连接：

```bash
# 检查服务器端防火墙状态
sudo ufw status

# 如果需要，允许 SSH 连接
sudo ufw allow ssh
```

### 6. 增加 SSH 连接超时时间

对于大文件传输，可能需要增加超时时间：

```
Host your-server
    HostName your-server-ip
    User your-username
    ConnectTimeout 60
    ServerAliveInterval 30
    ServerAliveCountMax 10
```

### 7. 清理 Cursor 远程服务器缓存

有时清理缓存可以解决问题：

**在服务器端：**
```bash
rm -rf ~/.cursor-server
```

**在本地 Windows：**
```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Temp\cursor-server-*"
```

### 8. 使用代理或 VPN

如果您的网络环境有限制，可以尝试：

- 使用 VPN 连接
- 配置 SSH 代理跳板

```
Host target-server
    HostName target-ip
    User username
    ProxyJump jump-server
```

## 诊断命令

### 检查 SSH 连接详情

```bash
ssh -vvv user@server
```

### 检查服务器资源

```bash
# 检查磁盘空间
df -h

# 检查内存使用
free -h

# 检查 CPU 负载
top -bn1 | head -20
```

### 检查 Cursor 服务器进程

```bash
ps aux | grep cursor
```

## 获取帮助

如果以上方法都无法解决问题，请：

1. 收集完整的错误日志
2. 记录您的网络环境信息
3. 在 [Cursor 官方论坛](https://forum.cursor.com/) 提交问题
4. 或在项目 Issues 中报告问题

## 相关资源

- [Cursor 官方文档](https://docs.cursor.com/)
- [SSH 配置最佳实践](https://www.ssh.com/academy/ssh/config)
- [网络故障排除指南](https://www.cloudflare.com/learning/network-layer/network-troubleshooting/)
