# 如何在你的电脑上用 SourceTree 获取并使用本项目

> 这份文档面向**第一次拿到本项目**的人：从零开始，把仓库克隆到自己电脑上，用 SourceTree 图形界面完成**克隆 → 阅读/修改 → 提交 → 推送**。
>
> 覆盖 Windows 和 macOS，重点写 Windows（SourceTree 用户最多）。
> 本指南基于 **SourceTree 3.4.32** + **Git 2.50+** 实测编写，步骤中标注了实际会踩到的坑。

---

## 目录

- [一、五分钟速览：你总共要做四件事](#一五分钟速览你总共要做四件事)
- [二、第一步：安装 Git](#二第一步安装-git)
- [三、第二步：安装 SourceTree](#三第二步安装-sourcetree)
- [四、第三步（关键）：配置 SSH 免密](#四第三步关键配置-ssh-免密)
- [五、第四步：把项目克隆到本地](#五第四步把项目克隆到本地)
- [六、第五步（必做）：SourceTree 的 SSH 设置](#六第五步必做sourcetree-的-ssh-设置)
- [七、日常使用：改文件 → 提交 → 推送](#七日常使用改文件--提交--推送)
- [八、项目目录结构说明](#八项目目录结构说明)
- [九、拿到代码之后怎么跑起来](#九拿到代码之后怎么跑起来)
- [十、常见问题排查（FAQ）](#十常见问题排查faq)
- [十一、不用 SSH，改用 HTTPS 的做法](#十一不用-ssh改用-https-的做法)
- [十二、纯命令行速查（不想用图形界面时）](#十二纯命令行速查不想用图形界面时)
- [十三、想给这个项目贡献代码？](#十三想给这个项目贡献代码)

---

## 一、五分钟速览：你总共要做四件事

| 步骤 | 做什么 | 一次性还是每次 |
|---|---|---|
| 1 | 安装 **Git** | 一次性 |
| 2 | 安装 **SourceTree** | 一次性 |
| 3 | 配置 **SSH 密钥**（生成 → 复制公钥 → 贴到 GitHub） | 一次性（每台电脑各做一次） |
| 4 | **克隆项目**，之后就是「改 → 提交 → 推送」 | 克隆一次，之后反复用 |

**项目仓库地址（本指南统一用这个）：**

```bash
git@github.com:442616159/cuda-tutorial.git      # SSH 方式（推荐）
https://github.com:442616159/cuda-tutorial.git  # HTTPS 方式（见第十一章）
```

> 本项目是**公开仓库**：任何人都能克隆下来阅读。但要**推送**修改，你需要有该仓库的写权限（详见[第十三章](#十三想给这个项目贡献代码)）。

---

## 二、第一步：安装 Git

1. 打开 <https://git-scm.com/download/win>，下载 **64-bit Git for Windows Setup**
2. 双击安装，**全部保持默认**即可（默认选项已经是最合理的）
   - 唯一建议留意：`Adjusting your PATH environment` 保持默认的 **Git from the command line and also from 3rd-party software**
3. 验证安装：打开 **PowerShell** 或 **CMD**，输入

```bash
git --version
```

看到类似 `git version 2.50.1.windows.1` 就成功了。

> **为什么装了 SourceTree 还要装 Git？**
> SourceTree 自带了内嵌 Git，但内嵌版本通常较旧。用系统 Git 更新、更好排查问题。装好后再按[第六章](#六第五步必做sourcetree-的-ssh-设置)把 SourceTree 指向它。

**macOS 用户**：终端执行 `xcode-select --install`，或 `brew install git`。

---

## 三、第二步：安装 SourceTree

1. 打开 <https://www.sourcetreeapp.com/>，点 **Download for Windows** 下载安装包
2. 双击安装，一路下一步
3. 首次启动时会要求登录 **Atlassian 账号** —— **可以跳过**。不登录也能正常使用全部 Git 功能（克隆、提交、推送、拉取），本指南的做法完全不需要它

**关于安装位置（Windows）**：

SourceTree 采用用户级安装，路径固定在：

```
C:\Users\<你的用户名>\AppData\Local\SourceTree\
```

**官方安装器不提供"选择安装目录"的选项**，所以无法自定义到 D 盘等位置。（真要挪只能装完后用目录联接，风险大于收益，不建议。）

---

## 四、第三步（关键）：配置 SSH 免密

### 为什么用 SSH 而不是 HTTPS

| | SSH（推荐） | HTTPS |
|---|---|---|
| 需要输密码吗 | **不需要**，配一次永久免密 | 需要 Personal Access Token，会过期 |
| 国内网络 | 走 22 端口，**通常畅通** | 443 端口常被干扰，往往要挂代理 |
| 配置难度 | 一次性生成密钥 | 每次要管 token |

### 4.1 生成密钥

打开 **PowerShell**（macOS 用「终端」），执行：

```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
```

- 提示 `Enter file in which to save the key` → **直接回车**（用默认位置）
- 提示 `Enter passphrase` → **建议直接回车（不设口令）**，否则每次推送都要输
- 提示 `Enter same passphrase again` → 同样回车

生成后会有两个文件，都在 `C:\Users\<你的用户名>\.ssh\` 下：

| 文件 | 说明 |
|---|---|
| `id_ed25519` | **私钥** —— 绝不能给别人、绝不能上传 |
| `id_ed25519.pub` | **公钥** —— 这个才是要贴到 GitHub 的 |

### 4.2 复制公钥内容

**Windows PowerShell：**

```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | Set-Clipboard
```

执行完公钥就已经在剪贴板里了。

**macOS 终端：**

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

### 4.3 把公钥添加到 GitHub

1. 打开 <https://github.com/settings/ssh/new>
2. **Title**：随便填，建议写清是哪台电脑，例如 `我的笔记本-Windows`
3. **Key type**：选 `Authentication Key`
4. **Key**：把刚才复制的公钥粘进去（应以 `ssh-ed25519 AAAA...` 开头，以你的邮箱结尾）
5. 点 **Add SSH key**

### 4.4 验证是否配置成功

在 PowerShell / 终端执行：

```bash
ssh -T git@github.com
```

看到这样的回应就说明**成功了**：

```
Hi 你的用户名! You've successfully authenticated, but GitHub does not provide shell access.
```

> 如果第一次连接问 `Are you sure you want to continue connecting (yes/no)?` → 输入 `yes` 回车。
>
> 如果报 `Permission denied (publickey)` → 公钥没加上，或加到了别的 GitHub 账号上，回到 4.3 重做。

### 4.5 重要：每台电脑一把钥匙

- **不要在电脑之间复制私钥**（`id_ed25519`）
- **每台电脑各自生成一把**，把各自的公钥都加到**同一个** GitHub 账号下
- 这样某台电脑的密钥泄露或丢失时，只需在 GitHub 上单独删掉那一把，其他电脑不受影响

---

## 五、第四步：把项目克隆到本地

### 方法 A：用 SourceTree 图形界面克隆（推荐新手）

1. 打开 SourceTree，点顶部工具栏的 **「克隆」**（Clone）
2. 填写三个字段：

| 字段 | 填什么 |
|---|---|
| **源路径 / URL** | `git@github.com:442616159/cuda-tutorial.git` |
| **目标路径** | 选一个存放位置，例如 `D:\Projects` |
| **名字** | 自动填 `cuda-tutorial`，不用改 |

3. 点 **「克隆」**，等待完成

> **路径建议**：目标路径尽量**不要带中文、不要带空格、不要放在系统盘**（例如用 `D:\Projects` 而不是 `C:\Users\张三\桌面\我的项目`）。CUDA 编译工具链对中文路径的支持时好时坏，能避就避。

### 方法 B：命令行克隆 + SourceTree 添加

```bash
cd D:\Projects
git clone git@github.com:442616159/cuda-tutorial.git
```

然后在 SourceTree 里：左下角 **`+`** → **「添加」** → 浏览到 `D:\Projects\cuda-tutorial` → 添加。

---

## 六、第五步（必做）：SourceTree 的 SSH 设置

**这一步不做，推送大概率会失败 —— 这是 SourceTree 最常见的坑。**

打开：**工具 → 选项 → 通用**，找到 **「SSH 客户端」**，设置为：

```
✅ OpenSSH
❌ 不要选 PuTTY / Plink
```

**为什么？**
> SourceTree 如果配置成用 PuTTY 连接 SSH，而 PuTTY 只认 `.ppk` 格式的密钥；你第四章生成的是 **OpenSSH 格式**密钥（首行是 `-----BEGIN OPENSSH PRIVATE KEY-----`），PuTTY 用不了，于是推送时报认证失败。
>
> 选 **OpenSSH** 则会调用系统自带的 `ssh.exe` 配合你现有的密钥，一切正常。

**顺带建议的两项设置**（同一个「选项」窗口里）：

| 位置 | 设置 | 原因 |
|---|---|---|
| 选项 → Git → **Git 版本** | 选 **系统** | 用你刚装的 Git，版本更新 |
| 选项 → 通用 → **语言** | 选 **简体中文** | 界面更顺手 |

---

## 七、日常使用：改文件 → 提交 → 推送

### 7.1 提交（Commit）—— 保存到本地

1. 左侧边栏点 **「文件状态」**
2. 中间会列出所有改动过的文件，**勾选**你要提交的那些
3. 下方输入框写**提交说明**（一句话说清改了什么，例如「完成第 3 章归约练习」）
4. 点顶部 **「提交」**

> ⚠️ **提交只是保存到你的电脑上，还没有上传。** 很多人卡在这里以为已经传上去了。

### 7.2 推送（Push）—— 上传到 GitHub

点顶部 **「推送」** → 确认推送的分支是 `main` → 确定。

看到类似 `abc1234..def5678  main -> main` 就是成功了。

### 7.3 拉取（Pull）/ 获取（Fetch）—— 把远端的更新拿下来

| 按钮 | 作用 | 建议 |
|---|---|---|
| **获取**（Fetch） | 只把远端最新状态**下载下来看看**，不动你的文件 | 安全，动手前先点一下 |
| **拉取**（Pull） | 获取 + **合并**到你当前分支 | 日常用这个 |

**推荐的日常顺序**：先 **拉取** → 有更新就先合进来 → 改代码 → **提交** → **推送**。

### 7.4 查看历史和改动

左侧边栏 **「History」**：能看到每一次提交、改了哪些文件、具体改了哪几行（带颜色标注的 diff）。

---

## 八、项目目录结构说明

克隆下来之后，你会看到：

```
cuda-tutorial/
├─ README.md                    ← 教程总入口，先看这个
├─ CONTRIBUTING.md              ← 贡献指南
├─ docs/                        ← 教程正文（按章分文件）
│  ├─ CUDA学习步骤与教程大纲.md   ← 学习路线总纲
│  ├─ ch00_环境搭建与第一个程序.md
│  ├─ ch01_编程模型与线程索引.md
│  ├─ ch02_内存层次与访存优化.md
│  ├─ ch03_同步通信与原子操作.md
│  ├─ ch04_性能分析与优化方法论.md
│  ├─ ch05_核心算法模式实战.md
│  ├─ ch06_CUDA高级特性.md
│  ├─ ch07_生态库与工程化.md
│  └─ ch08_综合项目实战.md
├─ code/                        ← 全部示例代码
│  ├─ common/                   ← 公共头文件（计时、错误检查等）
│  ├─ lessons/                  ← 各章配套示例（按章分目录）
│  └─ projects/                 ← 综合项目
│     ├─ sgemm/                 ← 矩阵乘法渐进优化 v1~v7
│     ├─ raytracer/             ← 光线追踪
│     ├─ nbody/                 ← N 体模拟
│     └─ cmake_skeleton/        ← CMake 工程模板
└─ tools/                       ← 辅助脚本（环境自检、清理等）
```

---

## 九、拿到代码之后怎么跑起来

> 前提：一台 **NVIDIA 显卡**的机器 + 装好 **CUDA Toolkit**。环境搭建的完整步骤见 [`docs/ch00_环境搭建与第一个程序.md`](docs/ch00_环境搭建与第一个程序.md)。

**先跑环境自检脚本**，确认工具链齐了：

```powershell
# Windows
.\tools\check_env.ps1
```

```bash
# Linux / WSL / macOS（无 NVIDIA 显卡时部分项会失败，属正常）
bash ./tools/check_env.sh
```

**编译单个示例（最简单的方式）：**

```bash
cd code/lessons/ch00_hello
nvcc hello.cu -o hello
./hello
```

**用 CMake 编译整套：**

```bash
cd code
cmake -B build
cmake --build build
```

**用 Makefile（部分示例提供）：**

```bash
cd code/projects/sgemm
make
```

---

## 十、常见问题排查（FAQ）

| 现象 | 原因 | 解决办法 |
|---|---|---|
| `Permission denied (publickey)` | 公钥没加到 GitHub，或加到了别的账号 | 重做 [4.3](#43-把公钥添加到-github)、[4.4](#44-验证是否配置成功) |
| SourceTree 推送报密钥/认证错误 | **SSH 客户端设成了 PuTTY** | 按[第六章](#六第五步必做sourcetree-的-ssh-设置)改成 **OpenSSH** |
| 每次推送都要输账号密码 | 用的是 HTTPS 地址 | 换成 SSH 地址（本项目推荐） |
| `Repository not found` | 地址写错，或没有该仓库权限 | 核对 `442616159/cuda-tutorial` 拼写 |
| `fatal: detected dubious ownership` | 仓库属主与当前用户不一致 | 执行 `git config --global --add safe.directory <仓库路径>` |
| 中文文件名显示成 `\344\270\255\346\226\207` 这类乱码 | Git 的 quotepath 转义 | 执行 `git config --global core.quotepath false` |
| `.sh` 脚本在 Linux 报 `bad interpreter` | 换行符被改成了 CRLF | 本项目已用 `.gitattributes` 钉死 `*.sh` 为 LF，**不要自行修改换行符** |
| 国内网络推送卡住/超时 | HTTPS 被干扰 | 用 SSH（走 22 端口，通常畅通），即本指南的方案 |
| SourceTree 打开时弹「`git status` 失败：工作目录无效」 | SourceTree 自身的会话状态小毛病 | **点确定即可，不影响任何功能** |
| 拉取时提示有冲突 | 你改的和远端改的是同一处 | 在 SourceTree 里按提示逐块选择保留哪边，解决后提交 |

---

## 十一、不用 SSH，改用 HTTPS 的做法

如果你所处的网络环境 SSH 也不通，可以退回 HTTPS：

1. **克隆地址**改成：`https://github.com/442616159/cuda-tutorial.git`
2. GitHub **早已不支持账号密码**推送，需要 **Personal Access Token**：
   - 打开 <https://github.com/settings/tokens> → **Generate new token (classic)**
   - 勾选 `repo` 权限 → 生成 → **把 token 复制保存好**（只显示一次）
3. 推送时，用户名填你的 GitHub 用户名，**密码栏粘贴刚才的 token**
4. SourceTree 会把它记住

**HTTPS 的代价**：token 会过期（过期后要重新生成）；国内直连 `github.com:443` 常常被重置，**需要挂代理**才能用。

---

## 十二、纯命令行速查（不想用图形界面时）

```bash
# 克隆（只需一次）
git clone git@github.com:442616159/cuda-tutorial.git

# 看当前状态（改了什么、在哪个分支）
git status

# 提交
git add -A                        # 暂存所有改动（也可 git add 具体文件）
git commit -m "提交说明"

# 与远端同步
git pull                          # 拉取并合并
git fetch                         # 只获取不动工作区

# 上传
git push

# 看历史
git log --oneline --graph -20
```

---

## 十三、想给这个项目贡献代码？

分两种情况：

**情况一：用你自己的其他电脑（同一个 GitHub 账号）**

按本指南做完第四、六章，就能直接提交推送。

**情况二：别人的电脑 / 别人的 GitHub 账号**

本仓库是**公开**的，任何人都能克隆阅读，但**默认没有推送权限**。两种参与方式：

| 方式 | 怎么做 |
|---|---|
| **加为协作者**（仓库主人操作） | 仓库主人到 `Settings → Collaborators` 邀请你的账号，接受后你就能直接推送 |
| **Fork + Pull Request**（推荐，适合外部贡献） | ① 在 GitHub 页面点 **Fork**；② 克隆**你自己 fork 出来的**仓库；③ 改完推送到你的 fork；④ 在 GitHub 上发起 **Pull Request** 给原仓库 |

详情参考仓库里的 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

---

## 附：本指南对应的版本信息

| 项目 | 版本 / 说明 |
|---|---|
| SourceTree | 3.4.32（Windows 用户级安装） |
| Git | 2.50+（系统 Git） |
| 推荐认证方式 | SSH（`ed25519` 密钥） |
| 仓库默认分支 | `main` |
| 仓库地址 | <https://github.com/442616159/cuda-tutorial> |

---

**参考资料**

- Git for Windows：<https://git-scm.com/download/win>
- SourceTree 官方下载：<https://www.sourcetreeapp.com/>
- GitHub SSH 密钥设置：<https://docs.github.com/cn/authentication/connecting-to-github-with-ssh>
- 本项目贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)
