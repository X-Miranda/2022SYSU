# 系统登录监控工具

author: X

## 项目简介

系统登录监控工具是一个基于 Python 和 PyQt5 的自动化监控系统，专门用于监控 DMS 系统的登录状态。系统能够自动检测网络连接、执行登录操作、识别验证码，并在系统异常时发送邮件告警。

## 主要功能

### 🔍 系统监控

- 自动检测网络连接状态
- 定时执行系统登录检查
- 智能识别验证码（支持 ddddocr）
- 实时监控系统可用性

### ⚙️ 配置管理

- 完整的 GUI 配置界面
- 支持管理员密码保护
- 灵活的告警阈值设置
- 维护时间配置

### 📧 告警通知

- 邮件告警功能
- 支持 SMTP 协议
- 可配置收件人列表
- 告警幂等时间控制

### 🔒 安全特性

- 管理员密码保护
- 界面锁定功能
- 配置文件加密存储



## 安装说明

### 软件要求

- Windows 7/10/11（64位）
- Microsoft Edge 浏览器和 maedgedriver 驱动（程序可自动安装）



#### 若想修改代码并打包，按照以下步骤

1. 安装python 3.12.10，将python配置到环境变量中（验证方法：win+r，输入cmd进入终端，输入python看是否能使用）

2. 下载virtualenv（命令：`pip install virtualenv`）

3. 创建虚拟环境（virtualenv 你创建的虚拟环境名，例：`virtualenv my_env`）

4. 进入打包好的环境中的scripts文件夹，激活环境（`activate.bat`）

5. 把upx放在虚拟环境中的scripts下（可选）

6. 进入源代码文件夹下，可以发现有requirements.txt，安装requirements需要的包（`pip install -r requirements.txt`）（这时候环境配置完成，将虚拟环境配置到ide里就可以修改了。）

7. 打包程序：进入源代码文件夹下，可以发现有SystemMonitor.spec，输入`pyinstaller SystemMonitor.spec`即可打包

   

### 1. 准备配置文件

配置文件已打包到软件中，格式大致如下，管理员可在UI界面中更改设置，管理员密码默认为`aionaion`

```json
{
    "admin": {
        "password": "设置你的管理员密码"
    },
    "alert": {
        "check_interval": 60,
        "alert_threshold": 3,
        "recipients": ["接收告警的邮箱@example.com"],
        "from_email": "发件邮箱@example.com"
    },
    "smtp": {
        "server": "smtp.example.com",
        "port": 587,
        "username": "发件邮箱@example.com",
        "password": "SMTP授权码",
        "ssl": false
    },
    "login": {
        "url": "https://你要监控的系统登录地址",
        "username": "登录用户名",
        "password": "登录密码",
        "username_selector": "input[name='username']",
        "password_selector": "input[name='password']",
        "captcha_selector": "input[name='captcha']",
        "login_button_selector": "button[type='submit']",
        "success_indicators": ["dashboard", "main"],
        "failure_indicators": ["登录失败", "错误"]
    },
    "browser": {
        "headless": true,
        "window_size": "1920,1080"
    },
    "maintenance": {
        "start_year": "",
        "start_month": "",
        "start_day": "",
        "start_hour": "",
        "start_minute": "",
        "start_second": "",
        "end_year": "",
        "end_month": "",
        "end_day": "",
        "end_hour": "",
        "end_minute": "",
        "end_second": "",
        "reason": ""
    },
    "logging": {
        "max_days": 30, #日志保留时间
        "level": "INFO"
    }
}
```



### 2. 配置说明

#### 管理员设置

- `password`: 程序登录和管理员操作密码

#### 告警配置

- `check_interval`: 检查间隔（秒）
- `alert_threshold`: 告警阈值（连续失败次数）
- `recipients`: 收件邮箱列表
- `from_email`: 发件邮箱

#### SMTP配置

- `server`: SMTP服务器地址
- `port`: SMTP端口
- `username`: 发件箱用户名
- `password`: SMTP授权码
- `ssl`: 是否使用SSL加密

#### 登录配置

- `url`: 系统登录页面URL
- `username`: 登录用户名
- `password`: 登录密码
- `*_selector`: 页面元素选择器（需要根据实际页面调整）
- `success_indicators`: 登录成功标识
- `failure_indicators`: 登录失败标识

#### 浏览器配置

- `headless`: 是否使用无头模式
- `window_size`: 浏览器窗口大小

#### 维护时间配置

设置系统维护时间段，在维护期间暂停监控

#### 日志配置

- `max_days`: 日志保留天数
- `level`: 日志级别



## 使用指南

### 首次启动

1. 双击运行 `SystemMonitor.exe`
2. 输入管理员密码
3. 点击"开始监控"启动监控

### 主界面功能

#### 左侧配置面板

- **管理员设置**: 修改管理员密码
- **告警配置**: 设置检查间隔和告警阈值
- **SMTP配置**: 配置邮件服务器
- **登录配置**: 设置监控系统的登录信息
- **维护时间**: 设置系统维护时段
- **日志配置**: 设置日志保留策略

#### 右侧控制面板

- **开始监控**: 启动监控任务
- **停止监控**: 停止监控任务
- **恢复监控**: 在维护期间强制恢复监控
- **测试网络**: 手动测试网络连接
- **锁定系统**: 锁定程序界面

### 监控流程

1. **网络检测**: 检查网络连接状态

2. **环境准备**: 自动下载并配置 Edge 浏览器和驱动

3. **监控循环**: 进入监控循环

   - **登录检查**: 执行系统登录操作

   - **状态判断**: 根据登录结果判断系统状态

   - **告警处理**: 在连续失败达到阈值时发送邮件告警

     

## 注意*：

1. 在运行过程中请勿移动msedgedriver文件。否则会被记录为异常。
2. 如需修改配置，修改完后请点击“保存配置”，否则可能丢失
3. 为保证账号安全，部署监控后如离开界面请锁定系统，避免信息泄露