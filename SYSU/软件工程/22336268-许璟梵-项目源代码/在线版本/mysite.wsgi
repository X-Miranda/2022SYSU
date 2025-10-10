import os
import sys
from os.path import dirname, abspath

# 设置环境变量
os.environ['SECRET_KEY'] = '32b73c5e1c3f4a9a8e5d5e2c5a8b3d7'

# 项目根目录
project_path = '/home/demonnn7/code'

# 添加项目路径到系统路径
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# 导入 Flask 应用实例
from gym import app as application

# 配置密钥
#application.secret_key = 'your_secret_key'  # 替换为强密码