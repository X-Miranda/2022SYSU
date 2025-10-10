import json
import os
import logging
from datetime import datetime, timedelta
import sys

logger = logging.getLogger(__name__)

def get_config_path():
    """获取配置文件路径"""
    if getattr(sys, 'frozen', False):
        # 打包后的程序
        base_path = sys._MEIPASS
    else:
        # 开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, 'config.json')

def get_user_config_path():
    """获取用户配置文件的路径（用于保存修改后的配置）"""
    if getattr(sys, 'frozen', False):
        # 打包后的程序，保存在程序同一目录
        base_path = os.path.dirname(sys.executable)
    else:
        # 开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, 'config.json')

def load_config():
    """加载配置文件"""
    try:
        # 首先尝试加载用户配置文件（如果有修改）
        user_config_path = get_user_config_path()
        if os.path.exists(user_config_path):
            with open(user_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载用户配置文件: {user_config_path}")
            return config
        
        # 如果没有用户配置文件，加载打包的默认配置
        default_config_path = get_config_path()
        with open(default_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"加载默认配置文件: {default_config_path}")
        return config
        
    except FileNotFoundError:
        logger.error(f"配置文件不存在")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"配置文件格式错误: {e}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise

def save_config(config):
    """保存配置到用户配置文件"""
    try:
        user_config_path = get_user_config_path()
        
        with open(user_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"配置已保存到用户文件: {user_config_path}")
        return True
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")
        return False

# 其他函数保持不变...
def get_logs_directory():
    """获取日志目录路径"""
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, 'logs')

def get_today_log_file():
    """获取今天的日志文件路径"""
    logs_dir = get_logs_directory()
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(logs_dir, f"{today}.log")

def cleanup_old_logs(max_days=None):
    """清理超过指定天数的旧日志文件"""
    try:
        if max_days is None:
            config = load_config()
            max_days = config.get('logging', {}).get('max_days', 30)
        
        logs_dir = get_logs_directory()
        if not os.path.exists(logs_dir):
            return
            
        cutoff_date = datetime.now() - timedelta(days=max_days)
        cleaned_count = 0
        
        for filename in os.listdir(logs_dir):
            if filename.endswith('.log'):
                file_path = os.path.join(logs_dir, filename)
                file_date_str = filename.replace('.log', '')
                
                try:
                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                    if file_date < cutoff_date:
                        os.remove(file_path)
                        cleaned_count += 1
                except ValueError:
                    continue
        
        if cleaned_count > 0:
            logger.info(f"已清理 {cleaned_count} 个超过 {max_days} 天的旧日志文件")
                    
    except Exception as e:
        logger.error(f"清理旧日志文件时出错: {e}")