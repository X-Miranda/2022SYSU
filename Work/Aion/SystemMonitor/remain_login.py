import shutil
import subprocess
import time
import zipfile

import certifi
import ddddocr
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException

from PIL import Image, ImageFilter
import io
import base64
import re
from selenium.webdriver.edge.service import Service
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
from datetime import datetime
import os

import sys
import urllib.request
from urllib.error import URLError
import logging
from logging.handlers import TimedRotatingFileHandler
# 导入配置管理器
from config_manager import load_config, get_logs_directory, get_today_log_file, cleanup_old_logs


# 处理 stdout
if sys.stdout is not None and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 处理 stderr
if sys.stderr is not None and sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')


def resource_path(relative):
    base = getattr(sys, '_MEIPASS', os.path.abspath('.'))
    return os.path.join(base, relative)


# 设置日志
def setup_logging():
    """设置日志系统"""
    logger = logging.getLogger(__name__)
    
    # 关键：如果已有处理器，不再重复添加
    if logger.handlers:
        return logger

    # 设置日志级别
    logger.setLevel(logging.INFO)
    
    # 创建日志目录
    logs_dir = get_logs_directory()
    os.makedirs(logs_dir, exist_ok=True)
    
    # 清理旧日志（每次启动时清理）
    cleanup_old_logs()
    
    # 修改这里：每天创建新的日志文件，文件名就是当天日期
    log_file = get_today_log_file()  # 返回如 "2025-10-05.log"
    
    # 创建文件处理器 - 不使用TimedRotatingFileHandler，我们自己管理
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    # 设置文件处理器格式
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 设置一些第三方库的日志级别，避免过于冗长
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    
    return logger


logger = setup_logging()


class SystemMonitor:
    def __init__(self, config=None):
        # 如果没有传入配置，则从配置文件加载
        if config is None:
            self.config = load_config()
        else:
            self.config = config
        self.last_alert_time = None
        self.alert_cooldown = 3600  # 默认1小时冷却
        self.consecutive_failures = 0
        self.is_in_maintenance = False
        self.force_resume = False
        self.total = 0  # 整个监控周期的总识别次数
        self.correct = 0  # 整个监控周期的正确识别次数
        self.alert_threshold_multiplier = 1  # 新增：告警阈值乘数
        self.original_alert_threshold = self.config['alert']['alert_threshold']  # 保存原始阈值
        self.driver_path = None  # 添加driver路径存储
        self.driver_checked = False  # 添加driver检查标志
        self.current_automation = None
        self.network_available = True  # 添加网络检查标志

    def check_and_download_edge(self):
        """检查并自动下载Edge浏览器"""
        try:
            # 检查是否已安装Edge
            if self.is_edge_installed():
                return True
            
            logger.info("未检测到Edge浏览器，开始下载...")
            return self.download_edge()
            
        except Exception as e:
            logger.error(f"检查/下载Edge浏览器失败: {e}")
            return False

    def is_edge_installed(self):
        """检查Edge浏览器是否已安装"""
        try:
            # 方法1: 检查注册表
            try:
                proc = subprocess.run(
                    ["reg", "query", r"HKEY_CURRENT_USER\Software\Microsoft\Edge\BLBeacon", "/v", "version"],
                    capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
                )
                if proc.returncode == 0:
                    return True
            except:
                pass
            
            # 方法2: 检查常见安装路径
            edge_paths = [
                os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
                os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe")
            ]
            
            for path in edge_paths:
                if os.path.exists(path):
                    logger.info(f"找到Edge浏览器: {path}")
                    return True
                
            return False
            
        except Exception as e:
            logger.warning(f"检查Edge安装状态时出错: {e}")
            return False

    def download_edge(self):
        """下载并安装Edge浏览器"""
        try:
            # Edge离线安装包下载URL
            edge_download_url = "https://go.microsoft.com/fwlink/?linkid=2109047&Channel=Stable&language=zh-cn"
            
            # 临时文件路径
            temp_dir = os.path.join(os.path.dirname(sys.argv[0]), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            installer_path = os.path.join(temp_dir, "MicrosoftEdgeSetup.exe")
            
            logger.info(f"开始下载Edge安装包: {edge_download_url}")
            
            # 下载安装包
            try:
                response = requests.get(edge_download_url, stream=True, timeout=300, verify=certifi.where())
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(installer_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                if int(progress) % 10 == 0:  # 每10%记录一次
                                    logger.info(f"下载进度: {progress:.1f}%")
                
                logger.info("Edge安装包下载完成")
                
            except Exception as e:
                logger.error(f"下载Edge安装包失败: {e}")
                # 清理临时文件
                if os.path.exists(installer_path):
                    os.remove(installer_path)
                return False
            
            # 静默安装Edge
            logger.info("开始静默安装Edge浏览器...")
            try:
                # 使用静默安装参数
                install_process = subprocess.run(
                    [installer_path, "--silent", "--install", "standalone"],
                    timeout=300,  # 5分钟超时
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                if install_process.returncode == 0:
                    logger.info("Edge浏览器安装成功")
                    
                    # 等待系统注册表更新
                    time.sleep(5)
                    
                    # 验证安装是否成功
                    if self.is_edge_installed():
                        logger.info("Edge浏览器安装验证成功")
                        
                        # 清理安装包
                        try:
                            os.remove(installer_path)
                            os.rmdir(temp_dir)
                        except:
                            pass
                        
                        return True
                    else:
                        logger.error("Edge浏览器安装后验证失败")
                else:
                    logger.error(f"Edge安装过程返回错误代码: {install_process.returncode}")
                    logger.error(f"安装错误输出: {install_process.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error("Edge安装超时")
            except Exception as e:
                logger.error(f"安装Edge时发生异常: {e}")
            
            # 安装失败，清理临时文件
            try:
                if os.path.exists(installer_path):
                    os.remove(installer_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except:
                pass
                
            return False
            
        except Exception as e:
            logger.error(f"下载安装Edge浏览器过程中发生错误: {e}")
            return False

    def prepare_environment(self):
        """准备完整的运行环境（Edge浏览器 + Driver）"""
        # 第一步：检查并下载Edge浏览器
        if not self.check_and_download_edge():
            logger.error("Edge浏览器环境准备失败")
            return False
        
        # 第二步：准备Driver（直接调用原有方法）
        return self.prepare_driver()

    def check_network_connection(self, timeout=10):
        """检查网络连接状态"""
        test_urls = [
            "https://www.baidu.com",
            "https://www.qq.com",
            "https://www.163.com"
        ]
        
        for url in test_urls:
            try:
                logger.info(f"尝试连接: {url}")
                response = requests.get(url, timeout=timeout, verify=False)
                if response.status_code == 200:
                    logger.info(f"网络连接正常: {url}")
                    self.network_available = True
                    return True
            except requests.exceptions.RequestException as e:
                logger.warning(f"连接 {url} 失败: {e}")
            except Exception as e:
                logger.warning(f"检查 {url} 时发生异常: {e}")
        
        self.network_available = False
        return False

    def prepare_driver(self):
        """准备driver，检查版本并获取路径"""
        try:
            # 获取Edge版本
            version = self.get_edge_version()
            major_version = version.split(".")[0]
            logger.info(f"检测到本机 Edge 版本: {version}")

            # 确认driver路径
            self.driver_path = os.path.join(os.path.dirname(sys.argv[0]), "msedgedriver.exe")

            # 检查driver是否存在 & 版本匹配
            if not self.is_driver_matching(self.driver_path, major_version):
                logger.info("未找到匹配的 EdgeDriver，开始下载...")
                self.download_driver(major_version, self.driver_path)

            self.driver_checked = True
            return True

        except Exception as e:
            logger.error(f"准备driver失败: {e}")
            return False

    def get_edge_version(self):
        """获取本机 Edge 浏览器版本"""
        try:
            proc = subprocess.run(
                ["reg", "query", r"HKEY_CURRENT_USER\Software\Microsoft\Edge\BLBeacon", "/v", "version"],
                capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW
            )
            output = proc.stdout
            version = output.strip().split()[-1]
            return version
        except Exception as e:
            logger.error(f"获取 Edge 版本失败: {e}")
            raise

    def is_driver_matching(self, driver_path, major_version):
        """检查现有 driver 是否与 Edge 主版本匹配"""
        if not os.path.exists(driver_path):
            logger.info(f"Driver 文件不存在: {driver_path}")
            return False

        try:
            # 运行 driver 版本命令
            proc = subprocess.run([driver_path, "--version"], capture_output=True, text=True, timeout=30,
                                  creationflags=subprocess.CREATE_NO_WINDOW)

            if proc.returncode != 0:
                logger.warning(f"运行 driver 版本命令失败: {proc.stderr}")
                return False

            output = proc.stdout.strip()

            # 多种可能的版本号匹配模式
            version_patterns = [
                r'MSEdgeDriver\s+(\d+\.\d+\.\d+\.\d+)',  # MSEdgeDriver 140.0.2210.91
                r'(\d+\.\d+\.\d+\.\d+)',  # 直接匹配版本号
                r'Microsoft Edge WebDriver\s+(\d+\.\d+\.\d+\.\d+)'  # 其他可能的格式
            ]

            driver_version = None
            for pattern in version_patterns:
                match = re.search(pattern, output)
                if match:
                    driver_version = match.group(1)
                    break

            if not driver_version:
                logger.warning(f"无法从输出中解析版本号: {output}")
                return False

            logger.info(f"解析到的 Driver 版本: {driver_version}")

            # 提取主版本号
            driver_major = driver_version.split('.')[0]
            logger.info(f"Driver 主版本: {driver_major}, 浏览器主版本: {major_version}")

            return driver_major == major_version

        except subprocess.TimeoutExpired:
            logger.warning("获取 driver 版本超时")
            return False
        except Exception as e:
            logger.warning(f"检查 driver 版本时发生异常: {e}")
            return False

    def download_driver(self, major_version, driver_path):
        """下载并解压 EdgeDriver"""
        try:
            base = "https://msedgewebdriverstorage.blob.core.windows.net/edgewebdriver"
            version = None
            try:
                url = f"{base}/LATEST_RELEASE_{major_version}_WINDOWS"
                version = requests.get(url, timeout=10, verify=certifi.where()).text.strip()
            except Exception as e:
                logger.warning(f"尝试 {base} 获取版本失败: {e}")
            if not version:
                raise RuntimeError("无法获取 EdgeDriver 版本号")

            logger.info(f"获取到 EdgeDriver 最新版本: {version}")

            # 下载 zip
            try:
                zip_url = f"{base}/{version}/edgedriver_win64.zip"
                logger.info(f"尝试下载 {zip_url}")
                zip_path = os.path.join(os.path.dirname(driver_path), "edgedriver.zip")
                with requests.get(zip_url, stream=True, timeout=30,verify=certifi.where()) as r:
                    with open(zip_path, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
            except Exception as e:
                logger.warning(f"下载失败: {e}")
                raise

            # 解压
            with zipfile.ZipFile(zip_path, "r") as z:
                for file in z.namelist():
                    if file.endswith("msedgedriver.exe"):
                        # 先解压到临时位置
                        temp_path = os.path.join(os.path.dirname(driver_path), "temp_msedgedriver.exe")
                        with z.open(file) as zf, open(temp_path, "wb") as f:
                            shutil.copyfileobj(zf, f)

                        # 如果目标文件已存在，先删除
                        if os.path.exists(driver_path):
                            os.remove(driver_path)

                        # 移动文件到目标位置
                        shutil.move(temp_path, driver_path)
                        break

            # 清理临时文件
            if os.path.exists(zip_path):
                os.remove(zip_path)

            logger.info(f"EdgeDriver {version} 下载并解压完成")

        except Exception as e:
            logger.error(f"下载 EdgeDriver 失败: {e}")
            # 清理可能残留的文件
            if 'zip_path' in locals() and os.path.exists(zip_path):
                os.remove(zip_path)
            raise

    def cleanup_driver(self):
        """清理driver相关资源"""
        self.driver_path = None
        self.driver_checked = False
        logger.info("Driver资源已清理")

    def set_force_resume(self, force_resume):
        """设置强制恢复标志"""
        self.force_resume = force_resume
        if force_resume:
            logger.info("强制恢复监控，跳过维护检查")

    def is_system_under_maintenance(self):
        if self.force_resume:
            return False

        m = self.config.get('maintenance', {})

        # 获取维护时间
        start_year = m.get('start_year', '').strip()
        start_month = m.get('start_month', '').strip()
        start_day = m.get('start_day', '').strip()
        start_hour = m.get('start_hour', '').strip()
        start_minute = m.get('start_minute', '').strip()
        start_second = m.get('start_second', '').strip()

        end_year = m.get('end_year', '').strip()
        end_month = m.get('end_month', '').strip()
        end_day = m.get('end_day', '').strip()
        end_hour = m.get('end_hour', '').strip()
        end_minute = m.get('end_minute', '').strip()
        end_second = m.get('end_second', '').strip()

        # 校验是否完整
        if not all([start_year, start_month, start_day, start_hour, start_minute, start_second,
                    end_year, end_month, end_day, end_hour, end_minute, end_second]):
            return False

        try:
            start_str = f"{start_year}-{start_month.zfill(2)}-{start_day.zfill(2)} " \
                        f"{start_hour.zfill(2)}:{start_minute.zfill(2)}:{start_second.zfill(2)}"
            end_str = f"{end_year}-{end_month.zfill(2)}-{end_day.zfill(2)} " \
                      f"{end_hour.zfill(2)}:{end_minute.zfill(2)}:{end_second.zfill(2)}"

            start_time_obj = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
            end_time_obj = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')

            current_time = datetime.now()

            if start_time_obj <= current_time <= end_time_obj:
                reason = m.get('reason', '系统维护')
                logger.info(f"系统处于维护时段: {reason}")
                return True
        except Exception as e:
            logger.warning(f"解析维护时间时出错: {e}")

        return False

    def send_email_alert(self, subject, message, is_critical=False):
        try:
            if (self.last_alert_time and
                    (datetime.now() - self.last_alert_time).total_seconds() < self.alert_cooldown and
                    not is_critical):
                logger.info("在冷却时间内，跳过邮件发送")
                return False

            smtp_config = self.config['smtp']
            alert_config = self.config['alert']

            msg = MIMEMultipart()
            msg['From'] = alert_config['from_email']
            msg['To'] = ', '.join(alert_config['recipients'])
            msg['Subject'] = f"[系统监控] {subject}"
            msg.attach(MIMEText(message, 'plain', 'utf-8'))

            if smtp_config['ssl']:
                server = smtplib.SMTP_SSL(smtp_config['server'], smtp_config['port'])
            else:
                server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])

            server.login(smtp_config['username'], smtp_config['password'])
            server.sendmail(alert_config['from_email'], alert_config['recipients'], msg.as_string())
            server.quit()

            self.last_alert_time = datetime.now()

            # 移除这里的阈值调整逻辑，现在在调用处处理
            logger.info("邮件告警发送成功")
            return True

        except Exception as e:
            logger.error(f"发送邮件失败: {e}")
            return False

    def check_system_status(self):
        """检查系统状态"""
        # 检查是否处于维护时段
        if self.is_system_under_maintenance():
            if not self.is_in_maintenance:
                # 刚进入维护状态，清空异常记录和重置阈值
                self.consecutive_failures = 0
                self.alert_threshold_multiplier = 1  # 重置乘数
                logger.info("进入维护时段，清空异常记录和告警阈值，暂停监控")
            self.is_in_maintenance = True
            if self.force_resume:
                logger.info("强制恢复监控，跳过维护检查")
                self.is_in_maintenance = False
            else:
                return True

        # 如果刚从维护状态恢复
        if self.is_in_maintenance:
            logger.info("维护时段结束，恢复监控")
            self.is_in_maintenance = False

        logger.info("开始检查系统状态...")

        automation = None
        try:
            # 创建LoginAutomation实例
            self.current_automation = LoginAutomation(
                monitor_ref=self,
                headless=self.config['browser']['headless'],
                window_size=self.config['browser']['window_size'],
                driver_path=self.driver_path
            )
            automation = self.current_automation

            login_config = self.config['login']
            result = automation.login(
                login_config['url'],
                login_config['username'],
                login_config['password'],
                login_config['username_selector'],
                login_config['password_selector'],
                login_config['captcha_selector'],
                login_config['login_button_selector'],
                login_config.get('success_indicators', []),
                login_config.get('failure_indicators', []),
            )

            # 统一处理所有失败情况，都参与阈值乘法增加
            if result == "page_load_error":
                # 页面加载失败
                self.consecutive_failures += 1
                current_threshold = self.original_alert_threshold * self.alert_threshold_multiplier

                accuracy = (self.correct / self.total * 100) if self.total > 0 else 0
                logger.error(
                    f"❌ 系统状态异常：页面加载失败，连续失败次数: {self.consecutive_failures}/{current_threshold}")
                logger.info(f"📊 累计OCR准确率: {accuracy:.2f}% ({self.correct}/{self.total})")

                # 检查是否达到告警阈值
                if self.consecutive_failures >= current_threshold:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    subject = f"系统异常通知"
                    message = f"""
        系统监控告警

        时间: {current_time}
        系统: {login_config['url']}

        系统访问失败，页面无法加载，请及时检查系统状态！
        连续失败次数: {self.consecutive_failures}
        当前告警阈值: {current_threshold}
        """
                    if self.send_email_alert(subject, message, is_critical=True):
                        # 邮件发送成功后，幂等增加告警阈值
                        self.alert_threshold_multiplier *= 2
                        self.consecutive_failures = 0  # 重置连续失败计数
                        new_threshold = self.original_alert_threshold * self.alert_threshold_multiplier
                        logger.info(f"⚠️ 页面加载失败告警发送成功，告警阈值调整为: {new_threshold} 次连续失败")

                return False
            
            elif result == "account_error":
                # 账号密码错误，停止监控并通知用户
                logger.error("❌ 账号或密码无效，无法登录")
                # 发送信号给UI界面（需要在MonitorThread中添加相应信号）
                if hasattr(self, 'account_error_signal'):
                    self.account_error_signal.emit("账号或密码无效，无法登录")
                return "account_error"

            elif result is True:
                # 登录成功
                self.consecutive_failures = 0
                self.alert_threshold_multiplier = 1  # 重置乘数
                current_threshold = self.original_alert_threshold * self.alert_threshold_multiplier

                accuracy = (self.correct / self.total * 100) if self.total > 0 else 0
                logger.info(f"✅ 系统状态正常：登录成功，告警阈值重置为: {current_threshold} 次")
                logger.info(f"📊 累计OCR准确率: {accuracy:.2f}% ({self.correct}/{self.total})")
                return True

            else:
                # 登录失败
                self.consecutive_failures += 1
                current_threshold = self.original_alert_threshold * self.alert_threshold_multiplier

                accuracy = (self.correct / self.total * 100) if self.total > 0 else 0
                logger.warning(f"❌ 系统状态异常：登录失败，连续失败次数: {self.consecutive_failures}/{current_threshold}")
                logger.info(f"📊 累计OCR准确率: {accuracy:.2f}% ({self.correct}/{self.total})")

                if self.consecutive_failures >= current_threshold:
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    subject = f"系统异常通知"
                    message = f"""
        系统监控告警

        时间: {current_time}
        系统: {login_config['url']}

        系统登录失败，请及时检查系统状态！
        连续失败次数: {self.consecutive_failures}
        当前告警阈值: {current_threshold}
        """
                    if self.send_email_alert(subject, message, is_critical=True):
                        # 邮件发送成功后，幂等增加告警阈值
                        self.alert_threshold_multiplier *= 2
                        self.consecutive_failures = 0  # 重置连续失败计数
                        new_threshold = self.original_alert_threshold * self.alert_threshold_multiplier
                        logger.info(f"⚠️ 登录失败告警发送成功，告警阈值调整为: {new_threshold} 次连续失败")

                return False

        except TimeoutException as e:
            # 专门处理超时异常
            logger.error(f"❌ 页面加载超时，视为系统异常: {e}")
            self.consecutive_failures += 1
            current_threshold = self.original_alert_threshold * self.alert_threshold_multiplier

            accuracy = (self.correct / self.total * 100) if self.total > 0 else 0
            logger.warning(f"❌ 系统状态异常：页面加载超时，连续失败次数: {self.consecutive_failures}/{current_threshold}")
            logger.info(f"📊 累计OCR准确率: {accuracy:.2f}% ({self.correct}/{self.total})")

            if self.consecutive_failures >= current_threshold:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                subject = f"系统异常通知"
                message = f"""
        系统监控告警

        时间: {current_time}
        系统: {login_config['url']}

        系统访问超时，页面无法加载，请及时检查系统状态！
        连续失败次数: {self.consecutive_failures}
        当前告警阈值: {current_threshold}
        """
                if self.send_email_alert(subject, message, is_critical=True):
                    # 邮件发送成功后，幂等增加告警阈值
                    self.alert_threshold_multiplier *= 2
                    self.consecutive_failures = 0
                    new_threshold = self.original_alert_threshold * self.alert_threshold_multiplier
                    logger.info(f"⚠️ 超时异常告警发送成功，告警阈值调整为: {new_threshold} 次连续失败")

            return False

        except Exception as e:
            # 其他异常
            self.consecutive_failures += 1
            logger.error(f"❌ 系统检查过程中发生异常: {e}")
            current_threshold = self.original_alert_threshold * self.alert_threshold_multiplier

            accuracy = (self.correct / self.total * 100) if self.total > 0 else 0
            logger.warning(f"❌ 系统状态异常：发生异常，连续失败次数: {self.consecutive_failures}/{current_threshold}")
            logger.info(f"📊 累计OCR准确率: {accuracy:.2f}% ({self.correct}/{self.total})")

            if self.consecutive_failures >= current_threshold:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                subject = f"系统异常通知"
                message = f"""
        系统监控告警

        时间: {current_time}
        系统: {login_config['url']}

        系统检查过程中发生异常: {str(e)}
        请及时检查系统状态！
        连续失败次数: {self.consecutive_failures}
        当前告警阈值: {current_threshold}
        """
                if self.send_email_alert(subject, message, is_critical=True):
                    # 邮件发送成功后，幂等增加告警阈值
                    self.alert_threshold_multiplier *= 2
                    self.consecutive_failures = 0
                    new_threshold = self.original_alert_threshold * self.alert_threshold_multiplier
                    logger.info(f"⚠️ 系统异常告警发送成功，告警阈值调整为: {new_threshold} 次连续失败")

            return False

        finally:
            # 确保无论如何都会关闭automation
            if automation:
                try:
                    automation.close()
                except Exception as e:
                    logger.warning(f"关闭automation时出错: {e}")
            self.current_automation = None


class LoginAutomation:
    def __init__(self, monitor_ref, headless=True, window_size="1920,1080", driver_path=None):
        edge_options = Options()
        edge_options.use_chromium = True
        self.monitor_ref = monitor_ref  # 引用 SystemMonitor

        if headless:
            edge_options.add_argument("--headless=new")
            edge_options.add_argument("--disable-gpu")
            edge_options.add_argument("--no-sandbox")
            edge_options.add_argument("--disable-dev-shm-usage")

        edge_options.add_argument(f"--window-size={window_size}")
        edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        edge_options.add_experimental_option('useAutomationExtension', False)
        edge_options.add_argument("--disable-blink-features=AutomationControlled")

        # 使用传入的driver路径
        if not driver_path:
            raise ValueError("Driver路径不能为空")

        service = webdriver.edge.service.Service(driver_path)
        self.driver = webdriver.Edge(service=service, options=edge_options)

        # 防检测
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        self.wait = WebDriverWait(self.driver, 20)
        self.ocr = ddddocr.DdddOcr()
        self.headless = headless

    '''#################开始获取验证码###################'''
    def _get_captcha_via_javascript(self):
        try:
            captcha_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img[src*='code'], img[src*='captcha'], img[src*='verify']"))
            )
            js_script = """
            function imgToBase64(imgElement) {
                var canvas = document.createElement('canvas');
                var ctx = canvas.getContext('2d');
                canvas.width = imgElement.naturalWidth;
                canvas.height = imgElement.naturalHeight;
                ctx.drawImage(imgElement, 0, 0);
                return canvas.toDataURL('image/png');
            }
            return imgToBase64(arguments[0]);
            """
            base64_data = self.driver.execute_script(js_script, captcha_element)
            if base64_data and 'base64,' in base64_data:
                base64_data = base64_data.split('base64,')[1]
                image_data = base64.b64decode(base64_data)
                captcha_image = Image.open(io.BytesIO(image_data))
                # captcha_image.save("captcha_js.png")
                logger.info("使用JavaScript方法获取验证码成功")
                return captcha_image
        except Exception as e:
            logger.warning(f"JavaScript截图失败: {e}")
        return None

    def _get_captcha_via_cookie_session(self):
        try:
            captcha_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img[src*='code'], img[src*='captcha'], img[src*='verify']"))
            )
            captcha_src = captcha_element.get_attribute("src")
            if captcha_src and 'base64' in captcha_src:
                base64_data = captcha_src.split('base64,')[-1]
                image_data = base64.b64decode(base64_data)
                captcha_image = Image.open(io.BytesIO(image_data))
                # captcha_image.save("captcha_base64.png")
                logger.info("使用base64方法获取验证码成功")
                return captcha_image
            elif captcha_src:
                if captcha_src.startswith('/'):
                    current_url = self.driver.current_url
                    base_url = current_url.split('/')[0] + '//' + current_url.split('/')[2]
                    captcha_url = base_url + captcha_src
                else:
                    captcha_url = captcha_src
                cookies = self.driver.get_cookies()
                session = requests.Session()
                for cookie in cookies:
                    session.cookies.set(cookie['name'], cookie['value'])
                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Referer': self.driver.current_url,
                }
                response = session.get(captcha_url, headers=headers, timeout=10)
                response.raise_for_status()
                captcha_image = Image.open(io.BytesIO(response.content))
                # captcha_image.save("captcha_session.png")
                logger.info("使用会话方法获取验证码成功")
                return captcha_image
        except Exception as e:
            logger.warning(f"会话方法失败: {e}")
        return None

    def get_captcha_image(self):
        logger.info("开始获取验证码图片...")
        js_result = self._get_captcha_via_javascript()
        if js_result:
            return js_result
        session_result = self._get_captcha_via_cookie_session()
        if session_result:
            return session_result
        logger.error("所有验证码获取方法都失败了")
        return None

    def refresh_captcha(self):
        try:
            selectors = [
                "button[onclick*='refresh']", "button[onclick*='Code']",
                "img[onclick*='refresh']", "a[onclick*='refresh']",
                ".refresh-btn", ".captcha-refresh", "i.el-icon-refresh",
                "span[class*='refresh']", "*[class*='refresh']"
            ]
            for selector in selectors:
                try:
                    btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                    self.driver.execute_script("arguments[0].click();", btn)
                    logger.info("已刷新验证码")
                    time.sleep(2)
                    return True
                except:
                    continue
            try:
                captcha_img = self.driver.find_element(By.CSS_SELECTOR, "img[src*='code'], img[src*='captcha']")
                self.driver.execute_script("arguments[0].click();", captcha_img)
                logger.info("通过点击验证码图片刷新")
                time.sleep(2)
                return True
            except:
                pass
            return False
        except Exception as e:
            logger.warning(f"刷新验证码失败: {e}")
            return False

    def extract_text_from_image(self, image):
        try:
            if not image:
                return ""

            processed_image = self.preprocess_image(image)
            img_byte_arr = io.BytesIO()
            processed_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            text = self.ocr.classification(img_byte_arr)
            text = re.sub(r'\W+', '', text).strip()

            # 更新全局统计（SystemMonitor）
            self.monitor_ref.total += 1
            return text
        except Exception as e:
            logger.error(f"OCR识别时出错: {str(e)}")
            return ""

    def preprocess_image(self, image):
        try:
            image = image.convert('L')
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            threshold = 160
            image = image.point(lambda p: p > threshold and 255)
            image = image.filter(ImageFilter.MedianFilter(size=3))
            return image
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return image

    def detect_transient_popup(self, timeout=4.0, poll_interval=0.25):
        """
        专门检测带有 role='alert' 的弹窗
        """
        end = time.time() + timeout
        
        while time.time() < end:
            try:
                # 查找所有带有 role='alert' 的元素
                alert_elements = self.driver.find_elements(By.CSS_SELECTOR, "[role='alert']")
                
                for alert in alert_elements:
                    try:
                        # 检查元素是否可见
                        if alert.is_displayed():
                            # 获取弹窗的完整文本
                            text = alert.text.strip()
                            if text:
                                return text
                    except Exception as e:
                        logger.warning(f"检查 alert 元素时出错: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"查找 alert 元素时出错: {e}")
            
            time.sleep(poll_interval)
        
        return None


    def is_login_successful(self, success_indicators, failure_indicators):
        try:
            time.sleep(5)
            # 使用显式等待
            wait = WebDriverWait(self.driver, 30)
            
            # 先等待body标签出现（表示页面开始加载）
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            def page_loaded(driver):
                return driver.execute_script("return document.readyState") == "complete"
            
            wait.until(page_loaded)
            
            current_url = self.driver.current_url
            
            # 检查URL是否跳转（不再是登录页面）
            if "login" not in current_url.lower():
                logger.info(f"URL已跳转: {current_url}")
                return True
                
            logger.error("URL未跳转")
            return False
            
        except TimeoutException:
            logger.error("等待页面加载超时（30秒）")
            return False
        except Exception as e:
            logger.error(f"检查登录状态时出错: {e}")
            return False

    def input_captcha_and_submit(self, captcha_input_selector, login_button_selector, max_captcha_attempts=3):
        """输入验证码并提交；返回: True=成功, False=失败, 'captcha_error'=验证码错误, 'account_error'=账号密码错误"""
        captcha_attempt = 0

        while True:
            captcha_attempt += 1
            logger.info(f"验证码处理尝试第 {captcha_attempt} 次")

            # 获取验证码图片
            captcha_image = self.get_captcha_image()
            if not captcha_image:
                logger.error("无法获取验证码图片")
                if captcha_attempt >= max_captcha_attempts:
                    logger.error("已达到最大尝试次数，无法获取验证码图片")
                    return False
                continue

            # 识别验证码
            captcha_text = self.extract_text_from_image(captcha_image)
            logger.info(f"识别出的验证码: {captcha_text} (长度: {len(captcha_text)})")

            # 长度校验
            if len(captcha_text) != 4:
                logger.warning(f"验证码长度不正确: {len(captcha_text)}，将刷新并重试")
                if not self.refresh_captcha():
                    logger.error("刷新验证码失败")
                    return False
            else:
                break

        # 输入验证码
        try:
            captcha_input = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, captcha_input_selector)))
            captcha_input.clear()
            captcha_input.send_keys(captcha_text)
            logger.info("已输入验证码")
        except Exception as e:
            logger.error(f"输入验证码失败: {e}")
            return False

        # 点击登录按钮
        try:
            login_button = self.driver.find_element(By.CSS_SELECTOR, login_button_selector)
            self.driver.execute_script("arguments[0].click();", login_button)
            logger.info("已点击登录按钮")
        except Exception as e:
            logger.error(f"点击登录按钮失败: {e}")
            return False
        
        # 等待弹窗出现
        time.sleep(1.5)
        
        # 检测弹窗
        popup_text = self.detect_transient_popup(timeout=3.0)
        if popup_text:
            
            # 精确判断账号密码错误
            if "账号" in popup_text or "密码无效" in popup_text or "无法登录" in popup_text:
                logger.error(f"登录后弹窗提示: '{popup_text}'。判断为账号密码无效错误，无法登录。")
                return "account_error"
            # 判断验证码错误
            elif "验证码" in popup_text and ("无效" in popup_text or "错误" in popup_text):
                logger.warning(f"登录后弹窗提示: '{popup_text}'。判断为验证码无效错误。重新尝试......")
                return "captcha_error"
            else:
                logger.error(f"登录后弹窗提示: '{popup_text}'。判断为其他类型错误")
                return False

        return True

    def login(self, url, username, password, username_field_selector, password_field_selector, captcha_input_selector,
              login_button_selector, success_indicators, failure_indicators):
        max_load_attempts = 3
        for attempt in range(1, max_load_attempts + 1):
            try:
                logger.info(f"正在访问登录页: {url}  (尝试 {attempt}/{max_load_attempts})")
                self.driver.set_page_load_timeout(30)  # 30 秒超时
                self.driver.get(url)
                break  # ✅ 成功就跳出循环
            except (WebDriverException, TimeoutException) as e:
                logger.warning(f"第 {attempt} 次加载失败: {e}")
                if attempt == max_load_attempts:
                    # ✅ 第三次仍然失败，明确返回 page_load_error
                    logger.error("❌ 三次均无法加载页面，视为系统异常")
                    return "page_load_error"
                else:
                    logger.info("20 秒后重试...")
                    time.sleep(20)
        else:
            # 理论上不会到这里，但加兜底更安全
            logger.error("❌ 所有尝试均失败（未知原因），视为系统异常")
            return "page_load_error"

        try:
            # ==== 页面加载成功，开始输入用户名密码 ====
            time.sleep(3)
            username_field = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, username_field_selector)))
            username_field.clear()
            username_field.send_keys(username)

            password_field = self.driver.find_element(By.CSS_SELECTOR, password_field_selector)
            password_field.clear()
            password_field.send_keys(password)

            while True:
                captcha_result = self.input_captcha_and_submit(captcha_input_selector, login_button_selector)

                if captcha_result == "captcha_error":
                    time.sleep(1.5)
                elif captcha_result == "account_error":
                    return "account_error"  # 返回账号密码错误标识
                elif captcha_result is False:
                    return False
                else:
                    break

            # ==== 检查是否真的登录成功 ====
            result = self.is_login_successful(success_indicators, failure_indicators)
            if result is True:
                self.monitor_ref.correct += 1
                return True
            else:
                return False

        except Exception as e:
            # ✅ 兜底：任何异常都视为系统异常
            logger.error(f"登录过程中发生未预期异常: {e}")
            return "page_load_error"

    def close(self):
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()  # quit()会彻底终止WebDriver进程
                logger.info("WebDriver已正常关闭")
        except Exception as e:
            logger.warning(f"关闭WebDriver时出错: {e}")


if __name__ == "__main__":
    # 创建监控实例并启动
    monitor = SystemMonitor()