from datetime import datetime
import sys
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QTextEdit, QPushButton, QHBoxLayout, QStatusBar,
                             QMessageBox, QLabel, QLineEdit, QFormLayout, QGroupBox,
                             QSpinBox, QSplitter, QInputDialog, QStackedWidget)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QTextCursor, QFont
import time

# 导入你的原有代码
from config_manager import load_config, save_config, cleanup_old_logs
from remain_login import SystemMonitor, setup_logging


class LockScreen(QWidget):
    """锁定界面"""
    unlock_signal = pyqtSignal(str)  # 发送密码信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # 标题
        title_label = QLabel("系统已锁定")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")

        # 密码输入框
        password_layout = QHBoxLayout()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setPlaceholderText("请输入管理员密码")
        self.password_edit.setFixedWidth(200)
        self.password_edit.returnPressed.connect(self.on_unlock)  # 回车触发解锁

        # 解锁按钮
        unlock_btn = QPushButton("解锁")
        unlock_btn.clicked.connect(self.on_unlock)
        unlock_btn.setFixedWidth(80)

        password_layout.addWidget(self.password_edit)
        password_layout.addWidget(unlock_btn)
        password_layout.setAlignment(Qt.AlignCenter)

        # 提示信息
        hint_label = QLabel("请输入管理员密码解锁系统")
        hint_label.setAlignment(Qt.AlignCenter)
        hint_label.setStyleSheet("color: #666; margin-top: 10px;")

        layout.addWidget(title_label)
        layout.addLayout(password_layout)
        layout.addWidget(hint_label)
        layout.addStretch()

        self.setLayout(layout)

    def on_unlock(self):
        """解锁按钮点击事件"""
        password = self.password_edit.text().strip()
        if password:
            self.unlock_signal.emit(password)
            self.password_edit.clear()
        else:
            # 密码为空时显示提示
            QMessageBox.warning(self, "输入错误", "请输入管理员密码")


    def clear_password(self):
        """清空密码框"""
        self.password_edit.clear()


class ConfigWidget(QWidget):
    """配置界面"""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 管理员密码组（可在程序运行时修改，内存保存）
        admin_group = QGroupBox("管理员设置")
        admin_layout = QFormLayout()
        self.admin_password_edit = QLineEdit(self.config.get('admin', {}).get('password', ''))
        self.admin_password_edit.setEchoMode(QLineEdit.Password)
        admin_layout.addRow("管理员密码:", self.admin_password_edit)
        admin_group.setLayout(admin_layout)

        # 告警配置组
        alert_group = QGroupBox("告警配置")
        alert_layout = QFormLayout()

        self.check_interval_spin = QSpinBox()
        self.check_interval_spin.setRange(5, 3600)
        self.check_interval_spin.setValue(self.config['alert']['check_interval'])
        self.check_interval_spin.setSuffix(" 秒")

        self.alert_threshold_spin = QSpinBox()
        self.alert_threshold_spin.setRange(1, 10)
        self.alert_threshold_spin.setValue(self.config['alert']['alert_threshold'])
        self.alert_threshold_spin.setSuffix(" 次")

        self.recipients_edit = QLineEdit()
        self.recipients_edit.setText(", ".join(self.config['alert']['recipients']))
        self.recipients_edit.setPlaceholderText("多个邮箱用逗号分隔")

        alert_layout.addRow("检查间隔:", self.check_interval_spin)
        alert_layout.addRow("告警阈值:", self.alert_threshold_spin)
        alert_layout.addRow("收件邮箱:", self.recipients_edit)
        alert_group.setLayout(alert_layout)

        # SMTP配置组
        smtp_group = QGroupBox("SMTP配置")
        smtp_layout = QFormLayout()

        self.smtp_server_edit = QLineEdit(self.config['smtp']['server'])
        self.smtp_port_spin = QSpinBox()
        self.smtp_port_spin.setRange(1, 65535)
        self.smtp_port_spin.setValue(self.config['smtp']['port'])
        self.smtp_username_edit = QLineEdit(self.config['smtp']['username'])
        self.smtp_password_edit = QLineEdit(self.config['smtp']['password'])
        self.smtp_password_edit.setEchoMode(QLineEdit.Password)

        smtp_layout.addRow("SMTP服务器:", self.smtp_server_edit)
        smtp_layout.addRow("端口:", self.smtp_port_spin)
        smtp_layout.addRow("发件箱:", self.smtp_username_edit)
        smtp_layout.addRow("SMTP授权码:", self.smtp_password_edit)
        smtp_group.setLayout(smtp_layout)

        # 登录配置组
        login_group = QGroupBox("登录配置")
        login_layout = QFormLayout()

        self.login_url_edit = QLineEdit(self.config['login']['url'])
        self.login_username_edit = QLineEdit(self.config['login']['username'])
        self.login_password_edit = QLineEdit(self.config['login']['password'])
        self.login_password_edit.setEchoMode(QLineEdit.Password)

        login_layout.addRow("登录URL:", self.login_url_edit)
        login_layout.addRow("用户名:", self.login_username_edit)
        login_layout.addRow("密码:", self.login_password_edit)
        login_group.setLayout(login_layout)

        # 维护时间配置组
        maintenance_group = QGroupBox("系统维护时间设置")
        maintenance_layout = QFormLayout()

        # 清空按钮布局
        clear_button_layout = QHBoxLayout()
        self.clear_maintenance_btn = QPushButton("一键清空维护时间")
        self.clear_maintenance_btn.clicked.connect(self.clear_maintenance_time)
        self.clear_maintenance_btn.setToolTip("清空所有维护时间字段")
        clear_button_layout.addStretch()
        clear_button_layout.addWidget(self.clear_maintenance_btn)

        # 开始时间 - 年月日分开
        start_date_layout = QHBoxLayout()

        # 开始年份
        self.start_year_edit = QLineEdit(self.config['maintenance']['start_year'])
        self.start_year_edit.setPlaceholderText("年份")
        self.start_year_edit.setFixedWidth(60)
        self.start_year_edit.textChanged.connect(self.validate_maintenance_time)

        # 开始月份
        self.start_month_edit = QLineEdit(self.config['maintenance']['start_month'])
        self.start_month_edit.setPlaceholderText("月")
        self.start_month_edit.setFixedWidth(40)
        self.start_month_edit.textChanged.connect(self.validate_maintenance_time)

        # 开始日期
        self.start_day_edit = QLineEdit(self.config['maintenance']['start_day'])
        self.start_day_edit.setPlaceholderText("日")
        self.start_day_edit.setFixedWidth(40)
        self.start_day_edit.textChanged.connect(self.validate_maintenance_time)

        # 开始时间
        # 开始小时
        self.start_hour_edit = QLineEdit(self.config['maintenance']['start_hour'])
        self.start_hour_edit.setPlaceholderText("时")
        self.start_hour_edit.setFixedWidth(30)
        self.start_hour_edit.textChanged.connect(self.validate_maintenance_time)

        # 开始分钟
        self.start_minute_edit = QLineEdit(self.config['maintenance']['start_minute'])
        self.start_minute_edit.setPlaceholderText("分")
        self.start_minute_edit.setFixedWidth(30)
        self.start_minute_edit.textChanged.connect(self.validate_maintenance_time)

        # 开始秒钟
        self.start_second_edit = QLineEdit(self.config['maintenance']['start_second'])
        self.start_second_edit.setPlaceholderText("秒")
        self.start_second_edit.setFixedWidth(30)
        self.start_second_edit.textChanged.connect(self.validate_maintenance_time)

        start_date_layout.addWidget(self.start_year_edit)
        start_date_layout.addWidget(QLabel("年"))
        start_date_layout.addWidget(self.start_month_edit)
        start_date_layout.addWidget(QLabel("月"))
        start_date_layout.addWidget(self.start_day_edit)
        start_date_layout.addWidget(QLabel("日"))
        start_date_layout.addWidget(self.start_hour_edit)
        start_date_layout.addWidget(QLabel("时"))
        start_date_layout.addWidget(self.start_minute_edit)
        start_date_layout.addWidget(QLabel("分"))
        start_date_layout.addWidget(self.start_second_edit)
        start_date_layout.addWidget(QLabel("秒"))
        start_date_layout.addStretch()

        # 结束时间 - 年月日分开
        end_date_layout = QHBoxLayout()

        # 结束年份
        self.end_year_edit = QLineEdit(self.config['maintenance']['end_year'])
        self.end_year_edit.setPlaceholderText("年份")
        self.end_year_edit.setFixedWidth(60)
        self.end_year_edit.textChanged.connect(self.validate_maintenance_time)

        # 结束月份
        self.end_month_edit = QLineEdit(self.config['maintenance']['end_month'])
        self.end_month_edit.setPlaceholderText("月")
        self.end_month_edit.setFixedWidth(40)
        self.end_month_edit.textChanged.connect(self.validate_maintenance_time)

        # 结束日期
        self.end_day_edit = QLineEdit(self.config['maintenance']['end_day'])
        self.end_day_edit.setPlaceholderText("日")
        self.end_day_edit.setFixedWidth(40)
        self.end_day_edit.textChanged.connect(self.validate_maintenance_time)

        # 结束小时
        self.end_hour_edit = QLineEdit(self.config['maintenance']['end_hour'])
        self.end_hour_edit.setPlaceholderText("时")
        self.end_hour_edit.setFixedWidth(30)
        self.end_hour_edit.textChanged.connect(self.validate_maintenance_time)

        # 结束分钟
        self.end_minute_edit = QLineEdit(self.config['maintenance']['end_minute'])
        self.end_minute_edit.setPlaceholderText("分")
        self.end_minute_edit.setFixedWidth(30)
        self.end_minute_edit.textChanged.connect(self.validate_maintenance_time)

        # 结束秒钟
        self.end_second_edit = QLineEdit(self.config['maintenance']['end_second'])
        self.end_second_edit.setPlaceholderText("秒")
        self.end_second_edit.setFixedWidth(30)
        self.end_second_edit.textChanged.connect(self.validate_maintenance_time)

        end_date_layout.addWidget(self.end_year_edit)
        end_date_layout.addWidget(QLabel("年"))
        end_date_layout.addWidget(self.end_month_edit)
        end_date_layout.addWidget(QLabel("月"))
        end_date_layout.addWidget(self.end_day_edit)
        end_date_layout.addWidget(QLabel("日"))
        end_date_layout.addWidget(self.end_hour_edit)
        end_date_layout.addWidget(QLabel("时"))
        end_date_layout.addWidget(self.end_minute_edit)
        end_date_layout.addWidget(QLabel("分"))
        end_date_layout.addWidget(self.end_second_edit)
        end_date_layout.addWidget(QLabel("秒"))
        end_date_layout.addStretch()

        # 维护原因
        self.maintenance_reason_edit = QLineEdit(self.config['maintenance']['reason'])
        self.maintenance_reason_edit.setPlaceholderText("请输入维护原因")

        # 时间验证提示标签
        self.time_validation_label = QLabel("")
        self.time_validation_label.setStyleSheet("color: red; font-size: 10px;")
        self.time_validation_label.setWordWrap(True)

        # 格式提示
        format_label = QLabel("格式提示: 年份(4位) 月份(1-12) 日期(1-31) 时(0-23) 分(0-59) 秒(0-59)")
        format_label.setStyleSheet("color: #666; font-size: 15px;")

        maintenance_layout.addRow("", clear_button_layout)  # 单独一行放置清空按钮
        maintenance_layout.addRow("开始时间:", start_date_layout)
        maintenance_layout.addRow("结束时间:", end_date_layout)
        maintenance_layout.addRow("维护原因:", self.maintenance_reason_edit)
        maintenance_layout.addRow("", self.time_validation_label)  # 添加验证提示
        maintenance_layout.addRow("", format_label)
        maintenance_group.setLayout(maintenance_layout)

        # 日志配置组（新增）
        logging_group = QGroupBox("日志配置")
        logging_layout = QFormLayout()
        
        self.log_max_days_spin = QSpinBox()
        self.log_max_days_spin.setRange(1, 365)
        self.log_max_days_spin.setValue(self.config.get('logging', {}).get('max_days', 30))
        self.log_max_days_spin.setSuffix(" 天")
        
        logging_layout.addRow("日志保留天数:", self.log_max_days_spin)
        logging_group.setLayout(logging_layout)

        layout.addWidget(admin_group)
        layout.addWidget(alert_group)
        layout.addWidget(smtp_group)
        layout.addWidget(login_group)
        layout.addWidget(maintenance_group)
        # 将日志配置组添加到布局中（可以放在合适的位置）
        layout.addWidget(logging_group)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()  # 把按钮推到右边
        self.save_btn = QPushButton("保存配置")
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        layout.addStretch()

        self.setLayout(layout)

    def clear_maintenance_time(self):
        """清空所有维护时间字段"""
        reply = QMessageBox.question(self, "确认清空", 
                                   "确定要清空所有维护时间设置吗？\n此操作不可撤销。",
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 清空开始时间字段
            self.start_year_edit.clear()
            self.start_month_edit.clear()
            self.start_day_edit.clear()
            self.start_hour_edit.clear()
            self.start_minute_edit.clear()
            self.start_second_edit.clear()
            
            # 清空结束时间字段
            self.end_year_edit.clear()
            self.end_month_edit.clear()
            self.end_day_edit.clear()
            self.end_hour_edit.clear()
            self.end_minute_edit.clear()
            self.end_second_edit.clear()
            
            # 清空维护原因
            self.maintenance_reason_edit.clear()
            
            # 更新验证提示
            self.time_validation_label.setText("✅ 维护时间已清空")
            self.time_validation_label.setStyleSheet("color: green; font-size: 12px;")
            
            # 记录日志（如果需要）
            if hasattr(self.parent(), 'log_signal'):
                self.parent().log_signal.emit("维护时间设置已清空\n")

    def validate_maintenance_time(self):
        """验证维护时间是否有效"""
        # 获取所有时间字段
        start_year = self.start_year_edit.text().strip()
        start_month = self.start_month_edit.text().strip()
        start_day = self.start_day_edit.text().strip()
        start_hour = self.start_hour_edit.text().strip()
        start_minute = self.start_minute_edit.text().strip()
        start_second = self.start_second_edit.text().strip()

        end_year = self.end_year_edit.text().strip()
        end_month = self.end_month_edit.text().strip()
        end_day = self.end_day_edit.text().strip()
        end_hour = self.end_hour_edit.text().strip()
        end_minute = self.end_minute_edit.text().strip()
        end_second = self.end_second_edit.text().strip()

        # 如果任何字段为空，清除提示
        if not all([start_year, start_month, start_day, start_hour, start_minute, start_second,
                    end_year, end_month, end_day, end_hour, end_minute, end_second]):
            self.time_validation_label.setText("")
            self.time_validation_label.setStyleSheet("color: red; font-size: 10px;")
            return True

        try:
            # 确保时间格式正确（补零）
            start_month = start_month.zfill(2)
            start_day = start_day.zfill(2)
            start_hour = start_hour.zfill(2)
            start_minute = start_minute.zfill(2)
            start_second = start_second.zfill(2)

            end_month = end_month.zfill(2)
            end_day = end_day.zfill(2)
            end_hour = end_hour.zfill(2)
            end_minute = end_minute.zfill(2)
            end_second = end_second.zfill(2)

            # 格式化日期字符串
            start_date_str = f"{start_year}-{start_month}-{start_day}"
            start_str = f"{start_date_str} {start_hour}:{start_minute}:{start_second}"

            end_date_str = f"{end_year}-{end_month}-{end_day}"
            end_str = f"{end_date_str} {end_hour}:{end_minute}:{end_second}"

            # 解析开始时间
            start_time_obj = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')

            # 解析结束时间
            end_time_obj = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')

            # 检查结束时间是否在开始时间之前
            if end_time_obj <= start_time_obj:
                self.time_validation_label.setText("❌ 结束时间不能在开始时间之前")
                self.time_validation_label.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
                return False
            else:
                self.time_validation_label.setText("✅ 时间设置有效")
                self.time_validation_label.setStyleSheet("color: green; font-size: 12px;")
                return True

        except ValueError as e:
            error_msg = f"⚠️ 时间格式不正确: {str(e)}"
            self.time_validation_label.setText(error_msg)
            self.time_validation_label.setStyleSheet("color: orange; font-size: 12px;")
            return False
        except Exception as e:
            error_msg = f"⚠️ 时间验证错误: {str(e)}"
            self.time_validation_label.setText(error_msg)
            self.time_validation_label.setStyleSheet("color: orange; font-size: 12px;")
            return False

    def get_config(self):
        """获取配置"""
        config = self.config.copy()

        # 管理员密码
        if 'admin' not in config:
            config['admin'] = {}
        config['admin']['password'] = self.admin_password_edit.text().strip()

        # 更新告警配置
        recipients_text = self.recipients_edit.text().strip()
        if recipients_text:
            recipients = [email.strip() for email in recipients_text.split(',') if email.strip()]
            config['alert']['recipients'] = recipients
        else:
            config['alert']['recipients'] = DEFAULT_CONFIG['alert']['recipients']

        config['alert']['check_interval'] = self.check_interval_spin.value()
        config['alert']['alert_threshold'] = self.alert_threshold_spin.value()
        config['alert']['from_email'] = self.smtp_username_edit.text().strip()

        # 更新SMTP配置
        config['smtp']['server'] = self.smtp_server_edit.text().strip()
        config['smtp']['port'] = self.smtp_port_spin.value()
        config['smtp']['username'] = self.smtp_username_edit.text().strip()
        config['smtp']['password'] = self.smtp_password_edit.text().strip()

        # 更新登录配置
        config['login']['url'] = self.login_url_edit.text().strip()
        config['login']['username'] = self.login_username_edit.text().strip()
        config['login']['password'] = self.login_password_edit.text().strip()

        # 更新维护配置
        config['maintenance']['start_year'] = self.start_year_edit.text().strip()
        config['maintenance']['start_month'] = self.start_month_edit.text().strip()
        config['maintenance']['start_day'] = self.start_day_edit.text().strip()
        config['maintenance']['start_hour'] = self.start_hour_edit.text().strip()
        config['maintenance']['start_minute'] = self.start_minute_edit.text().strip()
        config['maintenance']['start_second'] = self.start_second_edit.text().strip()
        config['maintenance']['end_year'] = self.end_year_edit.text().strip()
        config['maintenance']['end_month'] = self.end_month_edit.text().strip()
        config['maintenance']['end_day'] = self.end_day_edit.text().strip()
        config['maintenance']['end_hour'] = self.end_hour_edit.text().strip()
        config['maintenance']['end_minute'] = self.end_minute_edit.text().strip()
        config['maintenance']['end_second'] = self.end_second_edit.text().strip()
        config['maintenance']['reason'] = self.maintenance_reason_edit.text().strip()

        # 更新日志配置
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['max_days'] = self.log_max_days_spin.value()
        config['logging']['level'] = self.config.get('logging', {}).get('level', 'INFO')

        return config


class MonitorThread(QThread):
    """监控线程，避免阻塞UI"""
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    account_error_signal = pyqtSignal(str)
    network_error_signal = pyqtSignal(str)  # 网络错误信号

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.monitor = None
        self.running = True
        self.schedule = None
        self.force_resume = False
        self.current_check_interval = config['alert']['check_interval']
        self.network_checked = False    # 🆕 网络检查标志
        self.driver_prepared = False    # 🆕 Driver准备标志

    def run(self):
        """主运行循环"""
        last_log_cleanup = time.time()
        last_log_date = datetime.now().date()  # 记录当前日期
        log_cleanup_interval = 24 * 3600  # 24小时清理一次

        while self.running:
            try:
                current_date = datetime.now().date()
                
                # 检查是否是新的一天
                if current_date != last_log_date:
                    logger.info(f"检测到日期变更: {last_log_date} -> {current_date}，重新初始化日志系统")
                    self.reinitialize_logging()
                    last_log_date = current_date
                
                # 每天执行一次日志清理
                current_time = time.time()
                if current_time - last_log_cleanup > log_cleanup_interval:
                    cleanup_old_logs()
                    last_log_cleanup = current_time
                    self.log_signal.emit("✅ 已执行每日日志清理检查\n")

                # 第一步：网络检测
                if not self.network_checked:
                    network_ok = self.perform_network_check()
                    if not network_ok:
                        # 网络失败，等待间隔时间后重试
                        self.wait_for_retry()
                        continue  # 重新开始循环
                    self.network_checked = True
                
                # 第二步：环境准备（浏览器 + Driver）
                if not self.driver_prepared:
                    environment_ok = self.prepare_environment()
                    if not environment_ok:
                        self.wait_for_retry()
                        continue
                    self.driver_prepared = True
                
                # 第三步：开始监控循环
                self.start_monitoring_loop()
                
            except Exception as e:
                self.log_signal.emit(f"监控主循环错误: {str(e)}\n")
                self.network_checked = False
                self.driver_prepared = False
                self.wait_for_retry()

    def reinitialize_logging(self):
        """重新初始化日志系统（用于日期变更时）"""
        global logger
        
        # 移除现有的文件处理器
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()
        
        # 重新设置日志系统
        logger = setup_logging()
        self.log_signal.emit("日志系统已重新初始化，开始新一天的日志记录\n")

    def perform_network_check(self):
        """执行网络检测"""
        self.log_signal.emit("🌐 开始网络连接检测...\n")
        self.status_signal.emit("检测网络连接")
        
        # 创建监控实例（如果还没有创建）
        if not self.monitor:
            self.monitor = SystemMonitor(self.config)
        
        # 直接使用SystemMonitor的网络检查方法
        return self.monitor.check_network_connection()
        
    def prepare_environment(self):
        """准备完整环境（浏览器 + Driver）"""
        self.log_signal.emit("🔧 开始准备完整运行环境...\n")
        self.status_signal.emit("准备运行环境")
        
        if not self.monitor:
            self.monitor = SystemMonitor(self.config)
        
        # 使用新的环境准备方法
        if self.monitor.prepare_environment():
            self.log_signal.emit("✅ 完整运行环境准备就绪\n")
            return True
        else:
            self.log_signal.emit("❌ 运行环境准备失败\n")
            return False

    def start_monitoring_loop(self):
        """开始监控循环"""
        self.log_signal.emit("🚀 开始监控循环...\n")
        self.status_signal.emit("监控运行中")
        
        import schedule

        # 清除所有现有任务
        schedule.clear()
        self.schedule = schedule

        # 添加新任务
        check_interval = self.config['alert']['check_interval']
        schedule.every(check_interval).seconds.do(self.execute_monitor_check)

        # 立即运行一次检查
        self.execute_monitor_check()

        # 监控循环
        while self.running:
            try:
                schedule.run_pending()
                # 使用更短的睡眠时间，以便更快响应停止请求
                for i in range(10):
                    if not self.running:
                        break
                    time.sleep(0.1)
            except Exception as e:
                self.log_signal.emit(f"监控循环错误: {str(e)}\n")
                self.network_checked = False
                self.driver_prepared = False
                break  # 跳出监控循环，等待重试

    def execute_monitor_check(self):
        """执行监控检查"""
        try:
            if self.monitor:
                result = self.monitor.check_system_status()
                
                # 处理不同的返回结果
                if result == "account_error":
                    # 账号密码错误，停止监控
                    self.account_error_signal.emit("账号或密码无效，无法登录")
                    self.running = False
                    
        except Exception as e:
            self.log_signal.emit(f"执行监控检查时出错: {str(e)}\n")
            self.network_checked = False
            self.driver_prepared = False

    def wait_for_retry(self):
        """等待重试"""
        if not self.running:
            return
        
        check_interval = self.config['alert']['check_interval']
        
        self.log_signal.emit(f"⏳ 等待 {check_interval} 秒后重新检测...\n")
        self.status_signal.emit(f"等待重试 ({check_interval}秒)")

        # 🆕 重试时重置标志，下次会重新检查网络和Driver
        self.network_checked = False
        self.driver_prepared = False
        
        # 清理资源
        if self.monitor:
            if self.monitor.current_automation:
                try:
                    self.monitor.current_automation.close()
                except Exception as e:
                    self.log_signal.emit(f"关闭driver出错: {e}\n")
                self.monitor.current_automation = None
            self.monitor.cleanup_driver()
        
        if self.schedule:
            self.schedule.clear()
        
        # 等待重试
        total_wait_seconds = check_interval
        wait_steps = total_wait_seconds * 2  # 每0.5秒检查一次
        
        for i in range(wait_steps):
            if not self.running:
                break
            time.sleep(0.5)
            
            # 更新状态显示剩余时间
            if i % 10 == 0:  # 每5秒更新一次
                remaining = total_wait_seconds - (i * 0.5)
                self.status_signal.emit(f"等待重试 (剩余{int(remaining)}秒)")
        

    def stop(self):
        """停止监控"""
        self.running = False
        self.force_resume = False

        self.network_checked = False
        self.driver_prepared = False


        if self.monitor:
            # 清理资源
            if self.monitor.current_automation:
                try:
                    self.monitor.current_automation.close()
                except Exception as e:
                    self.log_signal.emit(f"关闭driver出错: {e}\n")
                self.monitor.current_automation = None
            self.monitor.cleanup_driver()

        if self.schedule:
            self.schedule.clear()

        self.log_signal.emit("监控线程停止...\n")
        self.status_signal.emit("监控已停止")

    def update_config(self, new_config):
        """更新配置"""
        self.config = new_config
        self.current_check_interval = new_config['alert']['check_interval']
        
        if self.monitor:
            self.monitor.config = new_config

    def force_resume_monitoring(self):
        """强制恢复监控"""
        self.force_resume = True
        if self.monitor:
            self.monitor.set_force_resume(True)
        self.log_signal.emit("强制恢复监控，跳过维护检查...\n")
        self.status_signal.emit("监控已强制恢复")


class UIHandler(logging.Handler):
    """自定义日志处理器，将日志发送到UI"""

    def __init__(self, log_signal):
        super().__init__()
        self.log_signal = log_signal

    def emit(self, record):
        log_entry = self.format(record)
        self.log_signal.emit(log_entry + '\n')


class MainWindow(QMainWindow):
    """主窗口"""
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.monitor_thread = None
        self.ui_handler = None
        self.config = load_config()
        self.setup_logging()  # 先设置日志系统
        self.is_locked = False
        self.init_ui()  # 然后初始化UI
        self.auto_scroll = True  # 是否自动滚动到底部

    def save_config(self):
        """保存配置到程序内存"""
        # 先验证维护时间
        if not self.config_widget.validate_maintenance_time():
            QMessageBox.warning(self, "配置错误", "维护时间设置无效！\n结束时间不能在开始时间之前。")
            return

        new_config = self.config_widget.get_config()
        self.config = new_config

        # 保存到文件
        if save_config(new_config):
            # 如果监控正在运行，实时更新配置
            if self.monitor_thread and self.monitor_thread.isRunning():
                self.monitor_thread.update_config(new_config)
                self.log_signal.emit("配置已保存到文件并实时应用到运行中的监控\n")
            else:
                self.log_signal.emit("配置已保存到文件\n")

            QMessageBox.information(self, "成功", "配置已保存到文件" +
                                    ("并实时应用" if self.monitor_thread and self.monitor_thread.isRunning() else ""))
        else:
            QMessageBox.critical(self, "错误", "保存配置文件失败")
            self.log_signal.emit("保存配置文件失败\n")

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("系统登录监控工具")
        self.setGeometry(100, 100, 1000, 700)

        # 使用堆叠窗口来切换锁定/解锁状态
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.main_widget = QWidget()
        # 主布局
        main_layout = QHBoxLayout(self.main_widget)
        # 分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧配置面板
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        self.config_widget = ConfigWidget(self.config)
        config_layout.addWidget(self.config_widget)
        self.config_panel = config_widget
        # 保存配置按钮
        self.config_widget.save_btn.clicked.connect(self.save_config)

        # 右侧日志面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 控制按钮区域
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)

        self.start_btn = QPushButton("开始监控")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.stop_btn = QPushButton("停止监控")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.resume_btn = QPushButton("恢复监控")  # 新增恢复监控按钮
        self.resume_btn.clicked.connect(self.resume_monitoring)
        self.resume_btn.setEnabled(False)  # 初始状态禁用
        self.resume_btn.setToolTip("在维护期间强制恢复监控")
        self.network_test_btn = QPushButton("测试网络")
        self.network_test_btn.clicked.connect(self.test_network_connection)
        self.lock_btn = QPushButton("锁定系统")
        self.lock_btn.clicked.connect(self.lock_ui)

        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.resume_btn)  # 添加恢复按钮
        control_layout.addWidget(self.network_test_btn)
        control_layout.addWidget(self.lock_btn)
        self.control_panel = control_widget
        control_layout.addStretch()

        # 日志显示区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))

        right_layout.addWidget(control_widget)
        right_layout.addWidget(self.log_text)

        splitter.addWidget(config_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 700])
        main_layout.addWidget(splitter)

        # 获取垂直滚动条
        scroll_bar = self.log_text.verticalScrollBar()

        # 监听滚动条动作
        scroll_bar.actionTriggered.connect(self.on_scroll_action)

        # 创建锁定界面
        self.lock_screen = LockScreen()
        self.lock_screen.unlock_signal.connect(self.unlock_ui)

        # 添加到堆叠窗口
        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.addWidget(self.lock_screen)

        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪")

        # 连接日志信号
        self.log_signal.connect(self.append_log)

    def setup_logging(self):
        """设置日志，保留原有功能并添加UI输出"""
        # 调用原有的setup_logging
        logger = setup_logging()

        # 添加UI处理器
        self.ui_handler = UIHandler(self.log_signal)
        self.ui_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(self.ui_handler)

        # 设置日志级别
        logger.setLevel(logging.INFO)

    def append_log(self, message):
        scroll_bar = self.log_text.verticalScrollBar()
        was_at_bottom = scroll_bar.value() >= scroll_bar.maximum() - 10

        # 保存当前光标位置和滚动条位置
        cursor = self.log_text.textCursor()
        original_position = cursor.position()
        original_scroll = scroll_bar.value()

        # 插入文本
        self.log_text.moveCursor(QTextCursor.End)
        self.log_text.insertPlainText(message)

        # 如果用户没有在最底部，恢复原始位置和滚动条
        if not was_at_bottom:
            cursor.setPosition(original_position)
            self.log_text.setTextCursor(cursor)
            scroll_bar.setValue(original_scroll)

        else:
            scroll_bar.setValue(scroll_bar.maximum())

    def on_scroll_action(self, action):
        """
        0: QAbstractSlider.SliderNoAction
        1: QAbstractSlider.SliderSingleStepAdd
        2: QAbstractSlider.SliderSingleStepSub
        3: QAbstractSlider.SliderPageStepAdd
        4: QAbstractSlider.SliderPageStepSub
        5: QAbstractSlider.SliderToMinimum
        6: QAbstractSlider.SliderToMaximum
        7: QAbstractSlider.SliderMove
        """
        scroll_bar = self.log_text.verticalScrollBar()
        max_value = scroll_bar.maximum()
        current_value = scroll_bar.value()

        # 如果用户手动滚动（非到底部），关闭自动滚动
        if action in [2, 4, 7]:  # 向上滚动或拖拽
            if current_value < max_value - 10:  # 留一点缓冲
                self.auto_scroll = False
        elif current_value >= max_value - 10:
            self.auto_scroll = True

    # --- 锁定/解锁逻辑 ---
    def lock_ui(self):
        """锁定UI界面"""
        if self.is_locked:
            return

        # 切换到锁定界面
        self.stacked_widget.setCurrentIndex(1)
        self.is_locked = True
        self.log_signal.emit("系统已锁定\n")

    def unlock_ui(self, password):
        """解锁UI界面"""
        if not self.is_locked:
            return

        # 验证密码
        admin_password = self.config.get('admin', {}).get('password', '')
        if password == admin_password:
            # 切换到主界面
            self.stacked_widget.setCurrentIndex(0)
            self.is_locked = False
            self.log_signal.emit("系统已解锁\n")
            self.lock_screen.clear_password()
        else:
            QMessageBox.warning(self, "错误", "密码错误，请重新输入")
            self.lock_screen.clear_password()

    def test_network_connection(self):
        """手动测试网络连接"""
        self.log_signal.emit("🧪 手动测试网络连接...\n")
        
        temp_monitor = SystemMonitor(self.config)
        if temp_monitor.check_network_connection():
            QMessageBox.information(self, "网络测试", "✅ 网络连接正常")
            self.log_signal.emit("✅ 手动网络测试：连接正常\n")
        else:
            QMessageBox.warning(self, "网络测试", "❌ 网络连接失败，请检查网络设置")
            self.log_signal.emit("❌ 手动网络测试：连接失败\n")

    def start_monitoring(self):
        """开始监控"""
        # 先验证维护时间
        if not self.config_widget.validate_maintenance_time():
            QMessageBox.warning(self, "配置错误", "无法启动监控！\n维护时间设置无效，结束时间不能在开始时间之前。")
            return

        # 如果已有线程在运行，先停止它
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.stop_monitoring()
            # 等待一下确保线程完全停止
            import time
            time.sleep(1)

        # 获取当前配置
        current_config = self.config_widget.get_config()
        self.config = current_config

        check_interval = current_config['alert']['check_interval']
        self.log_signal.emit(f"🔔 启动监控流程\n")
        self.log_signal.emit(f"⏰ 监控间隔: {check_interval}秒\n")
        self.log_signal.emit("📋 流程: 网络检测 → Driver准备 → 监控循环\n")

        # 创建新的线程实例
        self.monitor_thread = MonitorThread(current_config)
        self.monitor_thread.log_signal.connect(self.append_log)
        self.monitor_thread.status_signal.connect(self.statusBar.showMessage)
        self.monitor_thread.account_error_signal.connect(self.handle_account_error)
        self.monitor_thread.network_error_signal.connect(self.handle_network_error)
        self.monitor_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.resume_btn.setEnabled(True)
        self.network_test_btn.setEnabled(False)

    def handle_network_error(self, error_message):
        """处理网络错误信号"""
        # 只在第一次网络错误时弹窗
        QMessageBox.warning(self, "网络连接问题", 
                            f"{error_message}\n\n系统将在设定的间隔时间后自动重试。")
        self._network_alert_shown = True

        self.statusBar.showMessage("网络失败，等待重试")

    
    def handle_account_error(self, error_message):
        """处理账号密码错误"""
        # 停止监控
        self.stop_monitoring()
        
        # 弹出警告窗口
        QMessageBox.critical(self, "账号密码错误", 
                           f"{error_message}\n\n请检查配置中的用户名和密码是否正确，然后重新启动监控。")
        
        # 在日志中记录
        self.log_signal.emit(f"❌ {error_message}，监控已停止\n")
        self.statusBar.showMessage("监控停止：账号密码错误")

    def stop_monitoring(self):
        """停止监控"""
        if self.monitor_thread:
            # 先停止线程
            self.monitor_thread.stop()

            # 等待线程结束
            if self.monitor_thread.isRunning():
                self.monitor_thread.wait(2000)  # 等待2秒

                # 如果线程还在运行，强制终止
                if self.monitor_thread.isRunning():
                    self.monitor_thread.terminate()
                    self.monitor_thread.wait()
                    self.log_signal.emit("监控线程已强制终止\n")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)  # 停止后禁用恢复按钮
        self.network_test_btn.setEnabled(True)
        
        self.statusBar.showMessage("监控已停止")

    def resume_monitoring(self):
        """恢复监控（在维护期间强制恢复）"""
        if self.monitor_thread and self.monitor_thread.isRunning():
            self.monitor_thread.force_resume_monitoring()
            self.log_signal.emit("用户手动恢复监控，跳过维护检查\n")
            self.resume_btn.setEnabled(False)  # 恢复后禁用按钮，避免重复点击
        else:
            self.log_signal.emit("监控未运行，无法恢复\n")

    def closeEvent(self, event):
        # 如果界面已锁定，要求密码
        if self.is_locked:
            admin_pw = self.config.get('admin', {}).get('password', '')
            pwd, ok = QInputDialog.getText(self, "退出确认", "界面已锁定，请输入管理员密码以退出:", QLineEdit.Password)
            if not ok or pwd != admin_pw:
                QMessageBox.warning(self, "错误", "密码错误，无法退出程序")
                event.ignore()
                return

        # 检查监控是否正在运行
        if self.monitor_thread and self.monitor_thread.isRunning():
            # 弹出退出确认对话框
            reply = QMessageBox.question(
                self,
                "确认退出",
                "⚠️ 监控正在运行中！\n\n确定要退出程序吗？\n退出将停止所有监控任务。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No  # 默认选择"No"
            )
            
            if reply == QMessageBox.No:
                event.ignore()  # 取消退出
                return
            
            # 用户确认退出，执行清理操作
            self.log_signal.emit("正在停止监控并退出程序...\n")
            
            try:
                if self.monitor_thread.monitor and self.monitor_thread.monitor.current_automation:
                    self.monitor_thread.monitor.current_automation.close()
                    self.monitor_thread.monitor.current_automation = None
            except Exception as e:
                self.log_signal.emit(f"关闭driver出错: {e}\n")

            self.monitor_thread.stop()
            self.monitor_thread.wait(2000)

        event.accept()


def main():
    app = QApplication(sys.argv)

    # 加载配置
    config = load_config()
    # 输入管理员密码
    admin_password = config.get("admin", {}).get("password", "aionaion")
    password, ok = QInputDialog.getText(None, "管理员登录", "请输入管理员密码:", QLineEdit.Password)
    if not ok:
        sys.exit(1)
    if password != admin_password:
        QMessageBox.critical(None, "错误", "密码错误，程序退出")
        sys.exit(1)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()