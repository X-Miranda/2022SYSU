import os
import re
from flask import Flask, render_template, url_for, request, redirect, session
from flask import flash
import mysql.connector
from datetime import datetime


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')  # 从环境变量获取密钥
app.config['DEBUG'] = False


DB_CONFIG = {
    "user": "demonnn7",
    "password": "abc14789sysu",
    "host": "demonnn7.mysql.pythonanywhere-services.com",
    "port": "3306",
    "database": "demonnn7$gym_system",
    "raise_on_warnings": True,
    "time_zone": "+08:00"
}


def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"数据库连接失败: {e}")
        return None

# 确保静态文件路径正确
@app.route('/')
def home():
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db_connection()
        if conn is None:
            return "Failed to connect to the database.", 500
        cursor = conn.cursor(dictionary=True)
        try:
            # 首先检查用户名是否存在
            cursor.execute('SELECT username FROM Users WHERE username = %s', (username,))
            if not cursor.fetchone():
                flash('用户名不存在', 'error')
                return redirect(url_for('login'))

            # 检查用户名和密码是否匹配
            cursor.execute('SELECT role FROM Users WHERE username = %s AND password = %s', (username, password))
            user = cursor.fetchone()
            if user:
                session['role'] = user['role']  # 将用户角色存储在会话中
                if user['role'] == 'admin':
                    return redirect(url_for('manage_members'))
                else:
                    cursor.execute('SELECT member_id FROM Members WHERE name = %s', (username,))
                    member_info = cursor.fetchone()
                    if member_info:
                        session['member_id'] = member_info['member_id']  # 将 member_id 存储在会话中
                        return redirect(url_for('view_courses'))
                    else:
                        flash('Member not found.', 'error')
                        return redirect(url_for('login'))
            else:
                flash('密码输入错误', 'error')
                return redirect(url_for('login'))
        except mysql.connector.Error as e:
            print(f"An error occurred while logging in: {e}")
            flash('登录失败，请重试', 'error')
            return redirect(url_for('login'))
        finally:
            cursor.close()
            conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        contact = request.form.get('contact')
        age = request.form.get('age')  # 年龄字段可能为空
        gender = request.form.get('gender')  # 性别字段可能为空

        # 验证用户名长度
        if len(username) > 8:
            return render_template('register.html', error="用户名不能超过8个字符")

        # 验证电话号码格式
        if not contact.isdigit() or len(contact) != 11:
            return render_template('register.html', error="电话号码必须是11位数字")

        # 验证年龄不小于12岁
        if age and int(age) < 12:
            return render_template('register.html', error="年龄必须不小于12岁")

        if password != confirm_password:
            return render_template('register.html', error="密码与确认密码不一致")

        conn = get_db_connection()
        if conn is None:
            return "Failed to connect to the database.", 500

        cursor = conn.cursor()
        try:
            cursor.execute('SELECT username FROM Users WHERE username = %s', (username,))
            if cursor.fetchone():
                return render_template('register.html', error="用户名已存在")

            cursor.execute('''
                INSERT INTO Users (username, password, role) VALUES (%s, %s, %s)
            ''', (username, password, 'user'))
            conn.commit()

            # 插入 Members 表，会员等级明确设置为整数1
            cursor.execute('''
                INSERT INTO Members (name, gender, age, contact, membership_level)
                VALUES (%s, %s, %s, %s, %s)
            ''', (username, gender if gender else None, int(age) if age else None, contact, 1))  # 确保membership_level是整数
            conn.commit()
        except mysql.connector.Error as e:
            print(f"An error occurred while registering: {e}")
            conn.rollback()
            return "Failed to register.", 500
        finally:
            cursor.close()
            conn.close()
        return redirect(url_for('manage_members'))

@app.route('/members', methods=['GET'])
def manage_members():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute('SELECT member_id, name, gender, age, contact, membership_level FROM Members;')
        members = cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"An error occurred while fetching members: {e}")
        return "Failed to fetch members from the database.", 500
    finally:
        cursor.close()
        conn.close()
    return render_template('members.html', members=members)

@app.route('/members/add', methods=['GET', 'POST'])
def add_member():
    if 'role' not in session or session['role']!= 'admin':
        return redirect(url_for('login'))
    if request.method == 'GET':
        return render_template('add_member.html')

    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500

    cursor = conn.cursor()
    try:
        name = request.form['name']
        gender = request.form['gender']
        age = request.form['age']
        contact = request.form['contact']
        membership_level = request.form['membership_level']

        # 验证用户名长度
        if len(name) > 8:
            flash("用户名不能超过8个字符", 'error')
            return redirect(url_for('add_member'))

        # 验证电话号码格式
        if not contact.isdigit() or len(contact) != 11:
            flash("电话号码必须是11位数字", 'error')
            return redirect(url_for('add_member'))

        # 检查用户名是否已存在
        cursor.execute('SELECT username FROM Users WHERE username = %s', (name,))
        if cursor.fetchone():
            flash("添加失败，该用户已存在", 'error')
            return redirect(url_for('add_member'))

        # 设置默认密码为 666666
        default_password = '666666'
        # 先向 Users 表添加用户信息
        cursor.execute(
            "INSERT INTO Users (username, password, role) VALUES (%s, %s, %s)",
            (name, default_password, 'user')
        )
        # 再向 Members 表添加成员信息
        cursor.execute(
            "INSERT INTO Members (name, gender, age, contact, membership_level) VALUES (%s, %s, %s, %s, %s)",
            (name, gender, age, contact, membership_level)
        )
        conn.commit()
        flash("会员添加成功", 'success')
    except mysql.connector.Error as e:
        print(f"An error occurred while adding a member: {e}")
        conn.rollback()
        flash("添加会员失败，请重试", 'error')
        return redirect(url_for('add_member'))
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('manage_members'))


@app.route('/members/edit/<int:member_id>', methods=['GET', 'POST'])
def edit_member(member_id):
    if 'role' not in session or session['role']!= 'admin':
        return redirect(url_for('login'))
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor(dictionary=True)
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            age = request.form['age']
            contact = request.form['contact']
            membership_level = request.form['membership_level']

            # 验证电话号码格式
            if not contact.isdigit() or len(contact) != 11:
                flash("电话号码必须是11位数字", 'error')
                return redirect(url_for('edit_member', member_id=member_id))

            cursor.execute(
                "UPDATE Members SET gender = %s, age = %s, contact = %s, membership_level = %s WHERE member_id = %s",
                (gender, age, contact, membership_level, member_id)
            )
            conn.commit()
            return redirect(url_for('manage_members'))
        except mysql.connector.Error as e:
            print(f"An error occurred while editing a member: {e}")
            conn.rollback()
            return "Failed to edit a member in the database.", 500
    else:
        try:
            cursor.execute('SELECT member_id, name, gender, age, contact, membership_level FROM Members WHERE member_id = %s', (member_id,))
            member = cursor.fetchone()
            if member is None:
                return "Member not found.", 404
        except mysql.connector.Error as e:
            print(f"An error occurred while fetching a member for editing: {e}")
            return "Failed to fetch a member for editing from the database.", 500
        finally:
            cursor.close()
            conn.close()
    return render_template('edit_member.html', member=member)

@app.route('/members/delete/<int:member_id>', methods=['POST'])
def delete_member(member_id):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500

    cursor = conn.cursor()
    try:
        # 首先获取会员名，用于删除Users表中的记录
        cursor.execute('SELECT name FROM Members WHERE member_id = %s', (member_id,))
        member_name = cursor.fetchone()

        if not member_name:
            return "Member not found.", 404

        # 先删除预约记录（如果有）
        try:
            cursor.execute('DELETE FROM Reservations WHERE member_id = %s', (member_id,))
            print(f"Deleted reservations for member {member_id}")
        except mysql.connector.Error as e:
            print(f"Warning: Could not delete reservations - {e}")
            # 继续执行，可能没有预约记录

        # 然后删除Members表中的成员信息
        cursor.execute('DELETE FROM Members WHERE member_id = %s', (member_id,))
        print(f"Deleted member {member_id} from Members table")

        # 最后删除Users表中的用户信息
        cursor.execute('DELETE FROM Users WHERE username = %s', (member_name[0],))
        print(f"Deleted user {member_name[0]} from Users table")

        conn.commit()
        return redirect(url_for('manage_members'))

    except mysql.connector.Error as e:
        print(f"Error during deletion: {e}")
        conn.rollback()
        # 返回更详细的错误信息
        return f"Failed to delete member. Database error: {e}", 500
    finally:
        cursor.close()
        conn.close()

@app.route('/search_members', methods=['GET'])
def search_members():
    if 'role' not in session or session['role']!= 'admin':
        return redirect(url_for('login'))
    search_query = request.args.get('search')
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor(dictionary=True)
    try:
        if search_query:
            # 搜索用户
            cursor.execute('''
                SELECT *
                FROM Members
                WHERE name LIKE %s
            ''', (f'%{search_query}%',))
        else:
            # 如果没有搜索内容，显示全部用户
            cursor.execute('SELECT * FROM Members')
        members = cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"An error occurred while searching members: {e}")
        return "Failed to search members from the database.", 500
    finally:
        cursor.close()
        conn.close()
    return render_template('members.html', members=members, current_page='members')


@app.route('/view_courses')
def view_courses():
    if 'role' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor(dictionary=True)
    try:
        # 获取课程信息
        cursor.execute('''
            SELECT c.course_id, c.course_name, c.duration, c.coach, c.price
            FROM Courses c
        ''')
        courses = cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"An error occurred while fetching courses: {e}")
        return "Failed to fetch courses from the database.", 500
    finally:
        cursor.close()
        conn.close()
    # 将 is_reserved_by_current_user 函数传递给模板
    return render_template('courses.html', courses=courses, current_page='courses', is_reserved_by_current_user=is_reserved_by_current_user)

@app.route('/courses/add', methods=['GET', 'POST'])
def add_course():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    if request.method == 'GET':
        return render_template('add_courses.html')

    # 获取表单数据
    course_name = request.form['course_name']
    duration = request.form['duration']
    coach = request.form['coach']
    price = request.form['price']

    # 验证上课时间格式 (HH:MM-HH:MM) 且在 0:00-24:00 之间
    if not re.match(r'^([01]?[0-9]|2[0-3]):([0-5][0-9])-([01]?[0-9]|2[0-3]):([0-5][0-9])$', duration):
        flash("上课时间格式不正确，请使用 HH:MM-HH:MM 格式且在0:00-24:00之间", 'error')
        return redirect(url_for('add_course'))

    # 验证价格是否为负数
    try:
        price = float(price)
        if price < 0:
            flash("请填写规范的价格（不能为负数）", 'error')
            return redirect(url_for('add_course'))
    except ValueError:
        flash("请填写有效的价格（数字）", 'error')
        return redirect(url_for('add_course'))

    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500

    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO Courses (course_name, duration, coach, price) VALUES (%s, %s, %s, %s)",
            (course_name, duration, coach, price)
        )
        conn.commit()
        flash("课程添加成功", 'success')
    except mysql.connector.Error as e:
        print(f"An error occurred while adding a course: {e}")
        conn.rollback()
        flash("添加课程失败，请重试", 'error')
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('view_courses'))

@app.route('/courses/edit/<int:course_id>', methods=['GET', 'POST'])
def edit_course(course_id):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500

    cursor = conn.cursor()

    if request.method == 'POST':
        # 获取表单数据
        course_name = request.form['course_name']
        duration = request.form['duration']
        coach = request.form['coach']
        price = request.form['price']

        # 验证上课时间格式 (HH:MM-HH:MM) 且在 0:00-24:00 之间
        if not re.match(r'^([01]?[0-9]|2[0-3]):([0-5][0-9])-([01]?[0-9]|2[0-3]):([0-5][0-9])$', duration):
            flash("上课时间格式不正确，请使用 HH:MM-HH:MM 格式且在0:00-24:00之间", 'error')
            return redirect(url_for('edit_course', course_id=course_id))

        # 验证价格是否为负数
        try:
            price = float(price)
            if price < 0:
                flash("请填写规范的价格（不能为负数）", 'error')
                return redirect(url_for('edit_course', course_id=course_id))
        except ValueError:
            flash("请填写有效的价格（数字）", 'error')
            return redirect(url_for('edit_course', course_id=course_id))

        try:
            cursor.execute(
                "UPDATE Courses SET course_name = %s, duration = %s, coach = %s, price = %s WHERE course_id = %s",
                (course_name, duration, coach, price, course_id)
            )
            conn.commit()
            flash("课程修改成功", 'success')
            return redirect(url_for('view_courses'))
        except mysql.connector.Error as e:
            print(f"An error occurred while editing a course: {e}")
            conn.rollback()
            flash("修改课程失败，请重试", 'error')
            return redirect(url_for('edit_course', course_id=course_id))
    else:
        try:
            cursor.execute('SELECT course_id, course_name, duration, coach, price FROM Courses WHERE course_id = %s', (course_id,))
            course = cursor.fetchone()
            if course is None:
                return "Course not found.", 404
            course = {'course_id': course[0], 'course_name': course[1], 'duration': course[2], 'coach': course[3], 'price': course[4]}
        except mysql.connector.Error as e:
            print(f"An error occurred while fetching a course for editing: {e}")
            return "Failed to fetch a course for editing from the database.", 500
        finally:
            cursor.close()
            conn.close()

    return render_template('edit_courses.html', course=course)



@app.route('/courses/delete/<int:course_id>', methods=['POST'])
def delete_course(course_id):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500

    cursor = conn.cursor()
    try:
        # 首先检查课程是否被预约
        cursor.execute('SELECT 1 FROM Reservations WHERE course_id = %s LIMIT 1', (course_id,))
        if cursor.fetchone():
            flash("该课程已被预约，无法删除", 'error')
            return redirect(url_for('view_courses'))

        # 如果没有预约，则删除课程
        cursor.execute('DELETE FROM Courses WHERE course_id = %s', (course_id,))
        conn.commit()
        flash("课程删除成功", 'success')
    except mysql.connector.Error as e:
        print(f"An error occurred while deleting a course: {e}")
        conn.rollback()
        flash("删除课程失败，请重试", 'error')
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for('view_courses'))

@app.route('/search_courses', methods=['GET'])
def search_courses():
    if 'role' not in session:
        return redirect(url_for('login'))
    search_query = request.args.get('search')
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor(dictionary=True)
    try:
        if search_query:
            # 只搜索课程基本信息，不检查预约状态
            cursor.execute('''
                SELECT c.course_id, c.course_name, c.duration, c.coach, c.price
                FROM Courses c
                WHERE c.course_name LIKE %s
            ''', (f'%{search_query}%',))
        else:
            # 如果没有搜索内容，显示全部课程
            cursor.execute('''
                SELECT c.course_id, c.course_name, c.duration, c.coach, c.price
                FROM Courses c
            ''')
        courses = cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"An error occurred while searching courses: {e}")
        return "Failed to search courses from the database.", 500
    finally:
        cursor.close()
        conn.close()
    # 使用与view_courses相同的模板和函数
    return render_template('courses.html', courses=courses, current_page='courses', is_reserved_by_current_user=is_reserved_by_current_user)

@app.route('/profile')
def view_profile():
    if 'role' not in session:
        return redirect(url_for('login'))
    member_id = session.get('member_id')  # 假设我们存储了用户的 member_id 在会话中
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor()
    try:
        # 移除对 member_id 的查询，只查询用户的其他信息
        cursor.execute('SELECT name, gender, age, contact, membership_level FROM Members WHERE member_id = %s', (member_id,))
        member = cursor.fetchone()
        if member is None:
            return "Member not found.", 404
        # 只存储所需的用户信息，不包括 member_id
        member = {'name': member[0], 'gender': member[1], 'age': member[2], 'contact': member[3], 'membership_level': member[4]}
    except mysql.connector.Error as e:
        print(f"An error occurred while fetching a member for profile: {e}")
        return "Failed to fetch a member for profile from the database.", 500
    finally:
        cursor.close()
        conn.close()
    return render_template('profile.html', member=member)


@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'role' not in session:
        return redirect(url_for('login'))

    member_id = session.get('member_id')
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500

    if request.method == 'GET':
        # 获取当前用户信息
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT contact FROM Members WHERE member_id = %s', (member_id,))
            contact = cursor.fetchone()[0]
        except mysql.connector.Error as e:
            print(f"An error occurred while fetching contact for edit profile: {e}")
            return "Failed to fetch contact for edit profile from the database.", 500
        finally:
            cursor.close()
        return render_template('edit_profile.html', contact=contact)

    elif request.method == 'POST':
        # 处理用户提交的修改信息
        new_contact = request.form.get('contact')
        new_password = request.form.get('password')

        # 验证电话号码格式
        if not new_contact.isdigit() or len(new_contact) != 11:
            flash("电话格式不正确", 'error')
            return redirect(url_for('edit_profile'))

        # 验证密码强度（如果提供了新密码）
        if new_password:
            # 检查密码长度
            if len(new_password) < 6 or len(new_password) > 20:
                flash("密码长度必须为6-20位", 'error')
                return redirect(url_for('edit_profile'))

            # 检查是否包含空格
            if ' ' in new_password:
                flash("密码不能包含空格", 'error')
                return redirect(url_for('edit_profile'))

            # 检查字符类型
            has_upper = any(c.isupper() for c in new_password)
            has_lower = any(c.islower() for c in new_password)
            has_number = any(c.isdigit() for c in new_password)
            has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', new_password))

            type_count = sum([has_upper, has_lower, has_number, has_special])
            if type_count < 2:
                flash("密码必须包含大写字母、小写字母、数字和特殊符号中的至少两种", 'error')
                return redirect(url_for('edit_profile'))

        cursor = conn.cursor()
        try:
            # 检查密码是否与原密码相同
            if new_password:
                # 获取原密码
                cursor.execute('''
                    SELECT password FROM Users
                    WHERE username = (SELECT name FROM Members WHERE member_id = %s)
                ''', (member_id,))
                old_password = cursor.fetchone()[0]

                if new_password == old_password:
                    flash("该密码与原密码一致，请重新填写", 'error')
                    return redirect(url_for('edit_profile'))

            # 更新 Members 表中的联系电话
            cursor.execute('UPDATE Members SET contact = %s WHERE member_id = %s', (new_contact, member_id))

            # 如果有新密码，更新 Users 表中的密码
            if new_password:
                cursor.execute('''
                    UPDATE Users SET password = %s
                    WHERE username = (SELECT name FROM Members WHERE member_id = %s)
                ''', (new_password, member_id))

            conn.commit()
            flash("个人信息更新成功", 'success')
            return redirect(url_for('view_profile'))

        except mysql.connector.Error as e:
            print(f"An error occurred while updating profile: {e}")
            conn.rollback()
            flash("更新个人信息失败，请重试", 'error')
            return redirect(url_for('edit_profile'))
        finally:
            cursor.close()
            conn.close()


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('role', None)  # 清除用户角色信息
    return render_template('login.html')

@app.route('/reservations_by_member/<int:member_id>')
def view_reservations_by_member(member_id):
    if 'role' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor()
    try:
        # 获取预约信息，同时关联 Courses 表获取课程名称、课程时长和教练信息
        cursor.execute('''
            SELECT r.reservation_id, r.course_id, c.course_name, c.duration AS course_duration,
           c.coach, CONVERT_TZ(r.reservation_time, '+00:00', '+08:00') AS reservation_time
            FROM Reservations r
            JOIN Courses c ON r.course_id = c.course_id
            WHERE r.member_id = %s
        ''', (member_id,))
        reservations = [{'reservation_id': row[0], 'course_id': row[1], 'course_name': row[2], 'course_duration': row[3], 'coach': row[4], 'reservation_time': row[5]} for row in cursor.fetchall()]
        # 获取 member 信息
        cursor.execute('SELECT name FROM Members WHERE member_id = %s', (member_id,))
        member = cursor.fetchone()
    except mysql.connector.Error as e:
        print(f"An error occurred while fetching reservations: {e}")
        return "Failed to fetch reservations from the database.", 500
    finally:
        cursor.close()
        conn.close()
    return render_template('view_reservations_by_member.html', reservations=reservations, member_id=member_id, member={'name': member[0]})

@app.route('/make_reservation/<int:course_id>', methods=['POST'])
def make_reservation(course_id):
    if 'role' not in session or session['role'] != 'user':
        return redirect(url_for('login'))
    member_id = session.get('member_id')
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor()
    try:
        # 使用Python的本地时间（东八区）
        local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO Reservations (member_id, course_id, reservation_time)
            VALUES (%s, %s, %s)
        ''', (member_id, course_id, local_time))
        conn.commit()
    except mysql.connector.Error as e:
        print(f"An error occurred while making a reservation: {e}")
        conn.rollback()
        return "Failed to make a reservation.", 500
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('view_courses'))


@app.route('/cancel_reservation/<int:course_id>', methods=['POST'])
def cancel_reservation(course_id):
    if 'role' not in session or session['role']!= 'user':
        return redirect(url_for('login'))
    member_id = session.get('member_id')
    conn = get_db_connection()
    if conn is None:
        return "Failed to connect to the database.", 500
    cursor = conn.cursor()
    try:
        # 删除当前用户对该课程的预约记录
        cursor.execute('''
            DELETE FROM Reservations
            WHERE member_id = %s AND course_id = %s
        ''', (member_id, course_id))
        conn.commit()
    except mysql.connector.Error as e:
        print(f"An error occurred while canceling a reservation: {e}")
        conn.rollback()
        return "Failed to cancel a reservation.", 500
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('view_courses'))

def is_reserved_by_current_user(course_id):
    if 'role' not in session or session['role']!= 'user':
        return False
    member_id = session.get('member_id')
    conn = get_db_connection()
    if conn is None:
        return False
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT 1 FROM Reservations
            WHERE member_id = %s AND course_id = %s
        ''', (member_id, course_id))
        result = cursor.fetchone()
        return result is not None
    except mysql.connector.Error as e:
        print(f"An error occurred while checking reservation: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

# if __name__ == '__main__':
#     app.run(debug=True)