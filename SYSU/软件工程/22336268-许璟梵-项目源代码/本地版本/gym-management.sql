--用户表--
CREATE TABLE Users (
    username VARCHAR(100) PRIMARY KEY, --用户名，主键
    password VARCHAR(100), --密码
    role VARCHAR(10)  -- 用户身份，可以是 'admin' 或 'member'
);

--会员表--
CREATE TABLE Members (
    member_id SERIAL PRIMARY KEY,  --会员号，主键
    name VARCHAR(100) NOT NULL,  --会员姓名
    gender CHAR(1),  --会员性别
    age INT,   --会员年龄
    contact VARCHAR(100),  --会员联系电话
    membership_level VARCHAR(50)  --会员等级
);

--课程表--
CREATE TABLE Courses (
    course_id SERIAL PRIMARY KEY,  --课程号，主键
    course_name VARCHAR(100) NOT NULL,  --课程名
    duration VARCHAR(50),  --课程持续时间
    coach VARCHAR(100),  --课程教练
    price DECIMAL(10, 2)  --课程价格
);

--预约表--
CREATE TABLE Reservations (
    reservation_id SERIAL PRIMARY KEY,  --预约号，主键
    member_id INT REFERENCES Members(member_id),  --会员号，来自会员表的外键
    course_id INT REFERENCES Courses(course_id),  --课程号，来自课程表的外键
    reservation_time TIMESTAMP  --预约时间段
);

--初始创建一个管理员用户--
INSERT INTO users (username, password, role) VALUES ('sysu', '8888', 'admin');

