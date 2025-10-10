-- 用户表
CREATE TABLE Users (
    username VARCHAR(100) PRIMARY KEY,
    password VARCHAR(100),
    role VARCHAR(10)
);

-- 会员表
CREATE TABLE Members (
    member_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    gender CHAR(1),
    age INT,
    contact VARCHAR(100),
    membership_level VARCHAR(50)
);

-- 课程表
CREATE TABLE Courses (
    course_id INT AUTO_INCREMENT PRIMARY KEY,
    course_name VARCHAR(100) NOT NULL,
    duration VARCHAR(50),
    coach VARCHAR(100),
    price DECIMAL(10, 2)
);

-- 预约表
CREATE TABLE Reservations (
    reservation_id INT AUTO_INCREMENT PRIMARY KEY,
    member_id INT,
    course_id INT,
    reservation_time DATETIME,
    FOREIGN KEY (member_id) REFERENCES Members(member_id),
    FOREIGN KEY (course_id) REFERENCES Courses(course_id)
);

-- 初始创建一个管理员用户
INSERT INTO Users (username, password, role) VALUES ('sysu', '8888', 'admin');