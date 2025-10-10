
#include "myglwidget.h"
#include <GL/glew.h>
#include <algorithm>
#include <chrono>

MyGLWidget::MyGLWidget(QWidget* parent)
	:QOpenGLWidget(parent)
{
}

MyGLWidget::~MyGLWidget()
{
	delete[] render_buffer;
	delete[] temp_render_buffer;
	delete[] temp_z_buffer;
	delete[] z_buffer;
}

vec3 normalize(vec3 v) {
	float len = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	return vec3(v.x / len, v.y / len, v.z / len);
}

void MyGLWidget::resizeBuffer(int newW, int newH) {
	delete[] render_buffer;
	delete[] temp_render_buffer;
	delete[] temp_z_buffer;
	delete[] z_buffer;
	WindowSizeW = newW;
	WindowSizeH = newH;
	render_buffer = new vec3[WindowSizeH * WindowSizeW];
	temp_render_buffer = new vec3[WindowSizeH * WindowSizeW];
	temp_z_buffer = new float[WindowSizeH * WindowSizeW];
	z_buffer = new float[WindowSizeH * WindowSizeW];
}

void MyGLWidget::initializeGL()
{
	WindowSizeW = width();
	WindowSizeH = height();
	glViewport(0, 0, WindowSizeW, WindowSizeH);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glDisable(GL_DEPTH_TEST);
	offset = vec2(WindowSizeH / 2, WindowSizeW / 2);
	// 对定义的数组初始化
	render_buffer = new vec3[WindowSizeH * WindowSizeW];
	temp_render_buffer = new vec3[WindowSizeH * WindowSizeW];
	temp_z_buffer = new float[WindowSizeH * WindowSizeW];
	z_buffer = new float[WindowSizeH * WindowSizeW];
	for (int i = 0; i < WindowSizeH * WindowSizeW; i++) {
		render_buffer[i] = vec3(0, 0, 0);
		temp_render_buffer[i] = vec3(0, 0, 0);
		temp_z_buffer[i] = MAX_Z_BUFFER;
		z_buffer[i] = MAX_Z_BUFFER;
	}
}

void MyGLWidget::keyPressEvent(QKeyEvent* e) {

	switch (e->key()) {
	case Qt::Key_0: scene_id = 0; update(); break;
	case Qt::Key_1: scene_id = 1; update(); break;
	case Qt::Key_2: draw_id = 2; std::cout << "DDA:" << std::endl; update(); break;
	case Qt::Key_3: draw_id = 3; std::cout << "bresenham:" << std::endl; update(); break;
	case Qt::Key_4: draw_id = 4; std::cout << "Gouraud:" << std::endl; update(); break;
	case Qt::Key_5: draw_id = 5; std::cout << "Phong:" << std::endl; update(); break;
	case Qt::Key_6: draw_id = 6; std::cout << "Blinn-Phong:" << std::endl; update(); break;
	case Qt::Key_9: degree += 35; update(); break;
	}
}

void MyGLWidget::paintGL()
{
	switch (scene_id) {
	case 0:scene_0(); break;
	case 1:scene_1(); break;
	}
}
void MyGLWidget::clearBuffer(vec3* now_buffer) {
	for (int i = 0; i < WindowSizeH * WindowSizeW; i++) {
		now_buffer[i] = vec3(0, 0, 0);
	}
}

void MyGLWidget::clearBuffer(int* now_buffer) {
	memset(now_buffer, 0, WindowSizeW * WindowSizeH * sizeof(int));
}


void MyGLWidget::clearZBuffer(float* now_buffer) {
	std::fill(now_buffer, now_buffer + WindowSizeW * WindowSizeH, MAX_Z_BUFFER);
}


// 窗口大小变动后，需要重新生成render_buffer等数组
void MyGLWidget::resizeGL(int w, int h)
{
	resizeBuffer(w, h);
	offset = vec2(WindowSizeH / 2, WindowSizeW / 2);
	clearBuffer(render_buffer);
}

void MyGLWidget::scene_0()
{
	// 选择要加载的model
	//objModel.loadModel("./objs/singleTriangle.obj");
	objModel.loadModel("./objs/teapot_600.obj");

	// 自主设置变换矩阵
	camPosition = vec3(100 * sin(degree * 3.14 / 180.0) + objModel.centralPoint.y, 100 * cos(degree * 3.14 / 180.0) + objModel.centralPoint.x, 10 + objModel.centralPoint.z);
	camLookAt = objModel.centralPoint;     // 例如，看向物体中心
	camUp = vec3(0, 1, 0);         // 上方向向量
	projMatrix = glm::perspective(radians(20.0f), 1.0f, 0.1f, 2000.0f);

	// 单一点光源，可以改为数组实现多光源
	lightPosition = objModel.centralPoint + vec3(-100, 0, 20);
	clearBuffer(render_buffer);
	clearZBuffer(z_buffer);
	auto start_time = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < objModel.triangleCount; i++) {
		Triangle nowTriangle = objModel.getTriangleByID(i);
		drawTriangle(nowTriangle);
	}
	auto end_time = std::chrono::high_resolution_clock::now(); // 结束时间
	// 计算并输出算法的执行时间
	std::chrono::duration<double, std::milli> ddaTime = end_time - start_time;
	std::cout << "Time: " << ddaTime.count() << " ms" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT);
	renderWithTexture(render_buffer, WindowSizeH, WindowSizeW);
}


void MyGLWidget::scene_1()
{
	// 选择要加载的model
	//objModel.loadModel("./objs/teapot_600.obj");
	//objModel.loadModel("./objs/teapot_8000.obj");
	objModel.loadModel("./objs/rock.obj");
	//objModel.loadModel("./objs/cube.obj");
	//objModel.loadModel("./objs/singleTriangle.obj");

	// 自主设置变换矩阵
	camPosition = vec3(100 * sin(degree * 3.14 / 180.0) + objModel.centralPoint.y, 100 * cos(degree * 3.14 / 180.0) + objModel.centralPoint.x, 10 + objModel.centralPoint.z);
	camLookAt = objModel.centralPoint;     // 例如，看向物体中心
	camUp = vec3(0, 1, 0);         // 上方向向量
	projMatrix = glm::perspective(radians(20.0f), 1.0f, 0.1f, 2000.0f);

	// 单一点光源，可以改为数组实现多光源
	lightPosition = objModel.centralPoint + vec3(0, 100, 100);
	clearBuffer(render_buffer);
	clearZBuffer(z_buffer);
	auto start_time = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < objModel.triangleCount; i++) {
		Triangle nowTriangle = objModel.getTriangleByID(i);
		drawTriangle(nowTriangle);
	}
	auto end_time = std::chrono::high_resolution_clock::now(); // 结束时间
	// 计算并输出算法的执行时间
	std::chrono::duration<double, std::milli> ddaTime = end_time - start_time;
	std::cout << "Time: " << ddaTime.count() << " ms" << std::endl;
	glClear(GL_COLOR_BUFFER_BIT);
	renderWithTexture(render_buffer, WindowSizeH, WindowSizeW);
}


void MyGLWidget::drawTriangle(Triangle triangle) {
	// 三维顶点映射到二维平面
	vec3* vertices = triangle.triangleVertices;
	vec3* normals = triangle.triangleNormals;
	FragmentAttr transformedVertices[3];
	clearBuffer(this->temp_render_buffer);
	clearZBuffer(this->temp_z_buffer);
	mat4 viewMatrix = glm::lookAt(camPosition, camLookAt, camUp);

	for (int i = 0; i < 3; ++i) {
		vec4 ver_mv = viewMatrix * vec4(vertices[i], 1.0f);
		float nowz = glm::length(camPosition - vec3(ver_mv));
		vec4 ver_proj = projMatrix * ver_mv;
		transformedVertices[i].x = ver_proj.x + offset.x;
		transformedVertices[i].y = ver_proj.y + offset.y;
		transformedVertices[i].z = nowz;
		transformedVertices[i].pos_mv = ver_mv;
		mat3 normalMatrix = mat3(viewMatrix);
		vec3 normal_mv = normalMatrix * normals[i];
		transformedVertices[i].normal = normal_mv;
	}
	int firstChangeLine = 0;
	// 将当前三角形渲染在temp_buffer中
	// HomeWork: 1、绘制三角形三边
	// 使用DDA算法绘制三角形
	if (draw_id == 2) {
		DDA(transformedVertices[0], transformedVertices[1], 1);
		DDA(transformedVertices[1], transformedVertices[2], 2);
		DDA(transformedVertices[2], transformedVertices[0], 3);
	}

	// 使用Bresenham算法绘制三角形
	else if (draw_id == 3) {

		bresenham(transformedVertices[0], transformedVertices[1], 1);
		bresenham(transformedVertices[1], transformedVertices[2], 2);
		bresenham(transformedVertices[2], transformedVertices[0], 3);
	}

	// 使用Bresenham算法绘制三角形
	else if (3 < draw_id < 7) {
		bresenham_light(transformedVertices[0], transformedVertices[1], 1);
		bresenham_light(transformedVertices[1], transformedVertices[2], 2);
		bresenham_light(transformedVertices[2], transformedVertices[0], 3);

	}

	// HomeWork: 2: 用edge-walking填充三角形内部到temp_buffer中
	firstChangeLine = edge_walking(transformedVertices);

	// 合并temp_buffer 到 render_buffer, 深度测试
	// 从firstChangeLine开始遍历，可以稍快
	for (int h = firstChangeLine; h < WindowSizeH; h++) {
		auto render_row = &render_buffer[h * WindowSizeW];
		auto temp_render_row = &temp_render_buffer[h * WindowSizeW];
		auto z_buffer_row = &z_buffer[h * WindowSizeW];
		auto temp_z_buffer_row = &temp_z_buffer[h * WindowSizeW];
		for (int i = 0; i < WindowSizeW; i++) {
			if (z_buffer_row[i] < temp_z_buffer_row[i])
				continue;
			else
			{
				z_buffer_row[i] = temp_z_buffer_row[i];
				render_row[i] = temp_render_row[i];
			}
		}

	}
}

std::pair<vec3, vec3> Interpolate(int x, int y, FragmentAttr vertices[3]) {
	vec3 p = vec3(x, y, 0); // 将输入的x和y坐标转换为vec3

	// 计算重心坐标
	vec3 v0 = vec3(vertices[1].x - vertices[0].x, vertices[1].y - vertices[0].y, vertices[1].z - vertices[0].z);
	vec3 v1 = vec3(vertices[2].x - vertices[0].x, vertices[2].y - vertices[0].y, vertices[2].z - vertices[0].z);
	vec3 v2 = vec3(p.x - vertices[0].x, p.y - vertices[0].y, 0); // 假设z值为0，因为x和y是屏幕坐标

	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);
	float denom = d00 * d11 - d01 * d01;

	vec3 bary;
	bary.y = (d11 * d20 - d01 * d21) / denom;
	bary.z = (d00 * d21 - d01 * d20) / denom;
	bary.x = 1.0f - bary.y - bary.z;

	// 通过重心坐标插值法向量
	vec3 interpolatedNormal = normalize(bary.x * vertices[0].normal + bary.y * vertices[1].normal + bary.z * vertices[2].normal);
	vec3 interpolatedPosmv = normalize(bary.x * vertices[0].pos_mv + bary.y * vertices[1].pos_mv + bary.z * vertices[2].pos_mv);
	return std::make_pair(interpolatedNormal, interpolatedPosmv);
}

int MyGLWidget::edge_walking(FragmentAttr transformedVertices[3]) {
	int firstChangeLine = WindowSizeH;

	for (int x = 0; x < WindowSizeH; x++) {
		bool inside = false;
		int start = 0;
		int end = 0;
		bool foundStart = false;
		float startDepth = 99999.0f, endDepth = 99999.0f;
		vec3 startColor, endColor;

		for (int y = 1; y < WindowSizeW; y++) {
			// 检测一行中，三角形左边的边界
			if (!inside && temp_render_buffer[x * WindowSizeW + y] != vec3(0, 0, 0) && temp_render_buffer[x * WindowSizeW + y - 1] == vec3(0, 0, 0)) {
				inside = true;
				start = y;
				foundStart = true;
				startDepth = temp_z_buffer[x * WindowSizeW + y];
				startColor = temp_render_buffer[x * WindowSizeW + y]; // 获取起始边界的颜色
			}
			// 检测一行中，三角形右边的边界
			else if (inside && temp_render_buffer[x * WindowSizeW + y] != vec3(0, 0, 0) && temp_render_buffer[x * WindowSizeW + y - 1] == vec3(0, 0, 0)) {
				end = y;
				endDepth = temp_z_buffer[x * WindowSizeW + y];
				endColor = temp_render_buffer[x * WindowSizeW + y]; // 获取结束边界的颜色
				break;
			}
		}

		if (foundStart) {
			firstChangeLine = std::min(firstChangeLine, x);

			for (int y = start; y <= end; y++) {
				float t = (float)(y - start) / (float)(end - start + 0.01f);
				float depth = startDepth + (endDepth - startDepth) * t;

				vec3 color = vec3(0.0f, 0.0f, 0.0f); // 声明color变量
				switch (draw_id) {
				case 2: {
					  color = vec3(10, 0, 255); // 填充像素为粉色
					  break;
				}
				case 3:{
					color = vec3(10, 0, 255); // 填充像素为粉色
					break;
				}
				case 4:{
					color = vec3(startColor.x + (endColor.x - startColor.x) * t,
								 startColor.y + (endColor.y - startColor.y) * t,
								 startColor.z + (endColor.z - startColor.z) * t);
					break;
				}
				case 5:{
					FragmentAttr tmp = FragmentAttr(x, y, 0, 0);
					std::pair<vec3, vec3> result = Interpolate(x, y, transformedVertices);
					tmp.normal = result.first;
					tmp.pos_mv = result.second;
					color = PhongShading(tmp);
					break;
				}
				
				case 6:{
					FragmentAttr tmp = FragmentAttr(x, y, 0, 0);
					std::pair<vec3, vec3> result = Interpolate(x, y, transformedVertices);
					tmp.normal = result.first;
					tmp.pos_mv = result.second;
					color = BlinnPhongShading(tmp);
					break;
				}
				default:
					break;
				}

				temp_render_buffer[x * WindowSizeW + y] = color; // 填充颜色
				temp_z_buffer[x * WindowSizeW + y] = depth;
			}
		}
	}

	return firstChangeLine;
}

vec3 MyGLWidget::GouraudShading(FragmentAttr& fragment) {
	vec3 ambientColor = vec3(0.1, 0.1, 0.1);  // 环境光颜色
	vec3 lightColor = vec3(1.0, 1.0, 1.0);    // 光源颜色
	vec3 materialColor = vec3(1.0, 1.0, 1.0); // 材料颜色

	// 环境光部分
	vec3 ambient = ambientColor * materialColor;

	// 漫反射部分
	vec3 norm = normalize(fragment.normal);
	vec3 lightDir = normalize(lightPosition - vec3(fragment.pos_mv));
	float diff = max(dot(norm, lightDir), 0.0f);
	vec3 diffuse = diff * lightColor * materialColor;

	// 总光照强度
	vec3 intensity = ambient + diffuse;

	return intensity;
}

vec3 MyGLWidget::PhongShading(FragmentAttr& fragment) {
	vec3 ambientColor = vec3(0.1, 0.1, 0.1);  // 环境光颜色
	vec3 diffuseColor = vec3(0.9, 0.9, 0.9); // 漫反射颜色
	vec3 specularColor = vec3(0.2, 0.2, 0.2); // 镜面反射颜色
	vec3 lightColor = vec3(1.0, 1.0, 1.0);    // 光源颜色

	// 环境光分量
	vec3 color = ambientColor;

	// 漫反射分量
	vec3 lightDir = normalize(lightPosition - vec3(fragment.pos_mv)); // 光源方向
	vec3 norm = normalize(fragment.normal); // 法线
	float diff = max(dot(norm, lightDir), 0.0f); // 漫反射强度
	color += diff * diffuseColor;

	// 镜面反射分量
	vec3 viewDir = normalize(camPosition - vec3(fragment.pos_mv)); // 观察方向
	vec3 reflectDir = reflect(-lightDir, norm); // 反射方向
	float spec = pow(max(dot(viewDir, reflectDir), 0.0f), 32); // 镜面反射强度
	color += spec * specularColor;

	return color * lightColor; // 乘以光源颜色
}

vec3 MyGLWidget::BlinnPhongShading(FragmentAttr& fragment) {
	vec3 ambientColor = vec3(0.1, 0.1, 0.1);  // 环境光颜色
	vec3 lightColor = vec3(1.0, 1.0, 1.0);    // 光源颜色
	vec3 materialColor = vec3(1.0, 1.0, 1.0); // 材料颜色

	// 环境光部分
	vec3 ambient = ambientColor * materialColor;

	// 漫反射部分
	vec3 norm = normalize(fragment.normal);
	vec3 lightDir = normalize(lightPosition - vec3(fragment.pos_mv));
	float diff = max(dot(norm, lightDir), 0.0f);
	vec3 diffuse = diff * lightColor * materialColor;

	// 镜面反射部分
	vec3 viewDir = normalize(-vec3(fragment.pos_mv)); // 观察方向
	vec3 halfDir = normalize(lightDir + viewDir); // 半角向量
	float spec = pow(max(dot(norm, halfDir), 0.0f), 32); // 镜面反射强度
	vec3 specular = spec * lightColor;

	// 总光照强度
	vec3 intensity = ambient + diffuse + specular;

	return intensity;
}

void MyGLWidget::DDA(FragmentAttr& start, FragmentAttr& end, int id) {
	int dx = end.x - start.x;
	int dy = end.y - start.y;
	int steps = std::max(abs(dx), abs(dy)); // 步骤数基于最大的dx或dy

	float xIncrement = dx / (float)steps;
	float yIncrement = dy / (float)steps;
	float zIncrement = (end.z - start.z) / (float)steps; // 添加深度变化率

	float x = start.x;
	float y = start.y;
	float z = start.z; // 初始化z为起点深度

	for (int i = 0; i <= steps; i++) {
		int ix = static_cast<int>(x);
		int iy = static_cast<int>(y);
		temp_render_buffer[ix * WindowSizeW + iy] = vec3(0, 255, 0); // 绘制绿色像素
		temp_z_buffer[ix * WindowSizeW + iy] = z; // 更新深度信息
		x += xIncrement;
		y += yIncrement;
		z += zIncrement; // 更新深度值
	}
}


void MyGLWidget::bresenham(FragmentAttr& start, FragmentAttr& end, int id) {
	int x0 = static_cast<int>(start.x);
	int y0 = static_cast<int>(start.y);
	int x1 = static_cast<int>(end.x);
	int y1 = static_cast<int>(end.y);

	// 计算差值
	int dx = abs(x1 - x0);
	int dy = abs(y1 - y0);

	// 确定步进方向，1为向右或向下
	int sx = x0 < x1 ? 1 : -1;
	int sy = y0 < y1 ? 1 : -1;

	// 初始化误差值
	int err = (dx > dy ? dx : -dy) / 2;
	int e2;

	// 绘制直线
	while (true) {
		int ix = x0, iy = y0;
		temp_render_buffer[ix * WindowSizeW + iy] = vec3(0, 255, 0); // 绘制绿色像素
		temp_z_buffer[ix * WindowSizeW + iy] = start.z; // 更新深度信息

		if (x0 == x1 && y0 == y1) break;

		e2 = err;
		if (e2 > -dx) {
			err -= dy;
			x0 += sx;
		}
		if (e2 < dy) {
			err += dx;
			y0 += sy;
		}
	}
}

void MyGLWidget::bresenham_light(FragmentAttr& start, FragmentAttr& end, int id) {
	int x0 = static_cast<int>(start.x);
	int y0 = static_cast<int>(start.y);
	int x1 = static_cast<int>(end.x);
	int y1 = static_cast<int>(end.y);

	int dx = abs(x1 - x0);
	int dy = abs(y1 - y0);
	int sx = x0 < x1 ? 1 : -1;
	int sy = y0 < y1 ? 1 : -1;
	int err = (dx > dy ? dx : -dy) / 2;

	vec3 start_color, end_color, color;

	switch (draw_id) {
	case 4:
		start_color = GouraudShading(start);
		end_color = GouraudShading(end);
		break;
	case 5:
		start_color = PhongShading(start);
		end_color = PhongShading(end);
		break;
	case 6:
		start_color = BlinnPhongShading(start);
		end_color = BlinnPhongShading(end);
		break;
	}
	start.color = start_color;
	end.color = end_color;
	temp_render_buffer[start.x * WindowSizeW + start.y] = start_color;
	temp_render_buffer[end.x * WindowSizeW + end.y] = end_color;

	int e2;

	int totalSteps = max(abs(end.x - start.x), abs(end.y - start.y));
	float step = 0;

	while (true) {
		// 计算插值深度
		float depth = start.z + (end.z - start.z) * (step / (float)totalSteps);

		// 计算当前步的颜色
		vec3 color = start_color + (end_color - start_color) * (step / (float)totalSteps);

		// 更新颜色和深度缓冲区
		temp_render_buffer[x0 * WindowSizeW + y0] = color;
		temp_z_buffer[x0 * WindowSizeW + y0] = depth;

		if (x0 == x1 && y0 == y1) break;

		e2 = err;
		if (e2 > -dx) {
			err -= dy;
			x0 += sx;
		}
		if (e2 < dy) {
			err += dx;
			y0 += sy;
		}

		step++;
	}
}