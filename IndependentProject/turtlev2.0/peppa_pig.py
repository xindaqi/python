from turtle import *
import time



class PeppaPig(object):
	# 画笔移动,避免移动绘制轨迹
	# pu抬笔,goto移动,pd落笔
	def go_to(self, x, y):
		pu()
		goto(x, y)
		pd()
	# 画笔移动轨迹
	def pen_draw(self, x, y, z, h):	
		pu()
		seth(x)
		fd(y)
		seth(z)
		fd(h)
		pd()

	# 循环轨迹
	def loop_range(self, a, len):
		for i in range(len):
			if 0 <= i <30 or 60 <= i <90:
				a += 0.08
				lt(3)
				fd(a)
			else:
				a -= 0.08
				lt(3)
				fd(a)

	# 鼻子
	def nose(self):
		self.go_to(-100, 100)
		seth(-30)
		begin_fill()
		self.loop_range(0.4, 120)
		end_fill()
		self.nose_left()
		self.nose_right()
	# 左鼻孔
	def nose_left(self):
		self.pen_draw(90, 25, 10, 10)
		pencolor(255, 155, 192)
		seth(10)

		begin_fill()
		circle(5)
		color(160, 82, 45)
		end_fill()
	# 右鼻孔
	def nose_right(self):
		pu()
		seth(0)
		fd(20)
		pd()
		pencolor(255, 155, 192)
		seth(10)
		begin_fill()
		circle(5)
		color(160, 82, 45)
		end_fill()
	

	# 头
	def head(self):
		color((255, 155, 192), "pink")
		self.pen_draw(90, 41, 0, 0)
		

		begin_fill()
		seth(180)
		circle(300, -30)
		circle(100, -60)
		circle(80, -100)
		circle(150, -20)
		circle(60, -95)
		seth(161)
		circle(-300, 15)

		self.go_to(-100, 100)

		seth(-30)
		self.loop_range(0.4, 60)
		end_fill()
	# 耳朵	
	def ear(self):
		color((255, 155, 192), "pink")
		self.pen_draw(90, -7, 0, 70)
		self.ear_pen(100, -50, 50, -10, 120, -50, 54)

		self.pen_draw(90, -12, 0, 30)
		self.ear_pen(100, -50, 50, -10, 120, -50, 56)
	# 耳朵画笔
	def ear_pen(self, a, b, c, d, e, f, g):
		begin_fill()
		seth(a)
		circle(b, c)
		circle(d, e)
		circle(f, g)
		end_fill()
	# 眼睛
	def eye(self):
		self.eye_draw((255, 155, 192), "white", 90, -20, 0, -95, 15)
		self.eye_draw((0, 0, 0), "black", 90, 12, 0, -3, 3)
		self.eye_draw((255, 155, 192), "white", 90, -25, 0, 40, 15)
		self.eye_draw((0, 0, 0), 'black', 90, 12, 0, -3, 3)

	# 眼睛画笔
	def eye_draw(self, color_1, color_2, a, b, c, d, e):
		color(color_1, color_2)
		self.pen_draw(a, b, c, d)
		begin_fill()
		circle(e)
		end_fill()
	# 脸颊
	def cheek(self):
		color((255, 155, 192))
		self.pen_draw(90, -95, 0, 65)
		begin_fill()
		circle(30)
		end_fill()
	# 嘴巴
	def mouse(self):
		color(239, 69, 19)
		self.pen_draw(90, 15, 0, -100)
		seth(-80)
		circle(30, 40)
		circle(40, 80)

	# 身体
	def body(self):
		color("red", (255, 99, 71))
		self.body_draw(90, -20, 0, -78)
		begin_fill()
		self.circle_draw(-130, 100, 10, 300, 30)
		seth(0)
		fd(230)
		seth(90)
		circle(300, 30)
		circle(100, 3)
		color((255, 155, 192), (255, 100, 100))
		self.circle_draw(-135, -80, 63, -150, 24)
		end_fill()
	# 身体画笔
	def body_draw(self, a, b, c, d):
		pu()
		seth(a)
		fd(b)
		seth(c)
		fd(d)
		pd()


	# 圆画笔
	def circle_draw(self, x, a, b, c, d):
		seth(x)
		circle(a, b)
		circle(c, d)

	# 手
	def hand(self):
		color((255, 155, 192))
		self.body_draw(90, -40, 0, -27)
		self.hand_draw(-160, 300, 15)

		self.body_draw(90, 15, 0, 0)
		self.hand_draw(-10, -20, 90)

		self.body_draw(90, 30, 0, 237)
		self.hand_draw(-20, -300, 15)

		self.body_draw(90, 20, 0, 0)
		self.hand_draw(-170, 20, 90)

	手画笔
	def hand_draw(self, x, y, z):
		seth(x)
		circle(y, z)
	# 脚
	def foot(self):
		self.foot_draw(10, (240, 128, 128), 90, -75, 0, -180, -90, 40,
						-180, "black", 15, 20)
		self.foot_draw(10, (240, 128, 128), 90, 40, 0, 90, -90,
						40, -180, "black", 15, 20)
	# 脚画笔
	def foot_draw(self, a, b, c, d, e, f, g,
					h, i, j, k, l):
		pensize(a)
		color(b)
		self.pen_draw(c, d, e, f)
		seth(g)
		fd(h)
		seth(i)
		color(j)
		pensize(k)
		fd(l)

	# 尾巴
	def tail(self):
		pensize(4)
		color((255, 155, 192))
		self.pen_draw(90, 70, 0, 95)
		self.circle_draw(0, 70, 20, 10, 330)
		circle(70, 30)
		color("red")

	# 绘制图像主函数
	def pig_draw(self):
		self.nose()
		self.head()
		self.ear()
		self.eye()
		self.cheek()
		self.mouse()
		self.body()
		self.hand()
		self.foot()
		self.tail()

if __name__ == "__main__":

	# 画笔宽度
	pensize(4)
	colormode(255)
	# 画笔颜色及填充色
	color((255, 155, 192), "pink")
	# 画布尺寸
	setup(650, 500)
	# 画笔移动速度
	speed(4)
	time.sleep(5)
	init_draw = PeppaPig()
	init_draw.pig_draw()
	write('幸福', font=(15))
	mainloop()



