from turtle import *
import time


def flower():
	color("red", "red")
	begin_fill()
	for _ in range(50):
		forward(200)
		left(170)
	end_fill()

	mainloop()


def triangle():
	for i in range(3):
		forward(200)
		right(120)

	mainloop()


class Doraemon(object):
	# 画笔移动,避免移动绘制轨迹
	# pu抬笔,goto移动,pd落笔
	def my_goto(self, x, y):
		penup()
		goto(x, y)
		pendown()
	# 转角,向前
	def seth_fd(self, a, b):
		seth(a)
		fd(b)
	# 转角,画圆
	def seth_circle(self, a, b, c):
		seth(a)
		circle(b, c)

	def circle_fd(self, a, b, c):
		circle(a, b)
		fd(c)

	# 眼睛	
	def eyes(self):
		tracer(False)
		a = 2.5
		for i in range(120):
			if 0 <= i < 30 or 60 <= i <90:
				a -= 0.05
				lt(3)
				fd(a)
			else:
				a += 0.05
				lt(3)
				fd(a)
		tracer(True)



	# 胡须位置
	def beard_draw(self, a, b, c, d):
		self.my_goto(a, b)
		self.seth_fd(c, d)


	# 胡须
	def beard(self):
		self.beard_draw(-37, 135, 165, 60)
		self.beard_draw(-37, 125, 180, 60)
		self.beard_draw(-37, 115, 193, 60)
		self.beard_draw(37, 135, 15, 60)
		self.beard_draw(37, 125, 0, 60)
		self.beard_draw(37, 115, -13, 60)

	# 嘴
	def mouth(self):
		self.beard_draw(5, 148, 270, 100)
		self.seth_circle(0, 120, 50)
		self.seth_circle(230, -120, 100)

	# 围巾
	def scarf(self):
		
		fillcolor('#e70010')
		begin_fill()
		self.seth_fd(0, 200)
		self.circle_fd(-5, 90, 10)
		self.circle_fd(-5, 90, 207)
		self.circle_fd(-5, 90, 10)
		circle(-5, 90)
		end_fill()

	def nose(self):
		self.my_goto(-10, 158)
		fillcolor('#e70010')
		begin_fill()
		circle(20)
		end_fill()

	def black_eyes(self):
		seth(0)
		self.my_goto(-20, 195)
		fillcolor('#000000')
		begin_fill()
		circle(13)
		end_fill()

		pensize(6)
		self.my_goto(20, 205)
		self.seth_circle(75, -10, 150)
		pensize(3)

		self.my_goto(-17, 200)
		seth(0)
		fillcolor('#ffffff')
		begin_fill()
		circle(5)
		end_fill()
		self.my_goto(0, 0)

	def face(self):
		fd(183)
		fillcolor('#ffffff')
		begin_fill()
		lt(45)
		circle(120, 100)

		seth(90)
		self.eyes()
		seth(180)
		penup()
		fd(60)
		pendown()

		seth(90)
		self.eyes()
		penup()
		seth(180)
		fd(64)
		pendown()
		seth(215)
		circle(120, 100)
		end_fill()
	def head(self):
		penup()
		circle(150, 40)
		pendown()
		fillcolor('#00a0de')
		begin_fill()
		circle(150, 280)
		end_fill()
	def haha(self):
		self.head()
		self.scarf()
		self.face()
		self.nose()
		self.mouth()
		self.beard()
		self.my_goto(0, 0)
		seth(0)
		penup()
		circle(150, 50)
		pendown()
		self.seth_fd(30, 40)
		self.seth_circle(70, -30, 270)

		fillcolor('#00a0de')
		begin_fill()

		self.seth_fd(230, 80)
		self.seth_circle(90, 1000, 1)
		self.seth_circle(-89, -1000, 10)
		self.seth_fd(180, 70)
		self.seth_circle(90, 30, 180)
		self.seth_fd(180, 70)
		self.seth_circle(100, -1000, 9)
		self.seth_circle(-86, 1000, 2)
		self.seth_fd(230, 40)
		
		circle(-30, 230)
		self.seth_fd(45, 81)
		self.seth_fd(0, 203)

		self.circle_fd(5, 90, 10)
		self.circle_fd(5, 90, 7)
		self.seth_circle(40, 150, 10)
		self.seth_fd(30, 40)
		end_fill()

		# left hand
		seth(70)
		fillcolor('#ffffff')
		begin_fill()
		circle(-30)
		end_fill()

		# foot
		self.my_goto(103.74, -182.59)
		seth(0)
		fillcolor('#ffffff')
		begin_fill()
		fd(15)
		self.circle_fd(-15, 180, 90)
		self.circle_fd(-15, 180, 10)
		end_fill()

		self.my_goto(-96.26, -182.59)
		seth(180)
		fillcolor('#ffffff')
		begin_fill()

		fd(15)
		circle(15, 180)
		fd(90)
		circle(15, 180)
		fd(10)
		end_fill()

		# right hand
		self.my_goto(-133.97, -91.81)
		seth(50)
		fillcolor('#ffffff')
		begin_fill()
		circle(30)
		end_fill()

		# bag
		self.my_goto(-103.42, 15.09)
		self.seth_fd(0, 38)
		seth(230)
		begin_fill()
		circle(90, 260)
		end_fill()

		self.my_goto(5, -40)
		self.seth_fd(0, 70)
		self.seth_circle(-90, -70, 180)
		self.seth_fd(0, 70)
		

		# ding
		self.my_goto(-103.42, 15.09)
		fd(90)
		seth(70)
		fillcolor('#ffd200')
		begin_fill()
		circle(-20)
		end_fill()
		seth(170)
		fillcolor('#ffd200')
		begin_fill()
		circle(-2, 180)
		seth(10)
		circle(-100, 22)
		circle(-2, 180)
		seth(170)
		circle(100, 22)
		end_fill()

		goto(-13.42, 15.09)
		seth(250)
		circle(20, 110)
		seth(90)
		fd(15)
		dot(10)
		self.my_goto(0, -150)
		self.black_eyes()





if __name__ == "__main__":
	# screensize(800, 600, '#f0f0f0')
	setup(650, 700)
	pensize(3)
	speed(9)
	time.sleep(4)

	d_e = Doraemon()
	d_e.haha()
	d_e.my_goto(200, -300)
	color((0, 0, 0))

	write("xindaqi", font=(15))
	mainloop()

	# triangle()
	# flower()

