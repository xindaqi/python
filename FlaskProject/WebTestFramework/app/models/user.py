from flask import Blueprint,render_template,redirect,request,flash, url_for
from app import db, app
# 引入表
from .databases import User
from flask_login import login_required,LoginManager,login_user, UserMixin



# user蓝图
user = Blueprint('user', __name__, url_prefix='/user')
login_manager = LoginManager()
login_manager.init_app(app)
# 登录视图
login_manager.login_view = "user.login"
login_manager.login_message = "请先登录以访问页面"

password = "123456"
@login_manager.user_loader
def user_loader(id):
	user = User.query.filter_by(id=id).first()
	return user
	print(user)
# 注册路由
@user.route('/login/', methods=['GET', 'POST'])
# @login_required
def login():
	if request.method == 'POST':
		name = request.form.get('username',None)
		print(name)
		user = User.query.filter_by(username=name).first()
		print(user)
		if not user:
			flash('该用户不存在')
			# return('该用户不存在')
		elif request.form.get('password') != user.password:
			flash('密码错误')
			# return('密码错误')
		else:
			login_user(user, remember=True)
			next_url = request.args.get('next')
			# return redirect(next_url or url_for('login_success'))
			users = User.query.all()
			return render_template('user/show.html',users=users)
	return render_template('user/login.html')

# @user.route('/login/',methods=['GET','POST'])
# def login_in():
# 	if request.method == 'POST':
# 		u_name = request.form.get('username',None)
# 		u_pwd = request.form.get('password',None)
# 		print(u_name,u_pwd)
# 		name = User.query.filter_by(username=u_name).all()
# 		pwd = User.query.filter_by(password=u_pwd).all()
# 		print(name,pwd)
# 		if not name or not pwd:
# 			return 'name or password error,try again!'

# 		return render_template('user/show.html')

# 添加人员信息

@user.route('/add/', methods=['GET','POST'])
@login_required
def add():
	if request.method == 'POST':
		username = request.form.get('username',None)
		email = request.form.get('email',None)
		pwd = request.form.get('password',None)
		print(username)

		if not username or not email or not pwd:
			return 'Input error, try again!'

		userInfo = User(username=username,email=email,password=pwd)
		db.session.add(userInfo)
		db.session.commit()
		users = User.query.all()
		return render_template('user/show.html',users=users)
	users = User.query.all()
	return render_template('user/add.html',users=users)



# 展示人员信息
@user.route('/show/',methods=['GET','POST'])
def show():
	users = User.query.all()
	return render_template('user/show.html',users=users)



