from flask import Flask, Blueprint, render_template, request 
from flask_login import (LoginManager, login_required, login_user,
                        logout_user, UserMixin)
from app.models.databases import User

app = Flask(__name__)

app.secret_key = "123456"
# app.config["SECRET_KEY"] = "123456"
login_manager = LoginManager()
login_manager.session_protection = "strong"
login_manager.login_view = "au.login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    print("id: {}".format(id))
    user = User()
    return user

auth = Blueprint("au", __name__, url_prefix="/api")

@auth.route("/login", methods=["GET", "POST"])
def login():
    # return render_template("login.html")
    user = User()
    username = request.form.get("username")
    password = request.form.get("password")
    admin = "admin"
    admin_pwd = "admin"
    if username == admin and password == admin_pwd:
        print("username: {}\npassword:{}".format(username, password))
        login_user(user)
        print("login value: {}".format(login_user(user)))
        return "login success"
    else:
        return render_template("login.html")

@auth.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    logout_user()
    return "logout"

@app.route("/test", methods=["GET", "POST"])
@login_required
def test():
    return "authorized"

app.register_blueprint(auth)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090, debug=True)
