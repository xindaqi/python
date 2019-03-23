from flask import Flask, Blueprint, request, render_template, redirect, url_for

log = Blueprint('log', __name__, url_prefix='/user')

@log.route('/login', methods=['GET', 'POST'])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		login_user(login_user)
		flask.flash('Logged in successfully!')

		next = request.args.get('next')
		if not next_is_valid(next):
			return abort(400)

		return redirect(next or url_for('index'))
	return render_template('log/login.html', form=form)