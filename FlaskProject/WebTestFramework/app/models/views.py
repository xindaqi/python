from app import app
from .user import user
from .robot import robot
from .project import project
from .login import log

app.register_blueprint(user)
app.register_blueprint(robot)
app.register_blueprint(project)
app.register_blueprint(log)