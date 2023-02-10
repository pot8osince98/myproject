# myportfolio/__init__.py

from flask import Flask

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"

from myportfolio.core.views import core
from myportfolio.projects.views import projects

app.register_blueprint(core)
app.register_blueprint(projects,url_prefix='/projects')