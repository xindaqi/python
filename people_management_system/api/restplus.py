from flask_restplus import Api, apidoc
# from flask.ext.restplus import Api


authorizations = {
	'schemes':['https', 'http', 'ws', 'wss'],
	'apikey':{
	'type':'apiKey',
	'in':'header',
	'name':'X-API-KEY',
	'schemes':'http',
	},
	'oauth2':{
	'type':'oauth2',
	'flow':'accessCode',
	'tokenUrl':'https://.com/token',
	'scopes':{
	'read':'Grant read-only access',
	'write':'Grant read-write access'
	}
	}
}
api = Api(version='1.0', title='XH Face Recogintion API', description='XHWL Connect The Wonderful Life' 
	      , authorizations=authorizations)
	      # authorizations=authorizations 
	      # description换行显示 \n Contact me: \n E-mail: xdq101@qq.com \n Phone:13691830510
@api.documentation
def custom_ui():
	return apidoc.ui_for(api)