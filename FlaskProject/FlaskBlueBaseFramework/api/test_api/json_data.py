from flask_restplus import fields
from api.restplus import api

# '''test get arguments'''
# test_get_arg = api.model('管理员参数', {
# 		'user_name':fields.String(required=True, description='管理员姓名'),
# 		'user_phone':fields.String(required=True, description='管理员电话')
# 	})

'''test post arg'''
test_post_arg = api.model('post args', {
		'user_addr':fields.String(required=True, default="shenzhen", description='user address'),
		'user_sex':fields.String(required=True, default="male", description='user sex')
	})


