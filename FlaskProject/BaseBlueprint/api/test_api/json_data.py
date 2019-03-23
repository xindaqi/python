from flask_restplus import fields
from api.restplus import api



'''json format data frame'''
json_data = api.model('json args', {
		'user_addr':fields.String(required=True, default="shenzhen", description='user address'),
		'user_sex':fields.String(required=True, default="male", description='user sex')
	})


