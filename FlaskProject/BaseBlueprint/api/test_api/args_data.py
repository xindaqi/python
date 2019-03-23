from flask_restplus import reqparse

'''add args data'''
args_data = reqparse.RequestParser()
args_data.add_argument('user_name', type=str, required=True, default='xindaqi', help='管理员电话姓名')
args_data.add_argument('user_phone', type=str, required=True, default='13691830510', help='管理员电话')


