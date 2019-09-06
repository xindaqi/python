from flask_restplus import fields
from api.restplus import api

# 管理员测试
admin_test_jsons = api.model('管理员参数', {
    'admin_name':fields.String(required=True, description="管理员姓名"),
    'admin_phone':fields.String(required=True, description="管理员电话")
})

# 普通用户信息
user_info_jsons = api.model('用户参数', {
    'u_id':fields.String(required=True, description="用户ID"),
    'u_name':fields.String(required=True, description="用户姓名"),
    'u_phone':fields.String(required=True, description="用户电话")
})