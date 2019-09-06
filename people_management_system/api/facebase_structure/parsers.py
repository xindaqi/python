from flask_restplus import reqparse

admin_test_parsers = reqparse.RequestParser()
admin_test_parsers.add_argument("admin_name", type=str, required=True, default="admin", help="管理员姓名")
admin_test_parsers.add_argument("admin_phone", type=str, required=True, default="123456", help="管理员电话")

user_info_parsers = reqparse.RequestParser()
user_info_parsers.add_argument("u_id", type=str, required=True, default="001", help="用户ID")
user_info_parsers.add_argument("u_name", type=str, required=True, default="小小", help="用户姓名")
user_info_parsers.add_argument("u_phone", type=str, required=False, default="13889231593", help="用户电话")
