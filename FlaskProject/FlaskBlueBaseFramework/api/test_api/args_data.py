from flask_restplus import reqparse

'''add get request args'''
test_add_get_arg = reqparse.RequestParser()
test_add_get_arg.add_argument('user_name', type=str, required=True, default='xindaqi', help='管理员电话姓名')
test_add_get_arg.add_argument('user_phone', type=str, required=True, default='13691830510', help='管理员电话')


# '''add post request args'''
# test_add_post_arg = reqparse.RequestParser()
# test_add_post_arg.add_argument('user_addr', type=str, required=True, default='shenzhen', help="come on")
# test_add_post_arg.add_argument('user_sex', type=str, required=True, default='male', help='enjoy')





