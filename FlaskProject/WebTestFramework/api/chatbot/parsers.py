from flask_restplus import reqparse, fields
from api.restplus import api

getTokenParams = api.model('获取认证',{
  'access_token':fields.String(required=True, description='认证token'),
  'group_id':fields.String(required=True, description='用户组')
  })

# 1--获取机器人access_token
pagination_arguments_token = reqparse.RequestParser()
pagination_arguments_token.add_argument('appId', type=str, required=True, default=1, help='固定的企业/个人ID')
pagination_arguments_token.add_argument('createTime', type=int, required=False, default=1, help='自1970年1月1日0时起至今的毫秒数(可随意填写,小于至今的秒数即可)')
pagination_arguments_token.add_argument('sign', type=str, required=True,
                                  help='签名(appId,appKey,createTime以字符串方式拼接后经过 MD5 加密)')
pagination_arguments_token.add_argument('expire', type=int, required=False, choices=[4,8,12,16,20,24], default=24,
                                           help='token 过期时间,单位小时(可传入大于 0,小于 25 的整数)')
# 2--机器人单轮对话
pagination_arguments_dialog = reqparse.RequestParser()
pagination_arguments_dialog.add_argument('access_token', type=str, required=True, default=1, help='调用接口凭证')
pagination_arguments_dialog.add_argument('sysNum', type=str, required=True, default=1, help='企业或账号标识')
pagination_arguments_dialog.add_argument('question', type=str, required=True, default='Hello', help='用户问题')
pagination_arguments_dialog.add_argument('robotFlag', type=int, required=True, choices=[1,2,3],
                                  default=1, help='机器人编号')


# 3--添加知识库对话
pagination_arguments_addDialog = reqparse.RequestParser()
pagination_arguments_addDialog.add_argument('access_token', type=str, required=True, default=1, help='调用接口凭证')
pagination_arguments_addDialog.add_argument('robotFlag', type=int, required=True, choices=[1,2,3],
                                  default=1, help='机器人编号')
pagination_arguments_addDialog.add_argument('questionTitle', type=str, required=True, default='Hi', help='标准问题')
pagination_arguments_addDialog.add_argument('questionTypeId', type=str, required=True, default='', help='词条分类')
pagination_arguments_addDialog.add_argument('answerDesc', type=str, required=True, default='', help='答案内容')
pagination_arguments_addDialog.add_argument('usedFlag', type=int, required=True, default='1',choices=[0,1], help='词条状态,0 开启,1 停用')
pagination_arguments_addDialog.add_argument('matchFlag', type=int, required=True, choices=[0,1,2],
                                  default=2, help='匹配模式,0 智能匹配,1 完全匹是配,2 包含匹配')
pagination_arguments_addDialog.add_argument('auditStatus', type=str, required=True, default='1', choices=[1,2], help='有效状态,1 永久有效,2 指定时间有效')

# 4--删除知识库对话
pagination_arguments_deleteDialog = reqparse.RequestParser()
pagination_arguments_deleteDialog.add_argument('access_token', type=str, required=True, default=1, help='调用接口凭证')
pagination_arguments_deleteDialog.add_argument('robotFlag', type=int, required=True, choices=[1,2,3],
                                  default=1, help='机器人编号')
pagination_arguments_deleteDialog.add_argument('docId', type=str, required=True, default=1, help='机器人编号')

# 5--修改知识库对话
pagination_arguments_editDialog = reqparse.RequestParser()
pagination_arguments_editDialog.add_argument('access_token', type=str, required=True, default=1, help='调用接口凭证')
pagination_arguments_editDialog.add_argument('docId', type=str, required=True, default=1, help='对话ID')
pagination_arguments_editDialog.add_argument('robotFlag', type=int, required=True, choices=[1,2,3],
                                  default=1, help='机器人编号')
pagination_arguments_editDialog.add_argument('questionId', type=str, required=False, help='问题ID')
pagination_arguments_editDialog.add_argument('questionTitle', type=str, required=True, default='Hi', help='标准问题')
pagination_arguments_editDialog.add_argument('questionTypeId', type=str, required=True, default='', help='词条分类')
pagination_arguments_editDialog.add_argument('matchFlag', type=int, required=True, choices=[0,1,2],
                                  default=2, help='匹配模式,0 智能匹配,1 完全匹是配,2 包含匹配')
pagination_arguments_editDialog.add_argument('answerId', type=str, required=False, help='答案ID')
pagination_arguments_editDialog.add_argument('answerDesc', type=str, required=True, help='答案内容')
pagination_arguments_editDialog.add_argument('usedFlag', type=int, required=True, default='1',choices=[0,1], help='词条状态,0 开启,1 停用')
pagination_arguments_editDialog.add_argument('auditStatus', type=str, required=True, default='1', choices=[1,2], help='有效状态,1 永久有效,2 指定时间有效') 

# 6--查询知识库对话
pagination_arguments_dialogQuery = reqparse.RequestParser()
pagination_arguments_dialogQuery.add_argument('access_token', type=str, required=True, default=1, help='调用接口凭证')
pagination_arguments_dialogQuery.add_argument('docId', type=str, required=True, default=1, help='对话ID')
pagination_arguments_dialogQuery.add_argument('robotFlag', type=int, required=True, choices=[1,2,3],
                                  default=1, help='机器人编号')

# 7--查询知识库分类
pagination_arguments_questionList = reqparse.RequestParser()
pagination_arguments_questionList.add_argument('access_token', type=str, required=True, default=1, help='调用接口凭证')
pagination_arguments_questionList.add_argument('typeFlag', type=str, required=True, default=1, choices=[1,2], help='分类类型:1-单轮问题分类(默认);2-多轮问题分类;')
pagination_arguments_questionList.add_argument('parentId', type=str, required=True, default='', help='父级分类id,顶级分类id值为-1')


# 8--查询知识库对话列表
pagination_arguments_dialogList = reqparse.RequestParser()
pagination_arguments_dialogList.add_argument('access_token', type=str, required=True, default=1, help='调用接口凭证')
pagination_arguments_dialogList.add_argument('questionTypeId', type=str, required=True, default='', help='词条分类')
pagination_arguments_dialogList.add_argument('robotFlag', type=int, required=True, choices=[1,2,3],
                                  default=1, help='机器人编号')
pagination_arguments_dialogList.add_argument('pageNo',type=int,required=True,default=1, help='当前页码')
pagination_arguments_dialogList.add_argument('keyFlag',type=int,required=True,default=1,choices=[1,2], help='1表示问题,2表示答案')