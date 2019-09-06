from argparse import ArgumentParser
import xlrd, xlwt
import pandas as pd

def get_data_path():
    path_default = "/home/xdq/xinPrj/datasets/rw_test.xlsx"
    parser = ArgumentParser()
    parser.add_argument("--path", default=path_default)
    args = parser.parse_args()
    path = args.path
    return path

def read_excel_data():
    path = get_data_path()
    print("默认文件路径: {}".format(path))
    print("使用其路径可使用:python rw_excel.py --path 绝对路径")
    # 打开文档
    workbook = xlrd.open_workbook(path)
    # 表格名称,排序为文档表格自左向右
    sheet_names = workbook.sheet_names()
    # 表格名称: ['人员信息', '项目信息']
    print("表格名称: {}".format(sheet_names))
    # 表格对象,类型为list,遍历取对应表格的数据
    sheets = workbook.sheets()
    # 读取表格:人员信息sheets[0]
    user_info = sheets[0]
    # 表格名称
    user_info_name = user_info.name 
    print("user infomation table name: {}".format(user_info_name))
    # 人员信息表格:行数
    user_info_row_nums = user_info.nrows
    # 人员信息表格:列数
    user_info_col_nums = user_info.ncols
    print("user infomation table row number: {}, column nubmer: {}".format(user_info_row_nums, user_info_col_nums))
    # 表格整行内容
    row_num = 0
    user_info_rows = user_info.row_values(row_num)
    # first row contents: ['编号', '姓名', '项目']
    print("first row contents: {}".format(user_info_rows))
    # 表格整列内容
    col_num = 0
    user_info_cols = user_info.col_values(col_num)
    # first column contents: ['编号', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
    # '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    print("first column contents: {}".format(user_info_cols))
    # 获取单元格内容A1
    cell_A1 = user_info.cell(row_num,col_num).value
    print("A1 cell data: {}".format(cell_A1))
    # 获取整行内容
    one_row = user_info.row(row_num)
    # one row data: [text:'编号', text:'姓名', text:'项目']
    print("one row data: {}".format(one_row))
    # 获取行某列值
    one_row_value = user_info.row(row_num)[col_num].value 
    # one row value in  certain column: 编号
    print("one row value in  certain column: {}".format(one_row_value))
    # 获取整列内容
    one_col = user_info.col(col_num)
    # one column data: [text:'编号', text:'1', text:'2', text:'3', text:'4', 
    # text:'5', text:'6', text:'7', text:'8',
    # text:'9', text:'10', text:'11', text:'12', text:'13', text:'14', 
    # text:'15', text:'16', text:'17', text:'18', text:'19', text:'20']
    print("one column data: {}".format(one_col))
    # 获取整列内容的某一行数据
    one_col_value = user_info.col(col_num)[row_num].value
    # one column value in certain row: 编号
    print("one column value in certain row: {}".format(one_col_value))
    # 数据类型
    data_type = user_info.row_types(row_num)
    # data type: array('B', [1, 1, 1])
    print("data type: {}".format(data_type))
    
def write_data_to_excel_rewrite():
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("写入数据测试", cell_overwrite_ok=True)
    row_num = 0
    col_num = 0
    try:
        sheet.write(row_num, col_num, 25)
        sheet.write(row_num, col_num, 250) 
        workbook.save("rw_test.xlsx")
        print("process status: saved successfully!")
    except Exception:
        print("Permission Denied for rewrite data")

def write_data_to_excel_write_once():
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("写入数据测试", cell_overwrite_ok=False)
    row_num = 0
    col_num = 0
    try:
        sheet.write(row_num, col_num, 25)
        sheet.write(row_num, col_num, 250) 
        workbook.save("rw_test.xlsx")
        print("process status: saved successfully!")
    except Exception:
        print("Permission Denied for rewrite data")    

def get_csv_path():
    parser = ArgumentParser()
    parser.add_argument("--path", default="/home/xdq/xinPrj/datasets/pandas_rw.csv")
    args = parser.parse_args()
    path = args.path
    return path


def pandas_read_csv():
    path = get_csv_path()
    print("CSV文件默认路径: {}".format(path))
    print("使用其他路径,格式为: python rw_excel.py --path 绝对路径")
    # 读取全部数据
    datas = pd.read_csv(path)
    #    编号   姓名
    # 0   1   小1
    # 1   2   小2
    # 2   3   小3
    # 3   4   小4
    # 4   5   小5
    # 5   6   小6
    # 6   7   小7
    # 7   8   小8
    # 8   9   小9
    # 9  10  小10
    print("Total datas: \n{}".format(datas))
        



if __name__ == "__main__":
    # read_excel_data()
    # write_data_to_excel_rewrite()
    pandas_read_csv()