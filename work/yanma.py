import IPy
import xlrd,xlwt

# 打开文件
workbook = xlrd.open_workbook(r'F:\demo.xlsx')
# 获取所有sheet
print(workbook.sheet_names()) # [u'sheet1', u'sheet2']
sheet2_name = workbook.sheet_names()[1]