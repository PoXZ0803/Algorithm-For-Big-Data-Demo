
#Mapper.py
import os
import sys

def mapper():
    #获取当前正在处理的文件的名字。由于有两个输入文件，所以要加以区分。
	filepath = os.environ["map_input_file"]
	filename = os.path.split(filepath)[-1]
	for line in sys.stdin:
		if line.strip()=="":
			continue
		fields = line[:-1].split("\t")
		sno = fields[0]
		#以下推断filename的目的是不同的文件有不同的字段。而且需加上不同的标记
		if filename == 'data_info':
			name = fields[1]
			# TODO: 为数据示例1加上'0'的统一标记
            #__________________________________#
		elif filename == 'data_grade':
			courseno = fields[1]
			grade = fields[2]
			# TODO: 为数据示例2加上'1'的统一标记
			#__________________________________#
 
if __name__=='__main__':
	mapper()


#Reducer.py
import sys
def reducer():
	#为了记录和上一个记录的区别，用lastsno记录上个sno
	lastsno = ""
	for line in sys.stdin:
		if line.strip() == "":
			continue
		fields = line[:-1].split("\t")
		sno = fields[0]
		'''
		处理思路：
		遇见当前key与上一条key不同并且label=0，就记录下来name值，
		当前key与上一条key相同并且label==1，则将本条数据的courseno、
		grade联通上一条记录的name一起输出成最终结果
		'''
		if sno != lastsno:
			name = ""
            # TODO: 判断label==0的情况
			#__________________________________#
		elif sno==lastsno:
			# TODO: 判断label==1的情况
			#__________________________________#
				if name:
					print('\t'.join((lastsno, name, courseno, grade)))
		lastsno = sno

if __name__=='__main__':
	reducer()