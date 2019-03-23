"""Train tickets query via command-line.
Usage:
	getData.py <from> <to> <date>

Options:
	-h,--help   show help menu
	from        start station
	to          terminal
	date        departure time
	
Example:

	filename 北京 上海 2018-11-29

"""
# above all's function is you can take parameters while input python/python3 getData.py 
# like python/python3 getData.py 北京 上海 2018-11-29

import requests, json
from stationsInfo import *
from prettytable import PrettyTable 
from colorama import init, Fore, Back, Style
from docopt import docopt

# define table color
init(autoreset=False)
class Colored(object):
	def red(self, s):
		return Fore.LIGHTRED_EX + s + Fore.RESET 
	def green(self, s):
		return Fore.LIGHTGREEN_EX + s + Fore.RESET 

	def yellow(self, s):
		return Fore.LIGHTYELLOW_EX + s + Fore.RESET 

	def white(self, s):
		return Fore.LIGHTWHITE_EX + s + Fore.RESET

	def blue(self, s):
		return Fore.LIGHTBLUE_EX + s + Fore.RESET 

# from url get info 
def getData(url):
	data = ''
	headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36'}
	res = requests.get(url, headers=headers)

	# # direction response 
	# # <class 'requests.models.Response'>
	# print(type(res))
	# print("Response: {}".format(res))


	# # text response 
	# # <class 'str'>
	# print("Type text content: {}".format(type(res.text)))
	# print("Text content: {}".format(res.text))

	# json response 
	# <class 'dict'>
	# res = res.json()["data"]["result"][0]
	# print("Json type: {}".format(type(res)))
	# print("Json content: {}".format(res))

	# # content response
	# # <class 'bytes'>
	# print("Content type: {}".format(type(res.content)))
	# print("Content: {}".format(res.content))
	return res

# get train's info and parse parameters
def processResponse(from_station, to_station, from_date):

	# get train info url, train_data is train's date
	# from_station is departure station
	# to_station is arrival station
	# stations2CODE is trans your input chinese into stations' abbreviated letters
	url = "https://kyfw.12306.cn/otn/leftTicket/query?leftTicketDTO.train_date={}&leftTicketDTO.from_station={}&leftTicketDTO.to_station={}&purpose_codes=ADULT".format(from_date, stations2CODE[from_station], stations2CODE[to_station])
	trainInfos = getData(url)
	trainInfos = trainInfos.json()["data"]["result"]
	trainInfosOptions = [ 'trainNumber', 'departureStation', 'terminal', 'departureTime', 'arrivalTime', 'duration', 'bussinessBlock', 
					'firstClass', 'secondClass', 'highgradeSoftBerth', 'softBerth',
					'motorBerth', 'hardBerth', 'softSeat', 'hardSeat', 'noneSeat','otherInfo', 'trainStatus']
	trainInfosTemp = []
	
	color = Colored()
	for trainInfo in trainInfos:
		data = {'trainNumber':'', 'departureStation':'', 'terminal':'', 'departureTime':'', 'arrivalTime':'', 'duration':'', 'bussinessBlock':'', 
					'firstClass':'' , 'secondClass':'' , 'highgradeSoftBerth':'' , 'softBerth':'' ,
					'motorBerth':'', 'hardBerth':'', 'softSeat':'', 'hardSeat':'', 'noneSeat':'', 'otherInfo':'', 'trainStatus':''}
		trainInfo = trainInfo.split('|')
		# print(trainInfo)
		
		data['trainNumber'] = trainInfo[3]
		data['departureStation'] = trainInfo[6]
		data['terminal'] = trainInfo[7]
		data['departureTime'] = trainInfo[8]
		data['arrivalTime'] = trainInfo[9]
		data['duration'] = trainInfo[10]
		data['bussinessBlock'] = trainInfo[32] or trainInfo[25]
		data['firstClass'] = trainInfo[31]
		data['secondClass'] = trainInfo[30]
		data['highgradeSoftBerth'] = trainInfo[21]
		data['softBerth'] = trainInfo[23]
		data['motorBerth'] = trainInfo[27]
		data['hardBerth'] = trainInfo[28]
		data['softSeat'] = trainInfo[24]
		data['hardSeat'] = trainInfo[29]
		data['noneSeat'] = trainInfo[26]
		data['otherInfo'] = trainInfo[22]
		data['trainStatus'] = trainInfo[1]

		for info in trainInfosOptions:
			if data[info] == '':
				data[info] = '-'
		trainInfosTemp.append(data)

	# info coloring by different color
	trainInfosShow = []
	for tp in trainInfosTemp:
		temps = []
		for op in trainInfosOptions:
			if op == "departureStation":
				s = color.green(stations2CN[tp[op]]) + '\n' + color.red(stations2CN[tp['terminal']])
				temps.append(s)
			elif op == 'departureTime':
				s = color.green(tp[op]) + '\n' + color.red(tp['arrivalTime'])
				temps.append(s)
			elif op == 'trainNumber':
				s = color.yellow(tp[op])
				temps.append(s)
			else:
				temps.append(tp[op])
		trainInfosShow.append(temps)

	for trainInfo in trainInfosShow:
		trainInfo.pop(2)
		trainInfo.pop(3)
	return trainInfosShow

# draw info table, show train's info details
def display(tickets):
	rowTable = PrettyTable('车次 出发/到达站 出发/到达时间 历时 商务座 一等座 二等座 高级软卧 软卧 动卧 硬卧 软座 硬座 无座 其他 备注'.split(' '))
	for ticket in tickets:
		rowTable.add_row(ticket)

	print(rowTable)



# take parameters while run and show results
def cli():
	arguments = docopt(__doc__)
	print(arguments)
	tickets = processResponse(arguments['<from>'], arguments['<to>'], arguments['<date>'])
	display(tickets)


if __name__ == "__main__":
	# tickets = processResponse()
	# # for ticket in tickets:
	# # 	print(ticket)
	# display(tickets)
	cli()
	input("Enter any key for quit...")


