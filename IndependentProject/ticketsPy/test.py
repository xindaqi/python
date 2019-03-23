import datetime
import re
from stationsInfo import stationLists
info = {
	'start_station':'',
	'end_station':'',
	'from_date':''
}



def inputArgs(start_station, end_station, d):
	# start_station = input("Input start station:\n")
	# end_station = input("Input end station: \n")
	# d = input("Input leave data:year-month-day: \n")
	now_time = datetime.datetime.now()
	flag1 = False
	flag2 = False
	flag3 = False

	while flag1 == False or flag2 == False or flag3 == False:
		start_index = stationLists.count(start_station)
		end_index = stationLists.count(end_station)

		if start_index >0 and end_station != start_station:
			flag1 = True
		if end_index > 0 and end_station != start_station:
			flag2 = True
		rdate = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', d) 
		if rdate:
			start_date = datetime.datetime.strptime(d, '%Y-%m-%d')
			sub_day = (start_date - now_time).days
			if -1 <= sub_day <15:
				flag3 = True
		if not flag1:
			print("start station is illegal!")
			start_station = input("Input start station:\n")

		if not flag2:
			print("end station is illegal!")
			end_station = input("Input end station: \n")

		if not flag3:
			print("leave date is illegal!")
			d = input("Input leave date: \n")
			start_date = datetime.datetime.strptime(d, '%Y-%m-%d')
			sub_day = (start_date - now_time).days
	info['start_station'] = start_station
	info['end_station'] = end_station
	info['from_date'] = d
	return info

if __name__ == "__main__":
	isContionue = 'Y'
	while isContionue == 'Y' or isContionue == 'y':
		start_station = input('Input start station: \n')
		end_station = input('Input end station: \n')
		from_date = input('Input leave date(foramt is year-month-day): \n')
		info = inputArgs(start_station, end_station, from_date)
		print(info)
		isContionue = input('continue or not Y/N\n')
	input('enter any key to quit')
