"""Train tickets query via command-line.
Usage:
	docoptTest.py <from> <to> <date>

Options:
	-h,--help   show help menu
	from        start station
	to          terminal
	date        departure time
	
Example:

	filename 北京 上海 2018-11-29

"""


from docopt import docopt

def cli():
	arguments = docopt(__doc__)
	print(arguments)


if __name__ == "__main__":
	cli()