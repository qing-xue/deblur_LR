import sys

class Logger(object):
	"""
	sys.stdout = Logger('a.log', sys.stdout)
	sys.stderr = Logger('a.log_file', sys.stderr)		# redirect std err, if necessary

	# now it works
	# print('print something')
	"""
	
	def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')
		
	def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)
		
	def flush(self):
	    pass
