#!/bin/python

import numpy as np
import sys, getopt

def printhelp():
  print '{} -n num_elements '.format(sys.argv[0])


try:
  opts, args = getopt.getopt(sys.argv[1:],"hn:")
except getopt.GetoptError:
  printhelp()
  sys.exit(2)

if len(opts) < 1:
  printhelp()
  sys.exit()
for opt, arg in opts:
  if opt == '-h':
      printhelp()
      sys.exit()
  elif opt in ("-n"):
      no_steps = int(arg)

print no_steps

f = open('random_colors.txt', 'w')

for i in range(no_steps):
    #We start with ID 4, as this is SWEPT_VOLUME_START
    type_id = 4+i
    f.write('<type_{}>\n  <rgba>\n    <r> {} </r>\n    <g> {} </g>\n    <b> {} </b>\n  </rgba>\n</type_{}>\n'.format(type_id, np.random.random(), np.random.random(), np.random.random(), type_id))

print 'written random_colors.txt to file...\n'
f.close()
