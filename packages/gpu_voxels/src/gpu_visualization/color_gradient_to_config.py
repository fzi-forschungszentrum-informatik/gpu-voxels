#!/bin/python

import numpy as np
import sys, getopt

def printhelp():
  print '{} -s start_color -e end_color -n num_elements -r repeat_steps (Colors are given as r,g,b where components are given in [0.0 ... 1.0] range.'.format(sys.argv[0])


try:
  opts, args = getopt.getopt(sys.argv[1:],"hs:e:n:r:")
except getopt.GetoptError:
  printhelp()
  sys.exit(2)

if len(opts) < 3:
  printhelp()
  sys.exit()
for opt, arg in opts:
  if opt == '-h':
      printhelp()
      sys.exit()
  elif opt in ("-s"):
      color_start = np.fromstring(arg, dtype=float, sep=',')
  elif opt in ("-e"):
      color_end = np.fromstring(arg, dtype=float, sep=',')
  elif opt in ("-n"):
      no_steps = int(arg)
  elif opt in ("-r"):
      repeat_steps = int(arg)
      
print color_start
print color_end
print no_steps

red_space = np.linspace(color_start[0], color_end[0], no_steps)
green_space = np.linspace(color_start[1], color_end[1], no_steps)
blue_space = np.linspace(color_start[2], color_end[2], no_steps)

f = open('gradient.txt', 'w')

for r in range(repeat_steps):
    for i in range(no_steps):
        #We start with ID 10, as this is SWEPT_VOLUME_START
        type_id = 10+(r*no_steps)+i
        f.write('<type_{}>\n  <rgba>\n    <r> {} </r>\n    <g> {} </g>\n    <b> {} </b>\n  </rgba>\n</type_{}>\n'.format(type_id, red_space[i], green_space[i], blue_space[i], type_id))

print 'written gradient to file...\n'
f.close()
