#!/bin/python

import numpy as np
import sys, getopt

try:
  opts, args = getopt.getopt(sys.argv[1:],"hs:e:n:")
except getopt.GetoptError:
  print '{} -s <start_color> -e <end_color> -n <num_elements>'.format(sys.argv[0])
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
      print '{} -s <start_color> -e <end_color> -n <num_elements>'.format(sys.argv[0])
      sys.exit()
  elif opt in ("-s"):
      color_start = np.fromstring(arg, dtype=float, sep=',')
  elif opt in ("-e"):
      color_end = np.fromstring(arg, dtype=float, sep=',')
  elif opt in ("-n"):
      no_steps = int(arg)
      
print color_start
print color_end
print no_steps

red_space = np.linspace(color_start[0], color_end[0], no_steps)
green_space = np.linspace(color_start[1], color_end[1], no_steps)
blue_space = np.linspace(color_start[2], color_end[2], no_steps)

f = open('gradient.txt', 'w')
for i in range(no_steps):
  f.write('<type_{}>\n  <rgba>\n    <r> {} </r>\n    <g> {} </g>\n    <b> {} </b>\n  </rgba>\n</type_{}>\n'.format(i+10, red_space[i], green_space[i], blue_space[i], i+10))

print 'written gradient to file...\n'
f.close()