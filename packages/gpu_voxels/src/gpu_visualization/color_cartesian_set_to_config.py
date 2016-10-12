#!/bin/python

import numpy as np
import sys, getopt

def printhelp():
  print '{} -s start_color -e end_color -n num_elements -r repeat_steps\n  Colors are given as r,g,b where components are given in [0.0 ... 1.0] range. repeat_steps is given in red_steps,green_steps,blue_steps. -b is base id (swept volumes start at 4).'.format(sys.argv[0])


try:
    opts, args = getopt.getopt(sys.argv[1:],"hs:e:n:r:b:")
except getopt.GetoptError:
  printhelp()
  sys.exit(2)

#default values

#We start with ID 4, as this is SWEPT_VOLUME_START
base_id = 4

#parse arguments
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
      steps = np.fromstring(arg, dtype=int, sep=',')
  elif opt in ("-r"):
      repeat_steps = int(arg)
  elif opt in ("-b"):
      base_id = int(arg)
      
print color_start
print color_end
print steps

red_space = np.linspace(color_start[0], color_end[0], steps[0])
green_space = np.linspace(color_start[1], color_end[1], steps[1])
blue_space = np.linspace(color_start[2], color_end[2], steps[2])

file_name = 'colormap.txt'
f = open(file_name, 'w')

total_steps = steps[0] + steps[1] + steps[2]

for r in range(repeat_steps):
    for i_red in range(steps[0]):
        for i_green in range(steps[1]):
            for i_blue in range(steps[2]):
                type_id = base_id+(r*total_steps)+(i_red + (steps[0] * (i_green + (steps[1] * i_blue))))
                f.write('<type_{}>\n  <rgba>\n    <r> {} </r>\n    <g> {} </g>\n    <b> {} </b>\n  </rgba>\n</type_{}>\n'.format(type_id, red_space[i_red], green_space[i_green], blue_space[i_blue], type_id))

print('wrote colormap to file ' + file_name +'\n')
f.close()
