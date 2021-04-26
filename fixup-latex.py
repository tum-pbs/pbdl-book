import sys, os, re 
#import sys, os, psutil, subprocess, time, signal, re, json, logging, datetime, argparse
#import numpy as np

# fix jupyter book latex output

#filter_mem = re.compile(r".+\|\s+(\d+)MiB\s+/\s+(\d+)MiB\s+\|")#')
ft2 = re.compile(r"tst")
ft3 = re.compile(r"’")
fte = re.compile(r"👋")

# TODO , replace phi symbol w text in phiflow

path = "tmp2.txt" # simple
path = "tmp.txt" # utf8
#path = "book.tex-in.bak" # full utf8
outf = "tmpOut.txt"

with open(outf, 'w') as fout:
	with open(path, 'r') as f:
		c = 0
		for line in iter(f.readline, ''):
			line = re.sub('’', '\'', str(line))
			line = re.sub('[abz]', '.', str(line))

			t = ft3.search(str(line))
			if t is not None:
				print("H " + format(t) +"  "+ format(t.group(0)) )

			t = fte.search(str(line))
			if t is not None:
				print("E " + format(t) + format(t.group(0)) )

			fout.write(line)
			print(line[:-1])
			c = c+1

