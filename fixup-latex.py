import sys, os, re 

# fix jupyter book latex output

#filter_mem = re.compile(r".+\|\s+(\d+)MiB\s+/\s+(\d+)MiB\s+\|")#')
#ft2 = re.compile(r"tst")
#ft3 = re.compile(r"’")
#fte = re.compile(r"👋")

# TODO check, remove full "name": "stderr" {} block?  (grep stderr *ipynb) ???
# TODO , replace phi symbol w text in phiflow

inf  = "book-in.tex" 
outf = "book-in2.tex"
print("Start fixup latex, "+inf+" -> "+outf+" \n\n")

reSkip = [] ; reSCnt = []
reSkip.append( re.compile(r"catcode") ) ; reSCnt.append( 1 )
reSkip.append( re.compile(r"sepackage{fontspec}") ) ; reSCnt.append( 1 )
reSkip.append( re.compile(r"defaultfontfeatures") ) ; reSCnt.append( 1 )
reSkip.append( re.compile(r"polyglossia") ) ; reSCnt.append( 1 )
reSkip.append( re.compile(r"setmainlanguage{english}") ) ; reSCnt.append( 1 )
reSkip.append( re.compile(r"addto.captionsenglish") ) ; reSCnt.append( 1 )
reSkip.append( re.compile(r"set....font{Free") ) ; reSCnt.append( 7 )
reSkip.append( re.compile(r"ucharclasses") ) ; reSCnt.append( 1 )
reSkip.append( re.compile(r"unicode-math") ) ; reSCnt.append( 1 )

#reSkip.append( re.compile(r"") )
#reSkip.append( re.compile(r"") )
#reSkip.append( re.compile(r"") )

# ugly, manually fix citations in captions one by one
rec1  = re.compile(r"example prediction from ....hyperlink.cite.references:id6..WKA.20...." )
rec1t = 'example prediction from {[}\\\\protect\\\\hyperlink{cite.references:id6}{WKA+20}{]}' # note, quad \ needed!

rec2  = re.compile(r"parametrized GAN {\[}.hyperlink{cite.references:id2}{CTS.21}{\]}" )
rec2t = "parametrized GAN {[}\\\\protect\\\\hyperlink{cite.references:id2}{CTS+21}{]}"

rec3  = re.compile(r"approach using continuous convolutions {\[}.hyperlink{cite.references:id12}{UPTK19}{\]}" )
rec3t = "approach using continuous convolutions {[}\\\\protect\\\\hyperlink{cite.references:id12}{UPTK19}{]}"


def parseF(inf,outf,reSkip,reSCnt):
	print("Fixup, "+inf+" -> "+outf+" ")
	with open(outf, 'w') as fout:
		with open(inf, 'r') as f:
			c = 0
			skip = 0
			skipTot = 0
			for line in iter(f.readline, ''):

				# skip lines?
				if skip==0:
					for r in range(len(reSkip)):
						t = reSkip[r].search(str(line))
						if t is not None:
							#print(format(c)+" skip due to '" + format(t) +"',  RE #"+format(r)+" , skip "+format(reSCnt[r]) )  # debug
							skip = reSCnt[r]
							skipTot += reSCnt[r]

				if skip>0:
					skip = skip-1
					fout.write("% SKIP due to RE #"+format(r)+" , L"+format(reSCnt[r]) +"   "+line)
					#print("S "+line[:-1]) # debug
				else:
					# fix captions
					line = rec1.sub( rec1t, line )  # replace 
					line = rec2.sub( rec2t, line )  # replace 
					line = rec3.sub( rec3t, line )  # replace 

					fout.write(line)
					#print(line[:-1]) # debug

				c = c+1

				# line = re.sub('’', '\'', str(line))
				# line = re.sub('[abz]', '.', str(line))

				# t = ft3.search(str(line))
				# if t is not None:
				# 	print("H " + format(t) +"  "+ format(t.group(0)) )

				# t = fte.search(str(line))
				# if t is not None:
				# 	print("E " + format(t) + format(t.group(0)) )
	print("Fixup -> "+outf+" done, skips: "+format(skipTot)  +" \n")

parseF(inf,outf,reSkip,reSCnt)

#---

inf  = "sphinxmessages-in.sty" 
outf = "sphinxmessages.sty"

reSkip = [] ; reSCnt = []
reSkip.append( re.compile(r"addto.captionsenglish") ) ; reSCnt.append( 1 )

parseF(inf,outf,reSkip,reSCnt)

#---

inf  = "book-in.aux" 
outf = "book.aux"

# remove selectlang eng statements from book aux
reSkip = [] ; reSCnt = []
reSkip.append( re.compile(r"selectlanguage...english") ) ; reSCnt.append( 1 )

parseF(inf,outf,reSkip,reSCnt)

#---

# same, selectlanguage for toc
inf  = "book-in.toc" 
outf = "book.toc"
parseF(inf,outf,reSkip,reSCnt)

#---

inf  = "sphinxmanual-in.cls" 
outf = "sphinxmanual.cls"

# remove openright option from style
reSkip = [] ; reSCnt = []
reSkip.append( re.compile(r"PassOptionsToClass.openright...sphinxdocclass") ) ; reSCnt.append( 1 )

parseF(inf,outf,reSkip,reSCnt)

