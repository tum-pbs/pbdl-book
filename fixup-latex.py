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

# latex fixup, remove references chapter
reSkip.append( re.compile(r"chapter.References" ) )
reSkip.append( re.compile(r"detokenize.references.references" ) )

#reSkip.append( re.compile(r"") )
#reSkip.append( re.compile(r"") )
#reSkip.append( re.compile(r"") )

# ugly, manually fix citations in captions one by one
recs = []; rect = []
recs.append( re.compile(r"example prediction from ....hyperlink.cite.references:id6..WKA.20...." ) )
rect.append( 'example prediction from {[}\\\\protect\\\\hyperlink{cite.references:id6}{WKA+20}{]}' ) # note, quad \ needed!

recs.append( re.compile(r"parametrized GAN {\[}.hyperlink{cite.references:id2}{CTS.21}{\]}" ) )
rect.append( "parametrized GAN {[}\\\\protect\\\\hyperlink{cite.references:id2}{CTS+21}{]}" )

recs.append( re.compile(r"approach using continuous convolutions {\[}.hyperlink{cite.references:id12}{UPTK19}{\]}" ) )
rect.append( "approach using continuous convolutions {[}\\\\protect\\\\hyperlink{cite.references:id12}{UPTK19}{]}" )

# fixup title , cumbersome...

# fix backslashes...  saves at least typing a few of them! still needs manual \ -> \\ , could be done better
tt =( 'hrule\n' + 
	'\\vspace{4cm}\n' + 
	'\\begin{center}\n' + 
	'\\sphinxstylestrong{\\Huge \\textsf{Physics-based Deep Learning}} \\\\ \\vspace{0.5cm} \n' + 
	'\\sphinxstylestrong{\\LARGE \\textsf{\\url{http://physicsbaseddeeplearning.org}}} \\\\ \\vspace{3cm} \n' + 
	'\\noindent\\sphinxincludegraphics[height=220\\sphinxpxdimen]{{teaser}.jpg} \\\\ \\vspace{2cm} \n' + 
	'\\textsf{\\large N. Thuerey, P. Holl, M. Mueller, P. Schnell, F. Trost, K. Um} \n' + 
	'\\end{center}\n' )

#print(tt);
recBST1 = re.compile(r"\\") 
recBST1t = '\\\\\\\\'  
tt = recBST1.sub( recBST1t, tt )  # replace all
#print(tt); exit(1)

# insert instead of sphinx version
recs.append( re.compile(r"sphinxmaketitle") )
rect.append( tt )

# remove authors
recs.append( re.compile(r"author{.*}") )
rect.append( 'author{}' )

# center date
recs.append( re.compile(r"date{(.*)}") )
rect.append( r'date{\\centering{\1}}' )



# ---

# only do replacements via recs for book.tex , via applyRecs=True
def parseF(inf,outf,reSkip,reSCnt,applyRecs=False):
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
					if applyRecs:
						# fix captions and apply other latex replacements
						for i in range(len(recs)):
							line = recs[i].sub( rect[i], line )  # replace all

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

parseF(inf,outf,reSkip,reSCnt,applyRecs=True)

#exit(1); print("debug exit!"); exit(1)

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

# disable for now?
if 0:
	inf  = "sphinxmanual-in.cls" 
	outf = "sphinxmanual.cls"

	# remove openright option from style
	reSkip = [] ; reSCnt = []
	reSkip.append( re.compile(r"PassOptionsToClass.openright...sphinxdocclass") ) ; reSCnt.append( 1 )

	parseF(inf,outf,reSkip,reSCnt)

