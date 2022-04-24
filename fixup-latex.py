import sys, os, re 

# fix jupyter book latex output

# TODOs 
# - check, remove full "name": "stderr" {} block?  (grep stderr *ipynb) ???
# 		or whole warning/err empty blocks...
# - replace phi symbol w text in phiflow

# older tests
#ft1 = re.compile(r"’")
#ft2 = re.compile(r"👋")


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
reSkip.append( re.compile(r"chapter.References" ) ); reSCnt.append( 1 )
reSkip.append( re.compile(r"detokenize.references.references" ) ); reSCnt.append( 1 )

#reSkip.append( re.compile(r"") ); reSCnt.append( 1 )
#reSkip.append( re.compile(r"") ); reSCnt.append( 1 )
#reSkip.append( re.compile(r"") ); reSCnt.append( 1 )

# ugly, manually fix citations in captions one by one
# need manual IDs!
recs = []; rect = []
# ID4 CTS
#recs.append( re.compile(r"parametrized GAN {\[}.hyperlink{cite.references:id4}{CTS.21}{\]}" ) )
recs.append( re.compile(r"parametrized GAN {\[}.hyperlink{cite.references:id5}{CTS.21}{\]}" ) )
rect.append( "parametrized GAN {[}\\\\protect\\\\hyperlink{cite.references:id5}{CTS+21}{]}" )
# ID8 WKA
recs.append( re.compile(r"example prediction from ....hyperlink.cite.references:id9..WKA.20...." ) )
rect.append( 'example prediction from {[}\\\\protect\\\\hyperlink{cite.references:id9}{WKA+20}{]}' ) # note, quad \ needed!
# ID14 UPTK
recs.append( re.compile(r"approach using continuous convolutions {.}.hyperlink{cite.references:id15}{UPTK19}{.}" ) )
rect.append( "approach using continuous convolutions {[}\\\\protect\\\\hyperlink{cite.references:id15}{UPTK19}{]}" )

# fixup unicode symbols 
# compare book-in2.tex -> book.tex after iconv 

recs.append( re.compile(r"’" ) ) # unicode ' 
rect.append( "\'" )

recs.append( re.compile(r"Φ") ) # phiflow , ... differentiable simulation framework ...
rect.append( "$\\\\phi$" )

recs.append( re.compile(r"“") ) # "..."
rect.append( "``" )

recs.append( re.compile(r"”") )
rect.append( "\'\'" )

recs.append( re.compile(r"–") )
rect.append( "-" )

recs.append( re.compile(r"…") )
rect.append( "..." )

recs.append( re.compile(r"‘") )
rect.append( "'" )

recs.append( re.compile(r" ") ) # weird spaces in bib?
rect.append( " " )

# recs.append( re.compile(r"") )
# rect.append( "" )

# recs.append( re.compile(r"") )
# rect.append( "" )

# recs.append( re.compile(r"") )
# rect.append( "" )


# fixup title , cumbersome...

# fix backslashes...  saves at least typing a few of them! still needs manual \ -> \\ , could be done better
tt =( 'hrule\n' + 
	'\\vspace{3cm}\n' + 
	'\\begin{center}\n' + 
	'\\sphinxstylestrong{\\Huge \\textsf{Physics-based Deep Learning}} \\\\ \\vspace{0.5cm} \n' + 
	'\\sphinxstylestrong{\\LARGE \\textsf{\\url{http://physicsbaseddeeplearning.org}}} \\\\ \\vspace{2cm} \n' + 
	'\\noindent\\sphinxincludegraphics[height=420\\sphinxpxdimen]{{logo-xl}.jpg} \\\\ \\vspace{1cm} \n' + 
	'\\textsf{\\large N. Thuerey, P. Holl, M. Mueller, P. Schnell, F. Trost, K. Um} \n' + 
	'\\\\ \\ \\\\ {\\Large(v0.2)} \n' +   # manually update version!
	'\\end{center}\n' )

#print(tt);
recBST1 = re.compile(r"\\") 
recBST1t = '\\\\\\\\'  
tt = recBST1.sub( recBST1t, tt )  # replace all
#print(tt); exit(1)

# skip html version logo-xl , todo: remove figure env around it, move divider-mult image above "Coming up" para
reSkip.append( re.compile(r"noindent.sphinxincludegraphics..logo-xl..jpg" ) ); reSCnt.append( 1 )

# insert instead of sphinx version
recs.append( re.compile(r"sphinxmaketitle") )
rect.append( tt )

# remove authors
recs.append( re.compile(r"author{.*}") )
rect.append( 'author{}' )

# center date
recs.append( re.compile(r"date{(.*)}") )
rect.append( r'date{\\centering{\1}}' )

#print(len(rect))
#print(len(recs))
#exit(1)

# sanity check
if len(rect) != len(recs):
	print("Error rect and recs len have to match!"); exit(1)

recsCnt = []
for n in range(len(recs)):
    recsCnt.append(0)

# ---

# only do replacements via recs for book.tex , via applyRecs=True
def parseF(inf,outf,reSkip,reSCnt,applyRecs=False):
	print("Fixup, "+inf+" -> "+outf+" ")

	if len(reSkip) != len(reSCnt): # sanity check
		print("Error for "+inf+" reSkip cnt: " + format([ len(reSkip), len(reSCnt) ]) )
		exit(1)

	with open(outf, 'w') as fout:
		with open(inf, 'r') as f:
			c = 0
			skip = 0
			skipTot = 0
			for line in iter(f.readline, ''):

				# skip lines?
				rSkip = -1
				if skip==0:
					for r in range(len(reSkip)):
						t = reSkip[r].search(str(line))
						if t is not None:
							#print(format(c)+" skip due to '" + format(t) +"',  RE #"+format(r)+" , skip "+format(reSCnt[r]) )  # debug
							skip = reSCnt[r]
							skipTot += reSCnt[r]
							rSkip = r

				if skip>0:
					skip = skip-1
					fout.write("% SKIP due to RE #"+format(rSkip)+" , L"+format(reSCnt[rSkip]) +"   "+line)
					#print("S "+line[:-1]) # debug
				else:
					if applyRecs:
						# fix captions and apply other latex replacements
						#print(len(rect)); print(len(recs))
						for i in range(len(recs)):
							ric = len( recs[i].findall( line ) )
							#if ric>0: print(ric)
							recsCnt[i] += ric  # count, for sanity check
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

haveError = False; recsCntT = 0
for i in range(len(recs)):
	recsCntT += recsCnt[i]
	if(recsCnt[i]==0):
		print("Error, re %d , '%s' not found!" % (i,recs[i]))
		haveError = True

if haveError: 
	print("Some REs were not found, maybe cite.references:idX is wrong! Those have to be manually checked")
	exit(1)
else:
	print("book-in2: %d re replacements\n" % (recsCntT) )

# print("debug exit!"); exit(1)

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

# disable for now? keep openRight
if 0:
	inf  = "sphinxmanual-in.cls" 
	outf = "sphinxmanual.cls"

	# remove openright option from style
	reSkip = [] ; reSCnt = []
	reSkip.append( re.compile(r"PassOptionsToClass.openright...sphinxdocclass") ) ; reSCnt.append( 1 )

	parseF(inf,outf,reSkip,reSCnt)

