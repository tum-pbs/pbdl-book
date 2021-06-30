import sys, os, re 
#import sys, os, psutil, subprocess, time, signal, re, json, logging, datetime, argparse
#import numpy as np

# fix jupyter book latex output

#filter_mem = re.compile(r".+\|\s+(\d+)MiB\s+/\s+(\d+)MiB\s+\|")#')
#ft2 = re.compile(r"tst")
#ft3 = re.compile(r"’")
#fte = re.compile(r"👋")

# notebooks, parse and remove lines with WARNING:tensorflow + next? ; remove full "name": "stderr" {} block?  (grep stderr *ipynb)

# TODO , replace phi symbol w text in phiflow

# TODO , filter tensorflow warnings? "WARNING:tensorflow:" eg in physloss-code
# also torch "UserWarning:" eg in supervised-airfoils
# from PINN burgers: 
#   u = np.asarray( [0.008612174447657694, 0.02584669669548606, 0.043136357266407785, 0.060491074685516746, 0.07793926183951633, 0.0954779141740818, 0.11311894389663882, 0.1308497114054023, 0.14867023658641343, 0.1665634396808965, 0.18452263429574314, 0.20253084411376132, 0.22057828799835133, 0.23865132431365316, 0.25673879161339097, 0.27483167307082423, 0.2929182325574904, 0.3109944766354339, 0.3290477753208284, 0.34707880794585116, 0.36507311960102307, 0.38303584302507954, 0.40094962955534186, 0.4188235294008765, 0.4366357052408043, 0.45439856841363885, 0.4720845505219581, 0.4897081943759776, 0.5072391070000235, 0.5247011051514834, 0.542067187709797, 0.5593576751669057, 0.5765465453632126, 0.5936507311857876, 0.6106452944663003, 0.6275435911624945, 0.6443221318186165, 0.6609900633731869, 0.67752574922899, 0.6939334022562877, 0.7101938106059631, 0.7263049537163667, 0.7422506131457406, 0.7580207366534812, 0.7736033721649875, 0.7889776974379873, 0.8041371279965555, 0.8190465276590387, 0.8337064887158392, 0.8480617965162781, 0.8621229412131242, 0.8758057344502199, 0.8891341984763013, 0.9019806505391214, 0.9143881632159129, 0.9261597966464793, 0.9373647624856912, 0.9476871303793314, 0.9572273019669029, 0.9654367940878237, 0.9724097482283165, 0.9767381835635638, 0.9669484658390122, 0.659083299684951, -0.659083180712816, -0.9669485121167052, -0.9767382069792288, -0.9724097635533602, -0.9654367970450167, -0.9572273263645859, -0.9476871280825523, -0.9373647681120841, -0.9261598056102645, -0.9143881718456056, -0.9019807055316369, -0.8891341634240081, -0.8758057205293912, -0.8621229450911845, -0.8480618138204272, -0.833706571569058, -0.8190466131476127, -0.8041372124868691, -0.7889777195422356, -0.7736033858767385, -0.758020740007683, -0.7422507481169578, -0.7263049162371344, -0.7101938950789042, -0.6939334061553678, -0.677525822052029, -0.6609901538934517, -0.6443222327338847, -0.6275436932970322, -0.6106454472814152, -0.5936507836778451, -0.5765466491708988, -0.5593578078967361, -0.5420672759411125, -0.5247011730988912, -0.5072391580614087, -0.4897082914472909, -0.47208460952428394, -0.4543985995006753, -0.4366355580500639, -0.41882350871539187, -0.40094955631843376, -0.38303594105786365, -0.36507302109186685, -0.3470786936847069, -0.3290476440540586, -0.31099441589505206, -0.2929180880304103, -0.27483158663081614, -0.2567388003912687, -0.2386513127155433, -0.22057831776499126, -0.20253089403524566, -0.18452269630486776, -0.1665634500729787, -0.14867027528284874, -0.13084990929476334, -0.1131191325854089, -0.09547794429803691, -0.07793928430794522, -0.06049114408297565, -0.0431364527809777, -0.025846763281087953, -0.00861212501518312] );
#        ->
#   u = np.asarray( [0.008612174447657694, 0.02584669669548606, ... ] )


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

# reSkip.append( re.compile(r"") )

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

reSkip = [] ; reSCnt = []
reSkip.append( re.compile(r"selectlanguage...english") ) ; reSCnt.append( 1 )

parseF(inf,outf,reSkip,reSCnt)

#---

# same, selectlanguage 
inf  = "book-in.toc" 
outf = "book.toc"
parseF(inf,outf,reSkip,reSCnt)
