import sys, json, re, os
# usage: json-cleanup-for-pdf.py <int>
# if int>0, disable PDF mode (only do WWW cleanup, note metadata.name still needs to be cleaned up manually)

# disableWrites = True # debugging

pdfMode = True

print(format(sys.argv))
if len(sys.argv)>1:
	if int(sys.argv[1])>0:
		print("WWW mode on")
		pdfMode = False

fileList = [ 
	"diffphys-code-burgers.ipynb", "diffphys-code-ns.ipynb", "diffphys-code-sol.ipynb", "physicalloss-code.ipynb", # TF
	"bayesian-code.ipynb", "supervised-airfoils.ipynb", # pytorch
	"reinflearn-code.ipynb", # phiflow
	"physgrad-comparison.ipynb", # jax
	"physgrad-code.ipynb", # pip
	]

#fileList = [ "physgrad-code.ipynb"] # debug, only 1 file
#fileList = [ "t1.ipynb" ] # debug


# main

for fnOut in fileList:
	if not os.path.isfile(fnOut):
		print("Error: "+fnOut+" not found!"); exit(1)

	# create backups
	fn0 = fnOut[:-5] + "bak"
	fn = fn0 + "0"; cnt = 0
	while os.path.isfile(fn):
		#print("Error: "+fn+" already exists!"); exit(1)
		print("Warning: "+fn+" already exists!")
		fn = fn0 + format(cnt); cnt=cnt+1

	print("renaming "+fnOut+ " to "+fn )
	if os.path.isfile(fnOut):
		os.rename(fnOut, fn)
	if not os.path.isfile(fn):
		print("Error: "+fn+" missing!")
		exit(1)

	with open(fn) as file:
		d = json.load(file)

	#print(d.keys()) #print(d["cells"][0].keys())

	# remove TF / pytorch warnings, build list of regular expressions to search for
	# double check, redundant with removing stderr cells (cf delE)
	res = []
	res.append( re.compile(r"WARNING:tensorflow:") )
	res.append( re.compile(r"UserWarning:") )
	res.append( re.compile(r"DeprecationWarning:") )
	res.append( re.compile(r"InsecureRequestWarning") ) # for https download
	res.append( re.compile(r"Building wheel") ) # phiflow install, also gives weird unicode characters
	res.append( re.compile(r"warnings.warn") )  # phiflow warnings
	res.append( re.compile(r"WARNING:absl") )  # jax warnings

	res.append( re.compile(r"ERROR: pip") )  # pip dependencies
	res.append( re.compile(r"requires imgaug") )  # pip dependencies
	res.append( re.compile(r"See the documentation of nn.Upsample") )  # pip dependencies

	# remove all "warnings.warn" from phiflow?

	# shorten data line: "0.008612174447657694, 0.02584669669548606, 0.043136357266407785"
	reD = re.compile(r"\[0.008612174447657694, 0.02584669669548606, 0.043136357266407785.+\]" )
	reDt = "[0.008612174447657694, 0.02584669669548606, 0.043136357266407785 ... ]"

	t="cells"
	okay = 0
	deletes = 0
	for i in range(len(d[t])):
		#for i in range(len(d[t])):
			#print(d[t][0]["cell_type"])
		#print(d[t][i]["cell_type"])

		# remove images after code

		if d[t][i]["cell_type"]=="code":
			#print(d[t][i].keys())
			#d[t][i]["outputs"] = ""
			#print(d[t][i]["outputs"])

			if pdfMode:
				for j in range(len( d[t][i]["source"] )):
					#print( d[t][i]["source"][j] )
					#print( type(d[t][i]["source"][j] ))
					dsOut = reD.sub( reDt, d[t][i]["source"][j] )  # replace long number string (only for burgers)
					d[t][i]["source"][j] = dsOut
					deletes = deletes+1
					#print( d[t][i]["source"][j] +"\n >>> \n" +d2 )

			delE = [] # collect whole entries (sections) to delete

			#print(len( d[t][i]["outputs"] ))
			for j in range(len( d[t][i]["outputs"] )):
				#print(type( d[t][i]["outputs"][j] ))
				#print( d[t][i]["outputs"][j].keys() )

				# search for error stderr cells
				if d[t][i]["outputs"][j]["output_type"]=="stream":
					#print("output j name: "+ format( d[t][i]["outputs"][j]["name"] ) )
					#print("output j: "+ format( d[t][i]["outputs"][j] ) )
					if d[t][i]["outputs"][j]["name"]=="stderr":
						print("stderr found! len text "+ format(len( d[t][i]["outputs"][j]["text"]) ) +", removing entry "+format(j) )
						delE.append(j) # remove the whole stderr entry

				# images
				if d[t][i]["outputs"][j]["output_type"]=="stream":
					#print("len "+  format(len( d[t][i]["outputs"][j]["text"] )) )

					dell = [] # collect lines to delete
					for k in range(  len( d[t][i]["outputs"][j]["text"] )  ):
						#print(" tout "+   d[t][i]["outputs"][j]["text"][k] ) # debug , print all lines
						nums = []; all_good = True
						for rr in range(len(res)):
							nums.append( res[rr].search( d[t][i]["outputs"][j]["text"][k] ) )
							if nums[-1] is not None:
								all_good = False # skip!

						if all_good:
							okay = okay+1
						else: # delete line "dell"
							deletes = deletes+1
							dell.append(d[t][i]["outputs"][j]["text"][k])
							#print( format(nums) +"  " + d[t][i]["outputs"][j]["text"][k] ) # len( d[t][i]["outputs"][j]["text"][k] ) )

					for dl in dell:
						d[t][i]["outputs"][j]["text"].remove(dl)
				#print("len after "+format( len( d[t][i]["outputs"][j]["text"] )) + " A") # debug

			# afterwards (potentially remove whole entries)
			# if len(delE)>0:
			# 	print("len bef "+format( len( d[t][i]["outputs"] )) + " A " + format(delE)) # debug
			for de in delE:
				#print(type(d[t][i]["outputs"])); print(de)
				d[t][i]["outputs"].pop(de) # remove array element
				deletes+=1
			# if len(delE)>0:
			# 	print("len after "+format( len( d[t][i]["outputs"] )) + " A") # debug

	if deletes==0:
		print("Warning: Nothing found in "+fn+"!")
		if not os.path.isfile(fnOut):
			os.rename(fn, fnOut)
		else:
			print("Error, both files exist!?")
			exit(1)

	else:
		print(" ... writing "+fnOut )
		with open(fnOut,'w') as fileOut:
			json.dump(d,fileOut, indent=1, sort_keys=True)

