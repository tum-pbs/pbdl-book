import json, re, os

fileList = [ 
	"diffphys-code-burgers.ipynb", "diffphys-code-sol.ipynb", "physicalloss-code.ipynb", # TF
	"bayesian-code.ipynb", "supervised-airfoils.ipynb" # pytorch
	]

#fileList = [ "diffphys-code-burgers.ipynb"] # debug

# shorten data line: "0.008612174447657694, 0.02584669669548606, 0.043136357266407785"

for fnOut in fileList:
	fn = fnOut[:-5] + "bak"
	print("renaming "+fnOut+ " to "+fn )
	if os.path.isfile(fnOut):
		os.rename(fnOut, fn)
	if not os.path.isfile(fn):
		print("Error: "+fn+" missing!")
		exit(1)

	#continue # exit(1)

	#fn="diffphys-code-burgers.ipynb"
	#fnOut="diffphys-code-burgers-r.ipynb"

	with open(fn) as file:
		d = json.load(file)

	#print(d.keys()) #print(d["cells"][0].keys())

	re1 = re.compile(r"WARNING:tensorflow:")
	re2 = re.compile(r"UserWarning:")
	re3 = re.compile(r"\[0.008612174447657694, 0.02584669669548606, 0.043136357266407785.+\]" )
	re3t = "[0.008612174447657694, 0.02584669669548606, 0.043136357266407785 ... ]"

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

			for j in range(len( d[t][i]["source"] )):
				#print( d[t][i]["source"][j] )
				#print( type(d[t][i]["source"][j] ))
				dsOut = re3.sub( re3t, d[t][i]["source"][j] )  # replace long number string (only for burgers)
				d[t][i]["source"][j] = dsOut
				deletes = deletes+1
				#print( d[t][i]["source"][j] +"\n >>> \n" +d2 )

			#print(len( d[t][i]["outputs"] ))
			for j in range(len( d[t][i]["outputs"] )):
				#print(type( d[t][i]["outputs"][j] ))
				#print( d[t][i]["outputs"][j].keys() )

				# images
				if d[t][i]["outputs"][j]["output_type"]=="stream":
					print(  len( d[t][i]["outputs"][j]["text"] ) )

					dell = [] # collect entries to delete
					for k in range(  len( d[t][i]["outputs"][j]["text"] )  ):
						nums = []
						nums.append( re1.search( d[t][i]["outputs"][j]["text"][k] ) )
						nums.append( re2.search( d[t][i]["outputs"][j]["text"][k] ) )
						if (nums[0] is None) and (nums[1] is None):
							okay = okay+1
						else: # delete line "dell"
							deletes = deletes+1
							dell.append(d[t][i]["outputs"][j]["text"][k])
							#print( format(nums) +"  " + d[t][i]["outputs"][j]["text"][k] ) # len( d[t][i]["outputs"][j]["text"][k] ) )

					for dl in dell:
						d[t][i]["outputs"][j]["text"].remove(dl)

					print( format( len( d[t][i]["outputs"][j]["text"] )) + " A")

	#print(d["cells"])

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

