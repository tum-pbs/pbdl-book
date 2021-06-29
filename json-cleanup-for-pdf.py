import json, re

fn="diffphys-code-burgers.ipynb"
fnOut="diffphys-code-burgers-r.ipynb"

with open(fn) as file:
	d = json.load(file)

print(d.keys())
#print(d["cells"][0].keys())

re1 = re.compile(r"WARNING:tensorflow:")

t="cells"
for i in range(len(d[t])):
	#for i in range(len(d[t])):
		#print(d[t][0]["cell_type"])
	#print(d[t][i]["cell_type"])

	# remove images after code

	if d[t][i]["cell_type"]=="code":
		#print(d[t][i].keys())
		#d[t][i]["outputs"] = ""
		#print(d[t][i]["outputs"])

		#print(len( d[t][i]["outputs"] ))
		for j in range(len( d[t][i]["outputs"] )):
			#print(type( d[t][i]["outputs"][j] ))
			#print( d[t][i]["outputs"][j].keys() )

			# images
			if d[t][i]["outputs"][j]["output_type"]=="stream":
				print(  len( d[t][i]["outputs"][j]["text"] ) )

				dell = []
				for k in range(  len( d[t][i]["outputs"][j]["text"] )  ):
					num = re1.search( d[t][i]["outputs"][j]["text"][k] )
					if num is not None:
						dell.append(d[t][i]["outputs"][j]["text"][k])
						print( format(num) +"  " + d[t][i]["outputs"][j]["text"][k] ) # len( d[t][i]["outputs"][j]["text"][k] ) )
				for dl in dell:
					d[t][i]["outputs"][j]["text"].remove(dl)

				print( format( len( d[t][i]["outputs"][j]["text"] )) + " A")

#print(d["cells"])

with open(fnOut,'w') as fileOut:
	json.dump(d,fileOut, indent=1, sort_keys=True)

