import json
# for spell checking single files

fnOut="temp_reduced.txt"

fn="temp.txt"
with open(fn) as file:
	d = json.load(file)

print(d.keys())
#print(d["cells"][0].keys())

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
			print( d[t][i]["outputs"][j].keys() )
			if d[t][i]["outputs"][j]["output_type"]=="display_data":
				#if d[t][i]["outputs"][j]["data"]=="display_data":
				print(  d[t][i]["outputs"][j]["data"].keys() )
				#if(  d[t][i]["outputs"][j]["data"].contains('image/png') ):
				if( 'image/png' in d[t][i]["outputs"][j]["data"].keys() ):
					d[t][i]["outputs"][j]["data"]['image/png'] = ""

with open(fnOut,'w') as fileOut:
	json.dump(d,fileOut, indent=1, sort_keys=True)

