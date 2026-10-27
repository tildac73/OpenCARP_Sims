import ast,csv,zipfile,struct,array,json
from pathlib import Path
source=Path('/Users/matildacotton/Downloads/boundaryPilotV2 2/results')
out=Path('boundaryPilotV2_clinical_R')
rows=[]; errors=[]
for condition in ['intact','lesion']:
 with zipfile.ZipFile(source/f'{condition}Traces.npz') as z:
  with z.open('measuredMv.npy') as f:
   assert f.read(6)==b'\x93NUMPY'
   v=f.read(2);size=struct.unpack('<H' if v[0]==1 else '<I',f.read(2 if v[0]==1 else 4))[0]
   h=ast.literal_eval(f.read(size).decode());assert h['descr']=='<f4' and not h['fortran_order']
   count,width=h['shape'];values=array.array('f');values.frombytes(f.read())
  table=list(csv.DictReader(open(source/f'{condition}Points.csv')));assert len(table)==count
  for row in table:
   i=int(row['traceIndex']);trace=values[i*width:(i+1)*width];pp=max(trace)-min(trace)
   errors.append(abs(pp-float(row['ppMv'])));row['tracePP']=pp;rows.append(row)
with open(out/'trace_verified_points.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'trace_verification.json').write_text(json.dumps({'rows':len(rows),'maximum_absolute_PP_difference_mV':max(errors),'source':str(source),'method':'Peak-to-peak recalculated directly from every saved measuredMv electrogram trace; final beat 7500–7900 ms.'},indent=2))
print('Verified',len(rows),'traces; maximum PP difference',max(errors))
