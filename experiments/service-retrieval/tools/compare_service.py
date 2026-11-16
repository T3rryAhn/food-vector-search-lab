"""Run baseline and query-prefix comparison in a separate process; never calls public APIs."""
import argparse, sys, json, time, hashlib, contextlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--service-root',required=True);p.add_argument('--cases',required=True);p.add_argument('--enhanced',action='store_true');p.add_argument('--only-enhanced',action='store_true');a=p.parse_args()
root=Path(a.service_root);sys.path.insert(0,str(root))
with contextlib.redirect_stdout(sys.stderr):
 from src import main
 main.load_resources()
import baseline
baseline.resources=main.resources
model=main.resources['model']
class QueryPrompt:
 def encode(self,texts,**kw):
  return model.encode(['query: '+t.removeprefix('Represent this sentence for searching relevant passages: ') for t in texts],**kw)
variants=['baseline','query_prompt']
if a.enhanced:
 from retrieval import recommend
 variants+=['menu_hybrid']
if a.only_enhanced:variants=['menu_hybrid']
results=[]
for variant in variants:
 main.resources['model']=model if variant=='baseline' else QueryPrompt()
 for case in json.loads(Path(a.cases).read_text(encoding='utf-8-sig')):
  start=time.monotonic()
  if variant=='menu_hybrid':
   main.resources['model']=model
   items=recommend(main.resources,case['query'],case['category'],case['k'])
  else:items=baseline.recommend_food(query=case['query'],category=case['category'],k=case['k'])
  results.append(dict(variant=variant,id=case['id'],query=case['query'],seconds=round(time.monotonic()-start,4),results=items))
  time.sleep(.5)
hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['src/main.py','src/query_expansion.py','src/diversify.py','data/food_data.csv','outputs/indices/food_faiss.index','outputs/indices/food_meta.pkl']}
print(json.dumps({'hashes':hashes,'results':results},ensure_ascii=False,indent=2))
