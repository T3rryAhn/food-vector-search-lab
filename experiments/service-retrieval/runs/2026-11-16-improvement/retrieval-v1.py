"""Menu-oriented retrieval. No model/index writes; conservative name/tag rules, not dietary guarantees."""
import re
from collections import Counter
MEAL_CATS={'밥류','면 및 만두류','국 및 탕류','찌개 및 전골류','구이류','볶음류','튀김류','전·적 및 부침류','찜류','조림류','죽 및 스프류'}
RAW_CATS={'수·조·어·육류','장류, 양념류','곡류, 서류 제품','과일류','두류, 견과 및 종실류','채소, 해조류'}
SPICY=re.compile('매운|매콤|얼큰|마라|불닭|김치|짬뽕|떡볶이|고추장')
MEAT=re.compile('소고기|쇠고기|돼지|닭|오리|고기|불고기|갈비|곱창|내장|햄|소시지|소세지|베이컨|오야꼬|돈코츠|차슈|돈가스|돈까스|탕수육')
def norm(s):return re.sub(r'[^가-힣a-z0-9]','',s.lower())
def kind(query,category):
 if category=='음료 및 차류':return 'drink'
 if category in {'빵 및 과자류','유제품류 및 빙과류'}:return 'snack'
 if category:return 'food'
 if re.search(r'음료|커피|라떼|주스|에이드|스무디|(?:^|\s)(?:차|티)(?:\s|$)',query):return 'drink'
 if re.search('디저트|간식|케이크|아이스크림|빙수|과자',query):return 'snack'
 return 'food'
def eligible(item,intent,category):
 name=item['food_name'];cat=item['cat_large']
 if cat in RAW_CATS or re.search(r'_(?:노른자|흰자|국물|면)$|소스/드레싱|밥제외',name):return False
 if category:return category in cat
 if intent=='drink':return cat=='음료 및 차류'
 if intent=='snack':return cat in {'빵 및 과자류','유제품류 및 빙과류'}
 return cat in MEAL_CATS or any(t in name for t in ['샐러드','샌드위치','월남쌈'])
def family(name):
 s=norm(name.split('_')[0]);return s.replace('물냉면','냉면').replace('회냉면','냉면').replace('비빔냉면','냉면')
def recommend(resources,query,category=None,k=6):
 intent=kind(query,category)
 # Keep negation intact: do not apply legacy expansion rules that invert salt intent.
 vector=resources['model'].encode([query],normalize_embeddings=True,prompt_name='query')
 index=resources['index'];scores,indices=index.search(vector,index.ntotal)
 cold=bool(re.search('차가운|시원한|냉면|아이스',query));hot=bool(re.search('따뜻한|뜨거운|뜨끈',query))
 mild=bool(re.search('맵지|안 매운|안매운|순한|담백한',query))
 meatless=bool(re.search('고기\s*없|고기\s*빼|채식',query))
 soup=bool(re.search('국물',query));noodle=bool(re.search('면\s*요리|냉면|국수',query))
 exact=norm(re.sub('추천|먹고 싶어|먹고싶어|주세요','',query))
 candidates=[]
 for score,i in zip(scores[0],indices[0]):
  item=resources['metadata'][int(i)];name=item['food_name'];tags=str(item.get('tags',''))
  if not eligible(item,intent,category):continue
  if cold and (not re.search('차가운|시원한|냉면|소바|냉국|스무디|아이스|ICED|빙수',name+' '+tags) or re.search('HOT|핫\(',name)):continue
  if hot and re.search('냉면|빙수|ICED|아이스',name):continue
  if mild and SPICY.search(name+' '+tags):continue
  if meatless and MEAT.search(name):continue
  if soup and not re.search('국|탕|찌개|전골|수프|스프',name):continue
  if noodle and item['cat_large']!='면 및 만두류':continue
  parts=[norm(p) for p in name.split('_')[:2]]
  lexical=bool(exact and (exact==parts[0] or len(parts)>1 and exact in [parts[0]+parts[1],parts[1]+parts[0]]))
  candidates.append((float(score)+(.4 if lexical else 0),item))
 candidates.sort(key=lambda x:x[0],reverse=True)
 result=[];seen=set();families=Counter()
 for score,item in candidates:
  name=item['food_name'];f=family(name)
  if norm(name) in seen or families[f]>=1:continue
  seen.add(norm(name));families[f]+=1
  result.append({'food_code':item['food_code'],'food_name':name,'category':item['cat_large'],'tags':str(item.get('tags','')),'score':score})
  if len(result)>=k:break
 return result
