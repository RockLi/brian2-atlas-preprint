"""Check source identity, manuscript structure, table values and captured pixels."""
from pathlib import Path
import re,json,hashlib,collections
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
m=json.loads((D/'source_manifest.json').read_text());checks={}
for key,path in [('english_pdf_sha256',ROOT/'output/pdf/brian2-atlas-preprint.pdf'),('english_manuscript_sha256',D.parent/'MANUSCRIPT.md'),('english_supplement_sha256',D.parent/'SUPPLEMENTARY.md')]:
    checks[key]=hashlib.sha256(path.read_bytes()).hexdigest()==m[key];assert checks[key]
e=(D.parent/'MANUSCRIPT.md').read_text();z=(D/'MANUSCRIPT.md').read_text();r=(D/'RESOURCE_ANALYSIS.md').read_text();sup=(D.parent/'SUPPLEMENTARY.md').read_text().split('## S21.',1)[1]
def headings(s):return [re.match(r'(#+) (\d+(?:\.\d+)*)(?:[. ]|$)',l).groups() for l in s.splitlines() if re.match(r'(#+) (\d+(?:\.\d+)*)(?:[. ]|$)',l)]
checks['numbered_section_sequence']=headings(e)==headings(z);assert checks['numbered_section_sequence']
checks['heading_level_counts']=collections.Counter(len(l.split(' ')[0]) for l in e.splitlines() if l.startswith('#'))==collections.Counter(len(l.split(' ')[0]) for l in z.splitlines() if l.startswith('#'));assert checks['heading_level_counts']
def prose_sections(s):
    parts=re.split(r'^#{2,4} (.+)$',s,flags=re.M)
    return {re.match(r'\d+(?:\.\d+)*',parts[i])[0]:sum(1 for t in parts[i+1].split('\n\n') if t.strip() and not t.strip().startswith(('!','**Figure','**图','**Table','**表','|'))) for i in range(1,len(parts),2) if re.match(r'\d+(?:\.\d+)*',parts[i])}
checks['per_section_prose_paragraph_counts']=prose_sections(e)==prose_sections(z);assert checks['per_section_prose_paragraph_counts']
checks['main_figures']=len(re.findall(r'^!\[',z,re.M))==10;assert checks['main_figures']
checks['citation_anchors']=collections.Counter(re.findall(r'\(#ref\d+\)',e))==collections.Counter(re.findall(r'\(#ref\d+\)',z));assert checks['citation_anchors']
checks['references_verbatim']=e.split('## References',1)[1].split('## Appendix A',1)[0].strip()==z.split('## 参考文献',1)[1].split('## 附录A',1)[0].strip();assert checks['references_verbatim']
checks['source_hashes_preserved']=set(re.findall(r'[a-f0-9]{40,64}',e))==set(re.findall(r'[a-f0-9]{40,64}',z));assert checks['source_hashes_preserved']
def tables(s):
    return [b.splitlines() for b in re.findall(r'(?:^\|.*\n)+',s,re.M)]
def nums(line):return re.findall(r'\d+(?:[,.]\d+)*',line)
et,zt=tables(e),tables(z);assert len(et)==len(zt)==7
checks['table_row_counts']=[len(x)==len(y) for x,y in zip(et,zt)];assert all(checks['table_row_counts'])
checks['table_numerical_values']=[all(nums(x)==nums(y) for x,y in zip(a[2:],b[2:])) for a,b in zip(et,zt)];assert all(checks['table_numerical_values']),checks['table_numerical_values']
def codes(t):return [re.findall(r'\b(?:B|X|NR)\b',line) for line in t[2:]]
checks['compatibility_codes']=codes(et[-1])==codes(zt[-1]);assert checks['compatibility_codes']
# Two S4 counts are translated from English billion/trillion to Chinese 亿/万亿.
es4=tables(sup)[0];zs4=tables(r)[0];assert len(es4)==len(zs4)==15
s4=[]
for i,(a,b) in enumerate(zip(es4[2:],zs4[2:])):
    if i==0:s4.append('86 billion / 86 trillion' in a and '860亿／86万亿' in b)
    elif i==11:s4.append('576 million' in a and '5.76亿' in b and '55,296,000,000,000' in b)
    else:s4.append(nums(a)==nums(b))
checks['s4_numerical_values']=s4;assert all(s4),s4
checks['resource_subsections']=len(re.findall(r'^### S21\.',r,re.M))==6;assert checks['resource_subsections']
es13='## S13.'+ (D.parent/'SUPPLEMENTARY.md').read_text().split('## S13.',1)[1].split('## S14.',1)[0]
zs13=(D/'DENDRITIC_SUPPLEMENT.md').read_text()
checks['s13_source_hashes']=set(re.findall(r'[a-f0-9]{40,64}',es13))==set(re.findall(r'[a-f0-9]{40,64}',zs13));assert checks['s13_source_hashes']
checks['s5_numerical_values']=[nums(x)==nums(y) for x,y in zip(tables(es13)[0][2:],tables(zs13)[0][2:])];assert len(checks['s5_numerical_values'])==2 and all(checks['s5_numerical_values'])
checks['superseded_dendritic_result_absent']='0.861' not in e+z+es13+zs13;assert checks['superseded_dendritic_result_absent']
checks['main_default_dendritic_rows']=len(et[4])==5 and '11.564384' not in e+z;assert checks['main_default_dendritic_rows']
checks['figure_data_geometry']=all(x['geometry_and_embedded_images_unchanged'] for x in json.loads((D/'figure_translation_validation.json').read_text()));assert checks['figure_data_geometry']
orig=json.loads((D.parent/'data/b2ir_visualization/capture.json').read_text());zh=json.loads((D/'b2ir_panels.json').read_text())
checks['s1_original_pixels_and_crops']=all(p['crop']==q['crop'] and p['image_sha256']==q['image_sha256'] and hashlib.sha256((D.parent/p['image']).read_bytes()).hexdigest()==p['image_sha256'] for p,q in zip(orig['panels'],zh['panels']));assert checks['s1_original_pixels_and_crops']
record={'status':'passed','source_pdf_sha256':m['english_pdf_sha256'],'checks':checks,'scope':m['scope'],'chinese_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'MANUSCRIPT.md',D/'DENDRITIC_SUPPLEMENT.md',D/'RESOURCE_ANALYSIS.md',D/'b2ir_panels.json']}}
(D/'translation_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(checks,ensure_ascii=False,indent=2))
