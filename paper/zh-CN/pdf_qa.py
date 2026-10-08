"""Validate the exported Chinese PDF and retain a review record."""
from pathlib import Path
import re,json,hashlib,logging,unicodedata
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('pypdf').setLevel(logging.ERROR)
from pypdf import PdfReader
import pypdfium2 as pdfium
D=Path(__file__).resolve().parent;ROOT=D.parents[1];p=ROOT/'output/pdf/brian2-atlas-preprint-zh-CN.pdf'
r=PdfReader(p);doc=pdfium.PdfDocument(p);texts=[unicodedata.normalize('NFKC',page.get_textpage().get_text_range()) for page in doc];joined='\n'.join(texts)
assert len(r.pages)>=36
assert '**' not in joined
assert 'Vec<u64>' in joined
assert '[1]' in joined
assert sum(len(re.findall(r'[\u4e00-\u9fff]',t)) for t in texts)>25000
assert all('S21.'+str(i) in joined for i in range(1,7))
assert all('图'+str(i)+' |' in joined for i in range(1,11))
assert all('表'+str(i)+' |' in joined for i in range(1,8))
assert '表S4 |' in joined and '补充图S2' in joined
assert 'S13. 附加树突CPU证据' in joined and '表S5 |' in joined
assert '0.861' not in joined
assert '11.564384' not in joined.split('S13. 附加树突CPU证据',1)[0]
assert all('补充图S1' in t for t in texts[-3:])
assert all(float(pg.mediabox.width)>float(pg.mediabox.height) for pg in r.pages[-3:])
assert all(float(pg.mediabox.width)<float(pg.mediabox.height) for pg in r.pages[:-3])
for value in ['1,087,452,431','1,062,437,152,208','46,298,756,762','3,000','24,000','55,296,000,000,000','110.592','810.215','1.720','4.578']:
 assert value in joined,value
outside=[]
for i,pg in enumerate(doc,1):
 tp=pg.get_textpage();width,height=pg.get_size()
 for j in range(tp.count_chars()):
  left,bottom,right,top=tp.get_charbox(j)
  if left<-.5 or right>width+.5 or bottom<-.5 or top>height+.5:outside.append({'page':i,'character_index':j})
 tp.close();pg.close()
assert not outside,outside[:5]
manifest=json.loads((D/'source_manifest.json').read_text());assert hashlib.sha256((ROOT/'output/pdf/brian2-atlas-preprint.pdf').read_bytes()).hexdigest()==manifest['english_pdf_sha256']
record={'status':'text-and-layout-checks-passed','pdf_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'page_count':len(r.pages),'chinese_character_count':sum(len(re.findall(r'[\u4e00-\u9fff]',t)) for t in texts),'outside_page_characters':0,'portrait_pages':len(r.pages)-3,'original_screenshot_landscape_pages':3,'english_pdf_matches_current_translation_manifest':True,'content_checks':'All 10 main figures, 7 main tables, S21.1–S21.6, S13/Table S5, Table S4, Figure S2 and three S1 panels; major exact event counts and resource quantities present.','visual_review':{'status':'pending'}}
(D/'pdf_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
(ROOT/'tmp/pdfs/zh-CN').mkdir(parents=True,exist_ok=True)
(ROOT/'tmp/pdfs/zh-CN/pdf_text.txt').write_text('\n\n'.join(f'PAGE {i}\n{t}' for i,t in enumerate(texts,1)))
for i,t in enumerate(texts,1):
 tags=[s for s in ['表2 |','表6 |','表S4 |','补充图S2 |','补充图S1 ·','附录A:'] if s in t]
 if tags:print(i,tags)
print(json.dumps({k:v for k,v in record.items() if k not in ['content_checks','visual_review']},ensure_ascii=False,indent=2))
