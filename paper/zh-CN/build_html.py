"""Build a self-contained Chinese translation, including integrated supplement panels."""
from pathlib import Path
import base64,json,re,hashlib,html
from io import BytesIO
from PIL import Image
from markdown_it import MarkdownIt
D=Path(__file__).resolve().parent
md=MarkdownIt('commonmark',{'html':True}).enable('table').render
main=md((D/'MANUSCRIPT.md').read_text())
resource=md((D/'RESOURCE_ANALYSIS.md').read_text())
main=main.replace('<h2>参考文献</h2>','<section class="references"><h2>参考文献</h2>')
main=main.replace('<h2>附录A：','</section><h2 class="compatibility-appendix">附录A：')
resource=resource.replace('<h2>S21.','<h2 class="resource-section">S21.',1)
dendritic=md((D/'DENDRITIC_SUPPLEMENT.md').read_text()).replace('<h2>S13.', '<h2 class="dendritic-supplement">S13.',1)
body=main+dendritic+resource
body=re.sub(r'<a href="#ref(\d+)">\d+</a>',lambda m:'<sup class="citation"><a href="#ref'+m[1]+'">['+m[1]+']</a></sup>',body)
body=body.replace('<h1>面向异构与分布式神经仿真的统一中间表示与执行架构</h1>','<h1>面向异构与分布式神经仿真的<br>统一中间表示与执行架构</h1>')
body,count=re.subn(r'<p>(<img[^>]+>)</p>\s*<p>(<strong>(?:图\d+|补充图S2).*?)</p>',r'<figure>\1<figcaption>\2</figcaption></figure>',body,flags=re.S)
assert count==11,count
body=re.sub(r'(<p><strong>表2 \|.*?</p>\s*)<table>',r'\1<table class="cohort-table">',body,flags=re.S)
body=re.sub(r'(<p><strong>表5 \|.*?</p>\s*)<table>',r'\1<table class="dendritic-table">',body,flags=re.S)
body=re.sub(r'(<p><strong>表S4 \|.*?</p>\s*)<table>',r'\1<table class="resource-table">',body,flags=re.S)
for f in (D/'figures').glob('*.svg'):
    body=body.replace('src="figures/'+f.name+'"','src="data:image/svg+xml;base64,'+base64.b64encode(f.read_bytes()).decode()+'"')
assert not re.search('src="figures/',body)
assert '**' not in body
record=json.loads((D/'b2ir_panels.json').read_text()); original=json.loads((D.parent/'data/b2ir_visualization/capture.json').read_text())
for p,src in zip(record['panels'],original['panels']):
    assert p['crop']==src['crop'] and p['image_sha256']==src['image_sha256']
    image=D.parent/p['image'];assert hashlib.sha256(image.read_bytes()).hexdigest()==p['image_sha256']
    im=Image.open(image).crop(p['crop']);buf=BytesIO();im.save(buf,format='PNG')
    body+='</main>' if p==record['panels'][0] else ''
    body+=f'<section class="s1-panel"><h2>补充图S1 · {html.escape(p["title"])}</h2><img alt="{html.escape(p["title"])}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"><p>{p["caption"]}</p></section>'
css='''
@page {size:A4;margin:19mm 18mm 19mm;@bottom-right{content:counter(page);font:8pt Arial;color:#67737c}}
@page screenshot {size:A4 landscape;margin:15mm;@bottom-right{content:counter(page);font:8pt Arial;color:#67737c}}
*{box-sizing:border-box;-webkit-print-color-adjust:exact;print-color-adjust:exact}
body{margin:0;color:#202830;background:#fff;font:10.5pt/1.62 "Hiragino Sans GB","PingFang SC",Arial,sans-serif}
main{margin:0;padding:0}
h1,h2,h3,h4{color:#1d405b;font-family:"Hiragino Sans GB","PingFang SC",Arial,sans-serif;break-after:avoid}
h1{font-size:23pt;line-height:1.35;margin:0 0 15pt}
h2{font-size:14pt;line-height:1.4;margin:18pt 0 9pt;padding-top:9pt;border-top:1px solid #dde3e7}
h3{font-size:11.5pt;line-height:1.4;margin:14pt 0 6pt}
h4{font-size:10.5pt;line-height:1.4;margin:12pt 0 6pt}
p{margin:0 0 9pt;orphans:3;widows:3;line-break:strict}
.author-block{font:13pt/1.55 "Hiragino Sans GB",Arial,sans-serif;margin:0 0 15pt}
.author-block span{font-size:10pt;color:#64707a}
a{color:#245c85;text-decoration:none;overflow-wrap:anywhere}
sup.citation{font-size:.75em;line-height:0;vertical-align:super}
code{font:9pt/1.5 Menlo,"Hiragino Sans GB",monospace;overflow-wrap:anywhere}
figure{margin:14pt 0;break-inside:avoid}
figure img{display:block;max-width:100%;width:100%;max-height:156mm;object-fit:contain;margin:0 auto 8pt}
figcaption{font:9pt/1.5 "Hiragino Sans GB",Arial,sans-serif;line-break:strict}
table{border-collapse:collapse;width:100%;font:8pt/1.48 "Hiragino Sans GB",Arial,sans-serif;table-layout:fixed;margin:8pt 0 14pt;break-inside:avoid}
th,td{padding:5pt 4pt;border-bottom:1px solid #dbe1e5;text-align:left;vertical-align:top;overflow-wrap:anywhere;line-break:strict}
th{background:#edf3f6}
tr{break-inside:avoid}
thead{display:table-header-group}
table.cohort-table{break-inside:auto}
table.dendritic-table th:first-child,table.dendritic-table td:first-child{width:38%}
table.dendritic-table th:not(:first-child),table.dendritic-table td:not(:first-child){width:12.4%}
table.resource-table{break-inside:avoid}
p:has(+table){break-after:avoid;font-size:9pt;line-height:1.5}
h2.compatibility-appendix,h2.resource-section,h2.dendritic-supplement{break-before:page}
h2.compatibility-appendix+p+table th:first-child,h2.compatibility-appendix+p+table td:first-child{width:42%}
h2.compatibility-appendix+p+table th:not(:first-child),h2.compatibility-appendix+p+table td:not(:first-child){width:14.5%;text-align:center}
h2.compatibility-appendix+p+table th,h2.compatibility-appendix+p+table td{padding:4pt 3pt}
.references{font-size:9pt;line-height:1.45}
.references p,.references ol{margin-bottom:6pt;break-inside:avoid}
.references ol{padding-left:19pt}
.s1-panel{page:screenshot;break-before:page;break-after:page}
.s1-panel h2{border:0;margin:0 0 5mm;padding:0;font-size:13pt;line-height:1.4}
.s1-panel img{display:block;max-width:267mm;max-height:135mm;object-fit:contain;margin:0 auto 5mm}
.s1-panel p{font-size:9pt;line-height:1.5;margin:0}
'''
text='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="author" content="Xinjun Li"><title>面向异构与分布式神经仿真的统一中间表示与执行架构</title><style>'+css+'</style></head><body><main>'+body+'</body></html>'
(D/'MANUSCRIPT.html').write_text(text)
print('Self-contained HTML: 10 main figures, S13/S5, S21/S4/S2, 7 main tables, and three original S1 panels.')
