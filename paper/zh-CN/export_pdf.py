"""Print the Chinese paper offline, leaving English artifacts unchanged."""
from pathlib import Path
import subprocess,tempfile,hashlib,json
from pypdf import PdfReader,PdfWriter
import pypdfium2 as pdfium
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
OUTPUT=ROOT/'output/pdf/brian2-atlas-preprint-zh-CN.pdf'
manifest=json.loads((D/'source_manifest.json').read_text())
assert hashlib.sha256((ROOT/'output/pdf/brian2-atlas-preprint.pdf').read_bytes()).hexdigest()==manifest['english_pdf_sha256']
with tempfile.TemporaryDirectory(prefix='atlas-zh-pdf-') as profile:
    pending=Path(profile)/'paper.pdf'
    log=Path(profile)/'chrome.log'
    with log.open('wb') as handle:
        p=subprocess.Popen(['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','--headless','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-extensions','--no-pdf-header-footer','--allow-file-access-from-files','--user-data-dir='+profile,'--print-to-pdf='+str(pending),(D/'MANUSCRIPT.html').as_uri()],stdout=subprocess.DEVNULL,stderr=handle)
        try:p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
    err=log.read_bytes()
    assert pending.is_file() and b'%%EOF' in pending.read_bytes()[-1024:],err.decode(errors='replace')[-2000:]
    r=PdfReader(pending);w=PdfWriter();w.clone_document_from_reader(r)
    document=pdfium.PdfDocument(pending)
    for i,page in enumerate(document):
        t=page.get_textpage().get_text_range()
        if 'S13. 附加树突CPU证据' in t:w.add_outline_item('补充材料S13：树突CPU验证与缓存消融',i)
        if 'S21. 以860亿神经元' in t:w.add_outline_item('补充材料S21：条件资源分析',i)
        if '补充图S1' in t and 'a |' in t:w.add_outline_item('补充图S1：活动回放与B2IR检查',i)
    document.close()
    w.add_metadata({'/Title':'面向异构与分布式神经仿真的统一中间表示与执行架构','/Author':'Xinjun Li','/Subject':'brian2-atlas；中文译稿；异构与分布式神经仿真'})
    with OUTPUT.open('wb') as f:w.write(f)
    print(json.dumps({'pdf':str(OUTPUT),'pages':len(r.pages),'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest()},ensure_ascii=False))
