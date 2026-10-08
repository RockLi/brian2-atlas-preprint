"""Build the editable proposal and a paginated, linked Chinese PDF."""
from pathlib import Path
import re
import json
from html import escape
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer
from reportlab.lib.pagesizes import A4
from report_content import TITLE, REFS, PAGES

ROOT = Path(__file__).resolve().parents[3]
MD_PATH = ROOT / 'brian2-rust/docs/AFFECTIVE_AI_RESEARCH_PROPOSAL_20260912.md'
PDF_PATH = ROOT / 'output/pdf/affective-ai-research-proposal-20260912.pdf'
FONT = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
pdfmetrics.registerFont(TTFont('CJK', FONT))
PAGE_W, PAGE_H = A4
LEFT, RIGHT, TOP = 49, 49, 43
WIDTH = PAGE_W - LEFT - RIGHT
REF_GROUPS = [list(REFS)[i:i+8] for i in range(0,len(REFS),8)]

def inline(text):
    text = escape(text)
    def cite(m):
        n = int(m.group(1))
        assert n in REFS
        return f'<super><font size="7.5"><link href="#ref{n}" color="#333333">[{n}]</link></font></super>'
    return re.sub(r'\[(\d+)\]', cite, text)

def style(name, size=10.7, leading=None, **kwargs):
    return ParagraphStyle(name, fontName='CJK', fontSize=size, leading=leading or size*1.57,
                          textColor=colors.HexColor('#181818'), alignment=TA_LEFT,
                          wordWrap='CJK', splitLongWords=True, **kwargs)

def flows_for(page, size, first=False):
    body = style('body', size)
    flows = []
    if first:
        title_html=inline(TITLE).replace('网络的','网络的<br/>')
        flows += [Paragraph(title_html, style('title',22,31)), Spacer(1,15)]
    flows += [Paragraph(inline(page['title']),style('section',17,24)), Spacer(1,12)]
    for block in page['blocks']:
        if block[0] == 'p':
            flows += [Paragraph(inline(block[1]),body),Spacer(1,9)]
        elif block[0] == 'h':
            flows += [Spacer(1,3),Paragraph(inline(block[1]),style('sub',12,18)),Spacer(1,7)]
        elif block[0] == 'table':
            _, head, rows, ratios = block
            ratios = ratios or [1/len(head)]*len(head)
            col_widths=[WIDTH*x for x in ratios]
            cellstyle=style('cell',size-.6,(size-.6)*1.43)
            data=[[Paragraph(inline(x),cellstyle) for x in row] for row in [head]+rows]
            tab=Table(data,colWidths=col_widths,hAlign='LEFT')
            tab.setStyle(TableStyle([
                ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAEAEA')),
                ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#666666')),
                ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#CCCCCC')),
                ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ]))
            flows += [tab,Spacer(1,12)]
        else:
            raise ValueError(block[0])
    return flows

def height(flows):
    return sum(f.wrap(WIDTH,10000)[1] for f in flows)

def footnotes(page):
    lines=[]
    for n in page['refs']:
        short,_,url=REFS[n]
        if url.startswith('http'):
            txt=f'[{n}] <link href="{escape(url,quote=True)}" color="#333333">{escape(short)}</link>'
        else:
            txt=f'[{n}] {escape(short)}（完整来源见文末）'
        lines.append(Paragraph(txt,style('note',8.1,11.2)))
    return lines

def render_pdf():
    PDF_PATH.parent.mkdir(parents=True,exist_ok=True)
    canvas=Canvas(str(PDF_PATH),pagesize=A4,pageCompression=1)
    canvas.setTitle(TITLE)
    canvas.setSubject('独立合作课题研究报告：恐惧—安全学习、关联记忆与闭环智能体')
    metrics=[]
    for i, pg in enumerate(PAGES):
        notes=footnotes(pg)
        noteheight=height(notes)
        bottom=34+noteheight+(17 if notes else 0)
        available=PAGE_H-TOP-bottom
        for size in [12.0,11.7,11.4,11.1,10.7]:
            flows=flows_for(pg,size,first=i==0)
            used=height(flows)
            if used<=available: break
        if used>available:
            raise RuntimeError(f'Page {i+1} overflow {used:.1f}>{available:.1f}')
        canvas.bookmarkPage(f'section{i+1}')
        canvas.addOutlineEntry(pg['title'],f'section{i+1}',0)
        y=PAGE_H-TOP
        for flow in flows:
            _,hh=flow.wrap(WIDTH,10000)
            flow.drawOn(canvas,LEFT,y-hh)
            y-=hh
        if notes:
            ny=29+noteheight
            for n in notes:
                _,hh=n.wrap(WIDTH,10000)
                n.drawOn(canvas,LEFT,ny-hh)
                ny-=hh
        metrics.append({'page':i+1,'title':pg['title'],'body_font':size,
                        'used_height':round(used,1),'available_height':round(available,1)})
        canvas.showPage()
    for idx, group in enumerate(REF_GROUPS):
        y=PAGE_H-TOP
        title=f'参考文献与证据来源（{idx+1}/{len(REF_GROUPS)}）'
        ph=Paragraph(title,style('reftitle',17,24))
        _,hh=ph.wrap(WIDTH,10000);ph.drawOn(canvas,LEFT,y-hh);y-=hh+17
        if idx==0:
            pp=Paragraph('正文数字对应本页及后续来源编号。期刊论文、数据记录和内部工程证据分别注明；所有网页于本次研究中核验，未读取的原始文件已明确标记。链接可点击。',style('intro',10.1,16))
            _,hh=pp.wrap(WIDTH,10000);pp.drawOn(canvas,LEFT,y-hh);y-=hh+16
        for n in group:
            short,full,url=REFS[n]
            canvas.bookmarkHorizontalAbsolute(f'ref{n}',y)
            if url.startswith('http'):
                link=f'<link href="{escape(url,quote=True)}" color="#333333">{escape(url)}</link>'
            else:
                link=escape(url)
            pp=Paragraph(f'[{n}] {escape(full)}<br/>{link}',style('reference',9.6,14.9))
            _,hh=pp.wrap(WIDTH,10000)
            if y-hh<35: raise RuntimeError(f'References page {idx+1} overflows')
            pp.drawOn(canvas,LEFT,y-hh);y-=hh+16
        metrics.append({'page':len(PAGES)+idx+1,'title':title,'bottom_y':round(y,1)})
        canvas.showPage()
    canvas.save()
    (Path(__file__).parent/'layout_audit.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))

def render_md():
    lines=['# '+TITLE,'']
    def convert(txt): return re.sub(r'\[(\d+)\]',r'[^\1]',txt)
    for pg in PAGES:
        lines+=['## '+pg['title'],'']
        for b in pg['blocks']:
            if b[0] in ('h','p'):
                lines += [('### ' if b[0]=='h' else '')+convert(b[1]),'']
            elif b[0]=='table':
                _,head,rows,_=b
                lines += ['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']
                lines += ['| '+' | '.join(convert(s) for s in row)+' |' for row in rows]
                lines+=['']
    lines+=['## 参考文献与证据来源','']
    for n,(short,full,url) in REFS.items():
        lines += [f'[^{n}]: {full} [{short}](<{url}>).','']
    MD_PATH.write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    render_md()
    render_pdf()
    print(json.dumps({'markdown':str(MD_PATH),'pdf':str(PDF_PATH),'pages':len(PAGES)+len(REF_GROUPS)},ensure_ascii=False))
