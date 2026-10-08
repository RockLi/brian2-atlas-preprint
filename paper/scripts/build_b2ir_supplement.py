"""Build readable figure panels from the retained online workflow captures."""
from pathlib import Path
import argparse,hashlib,json
from io import BytesIO
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'docs/preprint'
record=json.loads((D/'data/b2ir_visualization/capture.json').read_text())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'output/pdf/brian2-atlas-supplementary-figure-s1.pdf')
parser.add_argument('--page-offset',type=int,default=0)
args=parser.parse_args()
OUTPUT=args.output;OUTPUT.parent.mkdir(parents=True,exist_ok=True)
W,H=landscape(A4);margin=15*mm;width=W-2*margin
pdf=canvas.Canvas(str(OUTPUT),pagesize=(W,H))
pdf.setTitle('Supplementary Figure S1 | Recorded activity and B2IR inspection')
pdf.setAuthor('Xinjun Li');pdf.setSubject('brian2-atlas; online workbench; recorded activity and B2IR model draft')
style=ParagraphStyle('caption',fontName='Helvetica',fontSize=9,leading=12,textColor='#25354b')
for i,panel in enumerate(record['panels'],1):
 image=D/panel['image']
 assert hashlib.sha256(image.read_bytes()).hexdigest()==panel['image_sha256']
 im=Image.open(image);x0,y0,x1,y1=panel['crop']
 assert 0<=x0<x1<=im.width and 0<=y0<y1<=im.height
 crop=im.crop(panel['crop']);buffer=BytesIO();crop.save(buffer,format='PNG');buffer.seek(0)
 p=Paragraph(panel['caption'],style);_,ph=p.wrap(width,100*mm)
 available_height=H-2*margin-10*mm-6*mm-ph
 draw_width=min(width,available_height*crop.width/crop.height)
 height=draw_width*crop.height/crop.width;y=H-margin-10*mm-height
 pdf.setFont('Helvetica-Bold',13);pdf.drawString(margin,H-margin,'Supplementary Figure S1 · '+panel['title'])
 pdf.drawImage(ImageReader(buffer),margin+(width-draw_width)/2,y,width=draw_width,height=height)
 assert y-6*mm-ph>=margin-.01
 p.drawOn(pdf,margin,y-6*mm-ph)
 pdf.setFont('Helvetica',8);pdf.drawRightString(W-margin,10*mm,str(args.page_offset+i));pdf.showPage()
pdf.save();print(OUTPUT)
