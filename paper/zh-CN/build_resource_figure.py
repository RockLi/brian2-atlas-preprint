"""Render the translated S2 plot from retained arithmetic, without new execution."""
from pathlib import Path
import json,ast,math,os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/b2-preprint-matplotlib')
ROOT=Path(__file__).resolve().parent
source=(ROOT.parent/'scripts/analyze_full_scale_resources.py').read_text()
a=json.loads((ROOT.parent/'data/full_scale/analysis.json').read_text()); measured=a['observed']
GIB=2**30;FULL=86_000_000_000;K=1000;BLOCK=86_000_000;LIMIT=2**31-1
activity=a['assumptions']['endpoint_spikes_per_neuron_per_100ms'];max_host_n=a['assumptions']['maximum_local_host_neurons']
function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='quantities')
exec(compile(ast.Module(body=[function],type_ignores=[]),'<retained quantities>','exec'))
plot=source[source.index("os.environ.setdefault('MPLCONFIGDIR'"):source.index('plt.close(fig)')+len('plt.close(fig)')]
plot=plot.replace("plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})", "from matplotlib import font_manager\nfont_manager.fontManager.addfont('/System/Library/Fonts/Hiragino Sans GB.ttc')\nplt.rcParams.update({'font.family':'Hiragino Sans GB','font.size':11,'svg.fonttype':'none'})")
labels={
'Tested layouts':'已测试布局','Conditional local-work mapping':'条件本地工作量映射','3,000 hosts / 24,000 cores':'3,000台主机／24,000个核',
'a  Fixed local-work host allocation':'a  固定本地工作量的主机分配','Hosts (8 worker cores per host)':'主机数（每台8个工作核）',
'Measured other-host peak':'其他主机实测峰值','Selected arrays (dense CSR)':'选定数组（稠密CSR）','Tested coordinator cap (768 GiB)':'已测协调主机限额（768 GiB）',
'b  Selected worker arrays':'b  选定工作进程数组','Per-host GiB (not a peak forecast)':'每主机GiB（非峰值预测）',
'Observed scientific binary files':'实测科学二进制文件','Conditional binary payload':'条件二进制载荷','Root logical record arrays':'根进程逻辑记录数组',
'c  Scientific recording (100 ms)':'c  科学记录（100 ms）','GiB (excludes provenance reports)':'GiB（不含溯源报告）',
'Derived at tested sizes':'在实测规模下推导','Conditional dense reporting':'条件稠密报告','d  Dense report growth':'d  稠密报告增长',
'Scalar entries per report copy':'每份报告的标量条目','Reference neuron count (%)':'参考神经元数量（%）'}
for old,new in labels.items(): plot=plot.replace(repr(old),repr(new))
exec(compile(plot,'<translated retained plot>','exec'))
(ROOT/'resource_figure_labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n')
print('S2 rendered from retained measured values and unchanged conditional formulas.')
