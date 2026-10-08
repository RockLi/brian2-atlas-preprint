const $=id=>document.getElementById(id);
const LABELS={right:['→','右移'],left:['←','左移'],up:['↑','上移'],down:['↓','下移'],looming:['◎','扩大'],receding:['⊙','缩小'],static:['●','静止'],flicker:['✧','闪烁'],blank:['—','空白'],flash:['✦','闪光'],bright:['☀','亮场'],dark:['◐','暗场']};
const COLORS={T4:'#d4f680',T5:'#75d8cb',LC4:'#f1b37b'};
let index,current=null,paired=null,kind='right',condition='intact',bin=25,playing=false,lastTime=0,elapsed=0,epoch=0,busy=false,cache=new Map();
const num=n=>Number(n).toLocaleString('en-US');
function decode(text,type=Uint8Array){const bytes=Uint8Array.from(atob(text),v=>v.charCodeAt(0));return new type(bytes.buffer);}
function unpack(trial){return {...trial,frames:decode(trial.frames_u8),strength:decode(trial.strength_u8),counts:decode(trial.input_counts_u16,Uint16Array)};}
async function json(url,options){const response=await fetch(url,options);if(!response.ok){let m;try{m=(await response.json()).error;}catch{m=`HTTP ${response.status}`;}throw new Error(m);}return response.json();}
async function getTrial(c,k){const key=c+'/'+k;if(cache.has(key))return cache.get(key);const raw=await json('/api/trial?'+new URLSearchParams({condition:c,kind:k}));const value=unpack(raw);cache.set(key,value);return value;}
function setPlaying(value){playing=value;lastTime=0;$('play').textContent=value?'暂停':'播放';$('play').setAttribute('aria-label',value?'暂停时间轴':'播放时间轴');}
function controls(){document.querySelectorAll('#kinds button').forEach(b=>{b.disabled=busy;b.setAttribute('aria-pressed',String(b.dataset.kind===kind));});$('condition').disabled=busy;$('compute').disabled=busy||!index?.live;$('play').disabled=!current;}
function rates(trial,family){if(!trial)return [];const names=trial.summary.group_names;const selected=names.map((n,i)=>n.startsWith(family)?i:-1).filter(i=>i>=0);const total=selected.reduce((s,i)=>s+trial.summary.group_sizes[names[i]],0);return trial.summary.group_counts.map(row=>selected.reduce((s,i)=>s+row[i],0)/(total*.01));}
function decoding(){
  const pilot=index?.pilot,classification=current?.classification;
  $('decoderStatus').textContent=pilot?(pilot.test_clips?'独立留出 ':'开发验证 ')+((pilot.test_clips?pilot.readouts.neural.test.accuracy:pilot.readouts.neural.development_validation_accuracy)*100).toFixed(1)+'%':'尚未训练';
  $('prediction').textContent='—';$('directionScores').replaceChildren();
  if(!pilot){$('decoderNote').textContent='尚无经过验证的方向读出器。';return;}
  $('decoderNote').textContent=`${pilot.fit_clips} 个训练片段 / ${pilot.validation_clips} 个开发验证片段`+(pilot.test_clips?` / ${pilot.test_clips} 个独立留出片段（${pilot.independent_test_groups} 组轨迹）。`:`（${pilot.independent_validation_groups} 组验证轨迹）。`)+ '预测使用完整 400 ms 刺激期；拖动时间轴不会重算方向。只训练外部读出，分数不是概率。接近分类器尚未训练。';
  if(!['right','left','up','down'].includes(kind)){$('prediction').textContent='仅适用于四方向';return;}
  if(!classification)return;
  $('prediction').textContent=LABELS[classification.direction][0]+' '+LABELS[classification.direction][1];
  classification.directions.forEach((direction,i)=>{const row=document.createElement('div');row.className='direction-score';const label=document.createElement('span');label.textContent=LABELS[direction][1];const value=document.createElement('span');value.textContent=classification.scores[i].toFixed(2);row.append(label,value);$('directionScores').append(row);});
}
function study(){
  const report=index?.pilot;
  if(!report?.test_clips)return;
  $('directionStudy').hidden=false;
  const pct=value=>(100*value).toFixed(1)+'%';
  $('studyScope').textContent=`${report.test_clips} 条留出片段 · ${report.independent_test_groups} 组轨迹`;
  $('confusionNote').textContent=`行是真实方向，列是预测方向；每行 ${report.test_clips/4} 条独立留出片段。`;
  $('studySummary').textContent=(report.selected_variant?'各层均使用外部线性读出；输入与读出方案只在开发数据上选择。':'在各层使用相同的线性读出方法。')+`训练 ${report.fit_clips} 条、验证 ${report.validation_clips} 条后锁定读出器，再运行从未参与选参的留出轨迹。每组四方向配对，亮暗各半；四分类机会水平为 25%。`;
  if(report.paired_improvement){
    const recipe=report.selected_cells?`训练集筛选 ${report.selected_cells} 个响应细胞`:{raw_standard:'原始下游计数',spatial12_standard:'按视觉位置汇总',fisher128_raw:'训练集筛选 128 个响应细胞'}[report.readouts.neural.recipe]||report.readouts.neural.recipe;
    $('studyChange').hidden=false;
    $('studyChange').textContent=`本轮改动：外部输入增益 ${report.input_weight_mv} mV（原方案 16 mV）；${report.input_mode==='contrast'?'亮暗对比度编码':'相邻帧亮暗变化编码'}；${recipe}。内部连接与动力学参数保持不变。`;
    const compared=$('pairedComparison');compared.hidden=false;compared.replaceChildren();
    for(const [name,label] of [['old_neural','旧模型 · 原读出'],['readout_only','旧模型 · 改进读出'],['neural','本轮选定方案']]){const card=document.createElement('div'),title=document.createElement('span'),value=document.createElement('strong'),note=document.createElement('small');if(!report.readouts[name])continue;title.textContent=report.comparison_labels?.[name]||label;value.textContent=pct(report.readouts[name].test.accuracy);note.textContent='同一批 '+report.test_clips+' 条新片段';card.append(title,value,note);compared.append(card);}
    const delta=report.paired_improvement.old_neural,detail=document.createElement('p');detail.className='small paired-note';detail.textContent=`相对同批上一版：${delta.accuracy_difference>=0?'+':''}${(delta.accuracy_difference*100).toFixed(1)} 个百分点；95% 配对轨迹组重采样区间 ${delta.group_bootstrap_95_interval.map(v=>(v*100).toFixed(1)).join('–')} 个百分点。`;compared.append(detail);
  }
  const stages=[['encoded_input','01 / 外部脉冲','图像编码后，实际送入网络的信号'],['input_neurons','02 / Mi1 与 Tm1','输入细胞自身的真实神经放电'],['neural','03 / T4、T5 与 LC4','下游真实放电；上方预测使用这一读出']];
  const path=$('pathResults');path.replaceChildren();
  for(const [name,title,note] of stages){
    const result=report.readouts[name].test,card=document.createElement('article');card.className='path-card';
    const heading=document.createElement('h3');heading.textContent=title;
    const metric=document.createElement('strong');metric.textContent=pct(result.accuracy);
    const bar=document.createElement('meter');bar.min=0;bar.max=1;bar.value=result.accuracy;bar.setAttribute('aria-label',title+' 留出准确率');
    const detail=document.createElement('p');detail.className='small';detail.textContent=`${result.correct} / ${result.clips} 正确 · `+(result.correct===result.clips?'本次全对不代表所有新轨迹都正确':`95% 轨迹组重采样区间 ${result.group_bootstrap_95_interval.map(pct).join('–')}`);
    const caption=document.createElement('p');caption.textContent=note;caption.className='small';
    const fitted=document.createElement('p');fitted.className='small';fitted.textContent=`训练 ${pct(report.readouts[name].fit_accuracy)} · 选参验证 ${pct(report.readouts[name].development_validation_accuracy)}`;
    const polar=document.createElement('p');polar.className='small';polar.textContent=`亮刺激 ${pct(result.by_polarity.bright)} · 暗刺激 ${pct(result.by_polarity.dark)}`;
    card.append(heading,metric,bar,detail,polar,fitted,caption);path.append(card);
  }
  const matrix=$('confusion');matrix.replaceChildren();
  const table=document.createElement('table'),caption=document.createElement('caption');caption.textContent='下游方向读出混淆矩阵';table.append(caption);
  const names=['真实 / 预测','右','左','上','下'],head=document.createElement('tr');
  for(const name of names){const th=document.createElement('th');th.textContent=name;head.append(th);}table.append(head);
  report.readouts.neural.test.confusion.forEach((values,i)=>{const row=document.createElement('tr'),label=document.createElement('th');label.textContent=names[i+1];row.append(label);values.forEach((value,j)=>{const cell=document.createElement('td');cell.textContent=value;cell.className=i===j?'correct-cell':'';row.append(cell);});table.append(row);});matrix.append(table);
  const controls=$('studyControls');controls.replaceChildren();
  for(const [name,label] of [['neural_types','按细胞亚型汇总'],['neural_time_sum','全刺激期求和，去掉时间窗'],['labels_shuffled','训练方向标签打乱']]){const row=document.createElement('div');row.className='study-control';const text=document.createElement('span');text.textContent=label;const value=document.createElement('b');value.textContent=pct(report.readouts[name].test.accuracy);row.append(text,value);controls.append(row);}
  $('studyLimits').textContent=`本节范围：固定速度、对比度和背景；${report.independent_test_groups} 组独立留出轨迹。区间为描述性估计，尚未校正多种读出的比较。本节未包含跨条件、统计匹配重连或接近分类评估，不能据此归因于真实连接结构。`;
}

function robustness(){
  const report=index?.audit;if(!report||report.status!=='complete')return;
  $('robustnessStudy').hidden=false;
  $('auditScope').textContent=`每条件 ${report.clips_per_condition} 条 · ${report.groups_per_condition} 组新轨迹`;
  $('auditSummary').textContent='固定同一个 32 细胞读出器，直接测试全部新条件；没有根据这些成绩重新训练或选择模型。右移单独列示。原条件本批结果与上次 87.5% 属于不同轨迹集。';
  const names={reference:'原条件 · 本批基线',slow:'慢速 · 0.5×',fast:'快速 · 1.25×',position:'位置偏移 · 半径 0.30',texture:'平滑纹理背景 · ±0.10',gray_low:'较暗背景 · 灰度 0.40',gray_high:'较亮背景 · 灰度 0.60',static_first:'只重复首帧 · 静止',static_last:'只重复末帧 · 静止',scrambled_middle:'保留首末帧 · 中间乱序'};
  const pct=x=>(100*x).toFixed(1)+'%';
  $('auditConditions').replaceChildren();$('auditControls').replaceChildren();
  for(const [name,result] of Object.entries(report.conditions)){
    const r=result.readouts.neural,card=document.createElement('article');card.className='audit-card';
    const title=document.createElement('h3');title.textContent=names[name]||name;
    const score=document.createElement('strong');score.textContent=pct(r.accuracy);
    const detail=document.createElement('p');detail.className='small';detail.textContent=`${r.correct} / ${r.clips} 与标签一致 · 95% 组重采样区间 ${r.group_bootstrap_95_interval.map(pct).join('–')}`;
    const right=document.createElement('p');right.className='small';right.textContent=`右移 ${r.confusion[0][0]} / ${report.clips_per_condition/4} · 亮刺激 ${pct(r.by_polarity.bright)} · 暗刺激 ${pct(r.by_polarity.dark)}`;
    const input=document.createElement('p');input.className='small';input.textContent=`同条件冻结读出：输入细胞 ${pct(result.readouts.input_neurons.accuracy)}，外部脉冲 ${pct(result.readouts.encoded_input.accuracy)}`;
    card.append(title,score,detail,right,input);$(result.source_label_control?'auditControls':'auditConditions').append(card);
  }
}

function metrics(){ decoding();
  $('stimulusLabel').textContent=LABELS[kind][1];
  $('provenance').textContent=current?(current.live?'本次实时仿真 · CPU f64':'真实仿真回放 · CPU f64'):'此条件暂无记录';
  if(!current){for(const id of ['active','inputSpikes','latency'])$(id).textContent='—';$('findingTitle').textContent='等待运行此条件';$('findingText').textContent='点击重新仿真，获得此刺激与通路条件的真实响应。';return;}
  const s=current.summary,names=s.group_names.filter(n=>!['Mi1','Tm1'].includes(n));
  const active=names.reduce((a,n)=>a+s.active_during_stimulus[n],0),total=names.reduce((a,n)=>a+s.group_sizes[n],0);
  $('active').textContent=`${num(active)} / ${num(total)}`;$('inputSpikes').textContent=num(current.input_spikes);$('latency').textContent=s.simulation_wall_seconds.toFixed(2)+' s';
  $('findingTitle').textContent=condition==='cut'?'输入仍在，输出通路已切断。':'从视觉输入追踪到下游活动。';
  $('findingText').textContent=condition==='cut'?'仅将输入细胞的输出传递设为零。细胞自身仍可放电，背景活动仍然保留；应与同样切断条件的空白刺激比较。':`刺激期间，T4 / T5 / LC4 监测集合中有 ${num(active)} 个细胞放电。活动存在不等于已经识别方向，也不等于真实连接结构带来优势。`;
  $('comparisonNote').textContent=paired?($('reference').value==='blank'?'虚线显示同一通路条件下的空白刺激响应。':`虚线显示同一刺激的${condition==='intact'?'切断通路':'完整通路'}响应。`):'当前还没有此刺激的配对记录。可切换条件后重新仿真。';
}
function clearCanvas(canvas){const c=canvas.getContext('2d');c.setTransform(1,0,0,1,0,0);c.clearRect(0,0,canvas.width,canvas.height);return c;}
function prepareCanvas(canvas){const box=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);const c=clearCanvas(canvas);c.setTransform(dpr,0,0,dpr,0,0);return {c,w:box.width,h:box.height};}
function drawStimulus(){const c=clearCanvas($('stimulus')),w=384,h=384;c.fillStyle='#808080';c.fillRect(0,0,w,h);const t=bin*10,f=Math.floor((t-100)/10);if(current&&f>=0&&f<40){const tmp=document.createElement('canvas');tmp.width=tmp.height=48;const tc=tmp.getContext('2d'),img=tc.createImageData(48,48),start=f*48*48;for(let i=0;i<48*48;i++){const v=current.frames[start+i];img.data.set([v,v,v,255],i*4);}tc.putImageData(img,0,0);c.imageSmoothingEnabled=false;c.drawImage(tmp,0,0,w,h);}$('phase').textContent=t<100?'预热 · 无视觉刺激':t<500?'刺激呈现中':'尾段 · 刺激已撤去';}
function drawEye(){const canvas=$('eye'),{c,w,h}=prepareCanvas(canvas),scale=Math.min(w/640,h/460);const mode=$('eyeMode').value,t=bin*10,f=Math.floor((t-100)/10);c.font='9px sans-serif';c.fillStyle='#7d8d78';c.textAlign='center';c.fillText('DORSAL',w/2,20);c.fillText('VENTRAL',w/2,h-9);c.save();c.translate(15,h/2);c.rotate(-Math.PI/2);c.fillText('ANTERIOR',0,0);c.restore();c.save();c.translate(w-12,h/2);c.rotate(Math.PI/2);c.fillText('POSTERIOR',0,0);c.restore();if(!index)return;
  index.channels.forEach((row,i)=>{const p=row.p-19,q=row.q-17,x=w/2+Math.sqrt(3)/2*(q-p)/22*(w*.42),y=h/2-(p+q)/2/22*(h*.42);const mi=row.cell_type==='Mi1',dx=(mi?-2.1:2.1)*scale;let value=0;if(current)value=mode==='spikes'?Math.min(1,current.counts[bin*index.channels.length+i]/2):(f>=0&&f<40?current.strength[f*index.channels.length+i]/255:0);c.globalAlpha=.14+value*.86;c.fillStyle=mi?'#d4f680':'#75d8cb';c.beginPath();c.arc(x+dx,y,(1.7+value*1.6)*scale,0,Math.PI*2);c.fill();});c.globalAlpha=1;
  $('eyeCaption').textContent=mode==='spikes'?'10 ms 内实际脉冲计数':(index?.manifest.input_mode==='temporal_difference_x4'?'帧间亮暗变化驱动强度':'图像对比度驱动强度');
}
function drawResponse(){const canvas=$('response'),{c,w,h}=prepareCanvas(canvas),L=34,R=w-24,T=15,B=h-28;c.fillStyle='#1d2b1c';c.fillRect(L+(R-L)/6,T,(R-L)*4/6,B-T);if(!current)return;const families=['T4','T5','LC4'],data=families.map(f=>rates(current,f)),other=$('compare').checked&&paired?families.map(f=>rates(paired,f)):[];const max=Math.max(1,...data.flat(),...other.flat())*1.12;
  c.font='10px sans-serif';c.fillStyle='#91a18a';c.textAlign='right';for(let i=0;i<=3;i++){const y=B-(B-T)*i/3;c.strokeStyle='#344332';c.beginPath();c.moveTo(L,y);c.lineTo(R,y);c.stroke();c.fillText((max*i/3).toFixed(max<3?1:0),L-10,y+5);}c.textAlign='left';c.fillText('Hz',6,15);c.textAlign='center';for(let t=0;t<=600;t+=100)c.fillText(t+' ms',L+(R-L)*t/600,B+26);
  const line=(values,color,dash)=>{c.strokeStyle=color;c.lineWidth=dash?2:3;c.setLineDash(dash?[6,5]:[]);c.globalAlpha=dash ? 0.55 : 1;c.beginPath();values.forEach((v,i)=>{const x=L+(R-L)*(i+.5)/60,y=B-(B-T)*v/max;i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();c.globalAlpha=1;c.setLineDash([]);};
  families.forEach((f,i)=>{if(other.length)line(other[i],COLORS[f],true);line(data[i],COLORS[f],false);});const x=L+(R-L)*(bin+.5)/60;c.strokeStyle='#e8f1dc99';c.lineWidth=1;c.beginPath();c.moveTo(x,T);c.lineTo(x,B);c.stroke();
}
function drawHeatmap(){const canvas=$('heatmap'),{c,w,h}=prepareCanvas(canvas);if(!current)return;const names=current.summary.group_names.filter(n=>!['Mi1','Tm1'].includes(n)),indices=names.map(n=>current.summary.group_names.indexOf(n));const L=42,R=w-10,T=8,rowH=25,cellW=(R-L)/60;const all=indices.flatMap((idx)=>current.summary.group_counts.map(row=>row[idx]/current.summary.group_sizes[current.summary.group_names[idx]]/.01));const max=Math.max(1,...all);c.font='10px sans-serif';names.forEach((name,i)=>{c.fillStyle='#a9b69f';c.textAlign='right';c.fillText(name,L-12,T+i*rowH+17);for(let b=0;b<60;b++){const value=current.summary.group_counts[b][indices[i]]/current.summary.group_sizes[name]/.01;const a=Math.sqrt(value/max);c.fillStyle=`rgb(${Math.round(28+174*a)},${Math.round(43+189*a)},${Math.round(28+81*a)})`;c.fillRect(L+b*cellW,T+i*rowH,Math.ceil(cellW)-1,rowH-3);}});c.strokeStyle='#ffffffaa';c.lineWidth=1.2;c.strokeRect(L+bin*cellW,T-2,cellW,names.length*rowH-1);c.fillStyle='#91a18a';c.font='10px sans-serif';c.textAlign='right';c.fillText(`颜色范围 0–${max.toFixed(1)} Hz / neuron`,R,h-9);}
function draw(){drawStimulus();drawEye();drawResponse();drawHeatmap();$('timeline').value=bin;$('time').textContent=`${bin*10} ms / 600 ms`;}
async function select(){const request=++epoch;setPlaying(false);current=paired=null;bin=25;controls();metrics();draw();$('error').textContent='';try{const value=await getTrial(condition,kind);if(request!==epoch)return;current=value;try{const other=await getTrial($('reference').value==='blank'?condition:(condition==='intact'?'cut':'intact'),$('reference').value==='blank'?'blank':kind);if(request!==epoch)return;paired=other;}catch{}if(request!==epoch)return;metrics();draw();controls();}catch(e){if(request!==epoch)return;$('comparisonNote').textContent='此条件尚无已保存的结果。';$('computeStatus').textContent=index.live?'可立即运行一次新的完整网络仿真。':e.message;controls();}}
async function compute(){if(busy)return;busy=true;const requestedKind=kind,requestedCondition=condition;setPlaying(false);controls();$('error').textContent='';$('computeStatus').textContent='正在计算完整网络响应，请稍候…';const started=performance.now();try{const result=unpack(await json('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:requestedKind,condition:requestedCondition})}));cache.set(requestedCondition+'/'+requestedKind,result);await select();$('computeStatus').textContent=`新的仿真已完成，页面等待 ${(performance.now()-started)/1000|0} 秒。`;}catch(e){$('error').textContent=e.message;$('computeStatus').textContent='计算未完成，可重试。';}finally{busy=false;controls();}}
$('play').addEventListener('click',()=>{if(current)setPlaying(!playing);});$('timeline').addEventListener('input',()=>{setPlaying(false);bin=Number($('timeline').value);draw();});$('eyeMode').addEventListener('change',drawEye);$('compare').addEventListener('change',drawResponse);$('reference').addEventListener('change',select);$('condition').addEventListener('change',()=>{condition=$('condition').value;select();});$('compute').addEventListener('click',compute);
function animate(now){if(playing&&current){if(lastTime)elapsed+=now-lastTime;const frameTime=10/Number($('speed').value);if(elapsed>=frameTime){bin=(bin+Math.floor(elapsed/frameTime))%60;elapsed%=frameTime;draw();}lastTime=now;}requestAnimationFrame(animate);}requestAnimationFrame(animate);window.addEventListener('resize',draw);
try{index=await json('/api/index');$('causalLink').hidden=!index.causal_available;$('refinementLink').hidden=!index.refinement_available;$('multispeedLink').hidden=!index.multispeed_available;$('generalizationLink').hidden=!index.generalization_available;$('backend').textContent=index.manifest.backend;$('neuronCount').textContent=num(index.manifest.neurons);$('edgeCount').textContent=num(index.manifest.edges)+' 条加权连接';$('notice').textContent='真实全图仿真 · 选择刺激，沿时间轴查看 Mi1 / Tm1 与 T4 / T5 / LC4 的响应。可切换输出切断条件进行配对比较。';$('computeStatus').textContent=index.live?'可在本机重新运行完整网络仿真。':'当前为已保存的真实仿真回放。';$('about').textContent=`模型步长 ${index.manifest.config.dt_ms} ms；每段视频从同一初始状态开始，帧间保持神经状态。网络内部权重保持固定；方向读出与其评估范围单独列示。`;for(const [k,[icon,label]] of Object.entries(LABELS)){const b=document.createElement('button');b.type='button';b.dataset.kind=k;b.setAttribute('aria-label',label);b.innerHTML=`<span class="icon" aria-hidden="true">${icon}</span>${label}`;b.addEventListener('click',()=>{kind=k;select();});$('kinds').append(b);}study();robustness();controls();await select();}catch(e){$('error').textContent=e.message;$('notice').textContent='实验数据未能加载。请检查本地服务后刷新。';}
