import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
import {fileURLToPath, pathToFileURL} from 'node:url';

// The runner copies this module into a fresh ignored build directory.
const BUILD=path.dirname(fileURLToPath(import.meta.url));
const ROOT=process.env.SLIDES_ROOT;
const SKILL=process.env.PRESENTATIONS_SKILL_DIR;
const PY=process.env.SLIDES_PYTHON;
if (!ROOT || !SKILL || !PY) throw new Error('Run slides/build.py with the documented runtime options.');
const {finalizePresentation, applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const DOI='https://doi.org/10.5281/zenodo.23273257';
const C={navy:'#163D53',ink:'#233742',green:'#247968',blue:'#2D6486',orange:'#B46D37',gray:'#667680',light:'#E4EBEF',white:'#FFFFFF'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
p.theme.defaultFont='Arial';
let idx=0;
function text(s,str,x,y,w,h,size=27,color=C.ink,bold=false){
 const z=s.shapes.add({geometry:'textbox',name:`text-${s.id}-${s.shapes.items.length}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=str;z.text.style={typeface:'Arial',fontSize:size,color,bold,insets:0,verticalAlignment:'top',lineSpacing:1.10};return z;
}
function slide(title,source,notes='',dark=false){
 const s=p.slides.add();idx++;s.background.fill=dark?C.navy:C.white;
 if(title)text(s,title,60,44,1155,110,44,dark?C.white:C.navy,true);
 text(s,`${idx.toString().padStart(2,'0')}`,1190,671,42,24,16,dark?'#BCD1DC':C.gray);
 if(source)text(s,source,60,665,1110,37,16,dark?'#BCD1DC':C.gray);
 s.speakerNotes.text=`Source: Xinjun Li, preprint v2, ${DOI}\n${source}\n\n${notes}`;
 return s;
}
function prose(s,heading,body,x,y,w=510){text(s,heading,x,y,w,42,29,C.green,true);text(s,body,x,y+55,w,130,27);}
function table(s,values,x,y,width,height,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:widths,values});
 t.borders.assign({fill:C.light,width:1,style:'solid'});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?C.navy:C.white;cell.text.style={typeface:'Arial',fontSize:r===0?23:24,bold:r===0,color:r===0?C.white:C.ink,insets:12,verticalAlignment:'middle'};
 }
 return t;
}
function chart(s,type,cats,series,position,opts={}){
 const z=s.charts.add(type,{position,categories:cats,series:series.map(x=>({smooth:false,...x})),lineOptions:{smooth:false},hasLegend:series.length>1,
  legend:{position:'bottom',overlay:false,textStyle:{fontSize:21,fill:C.ink}},
  chartFill:C.white,plotAreaFill:C.white,
  xAxis:{textStyle:{fontSize:21,fill:C.ink},line:{fill:C.gray,width:1},majorGridlines:null},
  yAxis:{min:0,textStyle:{fontSize:20,fill:C.gray},majorGridlines:{fill:C.light,width:1},numberFormatCode:'0.0'},
  dataLabels:{showValue:type==='bar',position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:23,fill:C.ink}},
  ...opts});
 applyPresentationChartFont(z,{fontFamily:'Arial'});return z;
}
async function image(s,file,x,y,w,h,alt){const b=await fs.readFile(file);s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt,position:{left:x,top:y,width:w,height:h},fit:'contain'});}
for(const n of [1,7])await sharp(path.join(BUILD,`assets/fig${n}.svg`),{density:180}).png().toFile(path.join(BUILD,`assets/fig${n}.png`));
const meta=await sharp(path.join(BUILD,'assets/fig7.png')).metadata();
// Preserve Figure 7b's complete workbench panel, excluding the separate profile diagram.
await sharp(path.join(BUILD,'assets/fig7.png')).extract({left:Math.round(meta.width*0.035),top:Math.round(meta.height*0.250),width:Math.round(meta.width*0.93),height:Math.round(meta.height*0.742)}).png().toFile(path.join(BUILD,'assets/workbench.png'));

const f9 = await sharp(path.join(BUILD,'assets/fig9.svg'),{density:180}).png().toBuffer();
const f9meta = await sharp(f9).metadata();
await sharp(f9).extract({left:0,top:0,width:f9meta.width,height:Math.floor(f9meta.height*0.568)}).png().toFile(path.join(BUILD,'assets/fig9-panel-a.png'));

// 01
let s=slide('', 'Preprint v2, 10 October 2026. CC BY 4.0',
 'Opening, about 45 seconds. Introduce Atlas as an execution architecture for admitted Brian2 models. This is a preprint and has not been peer reviewed.',true);
text(s,'brian2-atlas',64,60,1100,70,59,C.white,true);
text(s,'A Unified Intermediate Representation\nand Execution Architecture for\nHeterogeneous and Distributed\nNeural Simulation',64,174,1135,300,48,C.white,true);
text(s,'Xinjun Li\nNext Brain',64,526,700,88,28,'#D1E3EB');

// 02
s=slide('Execution semantics travel with the model','Preprint v2, Introduction and Section 2',
 'About 60 seconds. A common modeling language alone does not supply a common execution contract. Explain a delayed spike that must survive continuation and an RNG identity that should not change merely because a partition changes. The study does not measure developer productivity.');
prose(s,'Order changes observations','A threshold, summed update or linked read can observe a different state after reordering.',60,186);
prose(s,'Partitions change ownership','Distributed execution must define where state lives and how spikes reach its owner.',680,186);
prose(s,'Time extends across runs','Delayed events and mutable state must survive supported continuation boundaries.',60,416);
prose(s,'Randomness needs identity','Global identities keep supported random streams tied to the model instance.',680,416);

// 03
s=slide('Atlas owns the downstream execution paths','Preprint v2, Section 2.1 and Figure 1',
 'About 75 seconds. Brian2 retains model objects, equations, units, symbol resolution and abstract update statements. One-time initialization may use Brian NumPy code objects. Atlas replaces the simulation loop and implements its own execution paths. CUDA still requires NVIDIA compilation and driver tools. The diagram does not imply automatic placement across backend families. Figure 1 reproduced from the paper, CC BY 4.0.');
await image(s,path.join(BUILD,'assets/fig1.png'),52,150,794,492,'Paper Figure 1: Brian2 frontend, AtlasIR validation and independent execution paths');
text(s,'One Device integration',888,192,332,66,30,C.green,true);
text(s,'Brian2 model preparation\nAtlasIR validation\nTarget-specific execution',888,280,332,168,27);
text(s,'No Brian2CUDA,\nBrian2GeNN or GeNN\nruntime dependency',888,466,332,126,27,C.navy,true);

// MPI architecture
s=slide('MPI distributes one model across machines','Preprint v2, Section 3.1 and Sections 5.2, 5.10',
 'About 70 seconds. Distribution is a central architectural contribution, without claiming MPI itself or this scale is unprecedented. Neurons belong to one rank, incoming synapses to the target owner. Ranks may run across hosts. Static shards and procedural construction have distinct contracts. Distribution does not establish ideal strong scaling.');
text(s,'Partitioned storage',60,185,280,42,30,C.green,true);
text(s,'Each rank holds local neuron state and incoming connections.\nSynapses belong to the rank owning the target neuron.',350,184,850,79,28);
text(s,'Distributed build',60,296,280,42,30,C.green,true);
text(s,'Ranks build fixed-total connectivity from compact recipes,\navoiding full recurrent-edge materialization in one Python process.',350,295,850,85,28);
text(s,'Ordered events',60,407,280,42,30,C.green,true);
text(s,'Global neuron and edge identities survive partitioning.\nOrdered spike exchange drives local connections and delays.',350,406,850,79,28);
text(s,'Capacity across hosts',60,518,280,77,30,C.green,true);
text(s,'Ranks can span hosts, distributing state and connection storage.\nCommunication, global metadata and recording still limit scaling.',350,517,850,85,28);

// Heterogeneous placement
s=slide('Heterogeneous execution with a backend per rank','Preprint v2, Section 5.9 and Figure 9a. Cross-host mixed-vendor combinations remain unqualified.',
 'About 75 seconds. Figure 9a shows the architecture, not an executed cross-host mixed-vendor setup. CPU, Metal and CUDA are explicit alternatives per rank. The deployment requires compatible OS/CPU ABI and MPI. Cross-host rank placement does not establish arbitrary Mac/Linux interoperability, automatic balancing or mixed-mode speedup. Local CPU+Metal simulation and multi-host CPU capacity are separately qualified.');
await image(s,path.join(BUILD,'assets/fig9-panel-a.png'),60,155,1160,359,'Figure 9a: CPU or GPU neuron-update backend assigned per MPI rank');
text(s,'Device selection and rank placement',60,527,565,38,28,C.green,true);
text(s,'Explicit CPU, Metal or CUDA per rank.\nRanks may span compatible hosts.',60,577,565,70,24);
text(s,'Qualification and deployment boundary',675,527,550,38,28,C.green,true);
text(s,'Multi-host CPU and local CPU+Metal are tested.\nCompatible OS/CPU ABI and MPI are required.',675,577,550,77,24);

// 11
s=slide('The multi-area cortical model completed at full size','Preprint v2, Section 5.2. Frozen 32-area cohort, χ = 1.9, seed 1729.',
 'About 70 seconds. This is the macaque multi-area capacity experiment. 4,129,924 neurons and 24,126,516,728 recurrent synapses, with 254 populations and 8,344 projections. Four nodes and 32 ranks complete 100.5 s of model time in 39,301.456712 s. Proxy peaks are 178.5–184.8 billion bytes per node. Later Rust/NEST runs add descriptive evidence but do not establish prospective biological equivalence or a matched NEST speed ratio.');
text(s,'4.13M',60,193,510,100,78,C.green,true);text(s,'neurons across 32 cortical areas',64,307,520,68,30);
text(s,'24.13B',650,193,566,100,78,C.green,true);text(s,'recurrent synapses',654,307,556,68,30);
text(s,'4 nodes / 32 ranks',60,437,557,56,34,C.navy,true);text(s,'100.5 s model time\n10 h 55 min wall time',60,507,543,110,29);
text(s,'A capacity and output-audit result',650,433,560,89,33,C.navy,true);
text(s,'Biological equivalence and matched\nNEST performance remain unestablished.',650,538,560,95,27);

// 12
s=slide('Weak scaling reaches 860 million neurons','Preprint v2, Section 5.10 and Table 6. One timing observation per size.',
 'About 75 seconds. Hosts 3/6/12/24/30, with eight physical worker cores per host. 86/172/344/688/860 million neurons, exactly 1,000N connections, reference-f64, dt 0.1 ms and 100 ms model time. Launch excludes preparation, deployment and independent audit. Local simulation maxima may occur on different ranks than other stage maxima. Descriptive weak efficiencies are 0.355 simulation and 0.529 launch. These are synthetic recurrent E/I networks, not anatomical fractions of a human brain.');
chart(s,'scatter',['3','6','12','24','30'],[
 {name:'Launch to exit',xValues:[3,6,12,24,30],values:[605.101,663.992,793.267,1049.728,1142.852],line:{fill:C.blue,width:4},marker:{symbol:'square',size:8,fill:C.blue}},
 {name:'Maximum local simulation',xValues:[3,6,12,24,30],values:[236.931,297.882,412.833,612.796,666.834],line:{fill:C.green,width:4},marker:{symbol:'circle',size:8,fill:C.green}}
],{left:50,top:178,width:810,height:445},{scatterOptions:{style:'lineWithMarkers'},xAxis:{min:0,max:30,majorUnit:6,title:'Hosts, eight physical worker cores each',textStyle:{fontSize:21}},yAxis:{min:0,max:1200,majorUnit:300,title:'Time (s)',numberFormatCode:'0',textStyle:{fontSize:20},majorGridlines:{fill:C.light,width:1}}});
text(s,'860B',899,207,300,87,65,C.green,true);text(s,'connections at 30 hosts',899,304,310,79,28);
text(s,'100 ms',899,426,305,67,43,C.navy,true);text(s,'model time\nSynthetic recurrent E/I',899,505,318,93,27);

// 04
s=slide('AtlasIR separates definition, instance and run','Preprint v2, Section 2.2',
 'About 60 seconds. Each layer has an identity. The wire format uses canonical JSON, SHA-256 hashes and exact numeric encodings. The independent validator reconstructs types, dimensions and effects. Hashes establish identity and integrity, not scientific correctness. Compatibility rules still govern compiled-artifact reuse.');
table(s,[['Layer','What it identifies'],['Definition','Populations, update program, schedules, clocks and numerical contracts'],['Instance','Initial state, parameters, topology, pending events and seed'],['Run','Absolute start time, duration and clock tick intervals']],60,182,1160,326,[240,920]);
text(s,'Independent validation checks types, dimensions, events and declared effects.',60,558,1130,65,29,C.green,true);

// 05
s=slide('A linked read makes update order observable','Preprint v2, Section 2.3 and Figure 2',
 'About 60 seconds. Weight increments feed a summed destination, and another population reads that destination through a linked variable. Moving the summed update to the end of the run changes the observed trajectory. These are explanatory deterministic sequences, not measurements with uncertainty.');
chart(s,'line',['0','1','2','3'],[
 {name:'Declared schedule',values:[0,1,3,6],line:{fill:C.green,width:4},marker:{symbol:'circle',size:9,fill:C.green}},
 {name:'Erroneous final-only update',values:[0,0,0,0],line:{fill:C.orange,width:3},marker:{symbol:'square',size:8,fill:C.orange}}
],{left:50,top:188,width:762,height:422},{yAxis:{min:0,max:6,majorUnit:2,title:'Observed x',textStyle:{fontSize:21},majorGridlines:{fill:C.light,width:1}},xAxis:{title:'Start-of-tick sample',textStyle:{fontSize:21}}});
text(s,'[0, 1, 3, 6]',868,220,345,65,45,C.green,true);
text(s,'Completed read sets expose intermediate consumers and constrain optimization.',868,318,330,194,29);

// 06
s=slide('Each result has a defined measurement scope','Preprint v2, Methods and Sections 5.1–5.10',
 'About 45 seconds. The talk separates numerical conformance, timed comparisons, completion/capacity, and bounded functionality. Cohorts retain their actual implementations, inputs, hardware and timing intervals. Do not pool cohort counts or convert capacity to a speed claim.');
table(s,[['Evidence','Question answered','Examples'],['Conformance','Do checked outputs agree?','CPU arrays, event order, continuation'],['Performance','How long is a stated interval?','CPU replay, GPU host-array replay'],['Capacity','Did this configuration complete?','PD14, multi-area cortex, 860M'],['Functionality','Which combinations passed?','Browser, native training, mixed MPI']],60,185,1160,375,[240,445,475]);
text(s,'Qualification applies to the tested combination of model, target and protocol.',60,592,1160,40,26,C.green,true);

// 07
s=slide('CPU full-connectome replay improves on both hosts','Preprint v2, Section 5.3 and Figure 4. Report-derived medians, five repeats.',
 'About 60 seconds. The metric is simulation plus recording, not frontend-inclusive end-to-end time. Linux is restricted to 16 physical cores on one socket of a dual EPYC 9454 machine. Best measured configurations differ: Linux C++ 8 / Atlas 16 threads, M1 Ultra C++ 1 / Atlas 16. Raw samples are absent from the evidence package, so no uncertainty is reconstructed.');
chart(s,'bar',['Linux','M1 Ultra'],[
 {name:'Brian2 C++',values:[4.555,3.873],fill:C.blue},
 {name:'Atlas CPU',values:[0.839,0.767],fill:C.green}
],{left:45,top:188,width:770,height:416},{barOptions:{direction:'column',grouping:'clustered',gapWidth:95},yAxis:{min:0,max:5,title:'Simulation + recording (s)',numberFormatCode:'0',textStyle:{fontSize:20},majorGridlines:{fill:C.light,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{fontSize:24}}});
text(s,'5.43×',881,211,320,88,65,C.green,true);text(s,'Linux median-time ratio',881,307,320,77,27);
text(s,'5.05×',881,410,320,85,60,C.green,true);text(s,'M1 Ultra median-time ratio',881,498,330,75,27);
text(s,'Best measured configurations per backend. Thread counts differ.',60,613,1130,34,22,C.gray);

// 08
s=slide('GPU replay depends on the workload','Preprint v2, Section 5.3 and Figure 5. Delayed-STDP ring, 16,384 neurons.',
 'About 60 seconds. Values are medians of complete reset-to-host-array replays. Rust controls use f64, GPU paths f32. Each pair uses the corresponding host. All four own-GPU policies match complete compiled-f32 control arrays exactly. This comparison does not establish superiority over optimized multicore CPU execution.');
table(s,[['Host / accelerator','Atlas GPU (ms)','Rust f64 control (ms)'],['M3 / Metal','67.19','85.73'],['L4 / CUDA','34.82','216.02'],['A100 / CUDA','32.58','181.80']],60,187,1160,328,[470,345,345]);
text(s,'The replay includes reset, execution and complete host arrays.',60,553,1160,51,29,C.green,true);
text(s,'GPU uses f32. The reference control uses f64 on each corresponding host.',60,608,1160,42,24,C.gray);

// 09
s=slide('Recurrent CUBA exposes a slower CUDA case','Preprint v2, Section 5.3 and Figure 5. A100 host, complete replay medians.',
 'About 55 seconds. Network size is 4,096 neurons, 131,072 connections and 2,048 ticks. Direct GeNN uses the disclosed Brian-Euler adapter. Native CUDA passes the matched f32 numerical gate yet runs slower than GeNN and Rust f64. The f32/f64 diagnostic fails at this scale, so timings do not erase precision boundaries.');
chart(s,'bar',['Atlas CUDA','Direct GeNN','Rust CPU f64'],[{name:'Replay time',values:[137.13,52.11,48.46],fill:C.blue,points:[{idx:0,fill:C.orange},{idx:1,fill:C.blue},{idx:2,fill:C.green}]}],{left:55,top:186,width:765,height:419},{barOptions:{direction:'column',gapWidth:90},yAxis:{min:0,max:160,majorUnit:40,title:'Complete replay (ms)',numberFormatCode:'0',textStyle:{fontSize:20},majorGridlines:{fill:C.light,width:1}}});
text(s,'4,096 neurons\n131,072 connections\n2,048 ticks',870,219,344,178,30,C.navy,true);
text(s,'Passing a numerical gate does not ensure a faster physical plan.',870,432,338,146,28);

// 10
s=slide('298.9 million synapses on a 16 GB laptop','Preprint v2, Section 5.3. PD14 capacity study, M3 laptop, recording disabled.',
 'About 60 seconds. Exact counts are 77,169 neurons and 298,880,968 synapses. Atlas completes 10 seconds of model time. The C++ request was only 0.1 seconds, but preparation exceeded the planned memory budget before compilation. No completed C++ timing denominator exists, and there was no observed kernel OOM kill. Native-child and frontend peaks are separate measurements.');
text(s,'298.9M',60,188,610,105,83,C.green,true);text(s,'synapses',64,296,570,46,32);
text(s,'10 s',60,408,610,99,76,C.navy,true);text(s,'model time completed',64,511,570,62,30);
table(s,[['Measured interval / resource','Atlas'],['Simulation body','104.009 s'],['Frontend + build + run + load','125.586 s'],['Native-child peak','6.314 GB'],['Separate frontend peak','0.319 GB']],653,187,567,365,[389,178]);
text(s,'C++ preparation exceeded its controlled budget before simulation.',653,584,565,67,25,C.orange,true);

// 13
s=slide('Explicit NMDA workloads extend the CPU evidence','Preprint v2, Section 5.8.1 and Table 4. Matched eight-core RK4/f64 cohorts.',
 'About 60 seconds. One second model time, dt 0.1 ms, original delays and monitoring. Ratios compare compiled-region medians within each frozen cohort, including native initialization and dump. At 20,480 neurons, capacity is 754,974,720 synapses. Deterministic exact-input validation extends through 10,240 neurons, with a separate capacity/output scope at 20,480. Native sampled RSS increases about 5–8%. Do not divide accelerator f32 times by this f64 baseline.');
chart(s,'bar',['2,560','5,120','10,240','20,480'],[{name:'Brian2 C++ / Atlas',values:[1.13,1.49,2.10,1.96],fill:C.green}],{left:50,top:181,width:795,height:427},{barOptions:{direction:'column',gapWidth:75},xAxis:{title:'Neurons',textStyle:{fontSize:21}},yAxis:{min:0,max:2.5,majorUnit:0.5,title:'Median-time ratio',numberFormatCode:'0.0',textStyle:{fontSize:20},majorGridlines:{fill:C.light,width:1}}});
text(s,'Up to 2.10×',879,210,340,117,47,C.green,true);
text(s,'Matched eight-core\ncompiled-region timing',879,346,325,105,27);
text(s,'5–8% higher\nsampled native RSS',879,510,329,102,27,C.orange);

// 14
s=slide('Endpoint reuse improves the dendritic network','Preprint v2, Section 5.8.2, Table 5 and Supplement S13. Common-source cohort.',
 'About 65 seconds. The complete-topology network retest covers 100 ms, six measured samples per backend. Cython median 9.796826 s, Atlas default 7.631757 s. A separate same-source cache-policy ablation reduces Atlas runtime by 34.0%. Endpoint reuse preserves edge accumulation order. All 21 paired checks pass and complete native dumps are exact across policies. This protocol does not qualify the entire 45 s scientific workflow.');
chart(s,'bar',['Brian2 Cython','Atlas CPU'],[{name:'Simulation + recording',values:[9.796826,7.631757],fill:C.blue,points:[{idx:0,fill:C.blue},{idx:1,fill:C.green}]}],{left:50,top:194,width:732,height:406},{barOptions:{direction:'column',gapWidth:120},yAxis:{min:0,max:12,majorUnit:3,title:'Time (s)',numberFormatCode:'0',textStyle:{fontSize:20},majorGridlines:{fill:C.light,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:25}}});
text(s,'1.28×',836,210,374,93,63,C.green,true);text(s,'Cython / Atlas median ratio',836,310,373,82,27);
text(s,'34.0% less Atlas runtime',836,438,375,82,34,C.navy,true);text(s,'In the separate cache-policy ablation',836,537,375,86,25);

// 15
s=slide('Browser execution has distinct profiles','Preprint v2, Section 5.5 and Figure 7. Workbench screenshot from Figure 7b.',
 'About 65 seconds. The screenshot is a generic WASM/f64 run: 160 independent adaptive LIF neurons, 600 ms, dt 0.1 ms, seed 42 and 3,706 spikes. Its visible 403 ms is one UI observation, not a benchmark. A separate model-specific AOT application executes 139,255 neurons and 15,091,983 weighted edges. Thirteen fixed-input checks match the CPU oracle with maximum score error 7.4135e-14. This is not an independent test-set accuracy study. WebGPU remains an experimental independent-cell f32 subset. Screenshot reproduced from the paper.');
await image(s,path.join(BUILD,'assets/workbench.png'),60,173,573,470,'Actual Neural Lab browser workbench from Figure 7b');
prose(s,'Generic WASM','Validated bundles and a shared reference runtime power local experiments.',694,183,514);
prose(s,'Separate full-connectome AOT','139,255 neurons and 15.09M weighted edges. Thirteen fixed-input CPU/WASM checks.',694,390,514);
text(s,'WebGPU currently admits a narrower independent-cell f32 profile.',694,590,514,61,22,C.gray);

// 16
s=slide('Training and mixed MPI have bounded acceptance','Preprint v2, Sections 5.7 and 5.9, Supplements S12 and S17',
 'About 60 seconds. The newer Modal L4 run qualifies training increments: 708 formerly skipped cases and one ABI regression, 709 passed with no failures or skips. That covers one physical L4, including two MPI ranks sharing the GPU. It does not qualify CUDA simulation-offload, cross-host mixed vendors or multi-physical-GPU training. The local CPU+Metal simulation example records 82 spikes and dispatch counts [0,32]. Keep these different plan families separate.');
table(s,[['Path','Evidence in the preprint','Scope boundary'],['Native training','709 passed on Modal L4','One GPU, including two shared-GPU ranks'],['CPU + Metal simulation','Local rank ownership and dispatch checks','Neuron-update offload on one host'],['CUDA MPI simulation offload','Source generation and adapter implemented','No NVIDIA execution qualification'],['Cross-host mixed devices\nor multi-GPU training','Further qualification needed','No corresponding performance or scaling claim']],60,184,1160,399,[300,420,440]);
text(s,'Surrogate gradients and continuation contracts do not establish dataset accuracy.',60,607,1150,46,25,C.green,true);

// 17
s=slide('The current evidence has clear limits','Preprint v2, Discussion and target-specific qualification sections',
 'About 55 seconds. Close the scientific argument with boundaries that affect interpretation. Explicit contracts make unsupported behavior visible. They do not remove precision, lifecycle, communication or resource costs. Future work includes stronger scientific acceptance, repeated scaling and broader combinations of model and target.');
prose(s,'Eligibility remains target-specific','A valid AtlasIR document does not guarantee support on every execution target.',60,184);
prose(s,'Performance varies by workload','Precision, dispatch and lifecycle costs can reverse the ordering of backends.',680,184);
prose(s,'Capacity is one acceptance level','Completion at large scale does not establish biological equivalence or a NEST speed ratio.',60,421);
prose(s,'Scaling costs remain global','Global metadata, communication and recording still grow with network size.',680,421);

// 18
s=slide('Public implementation, evidence and preprint','All links are public. Preprint and original figures: CC BY 4.0.',
 'Closing, about 45 seconds. The main takeaway is an explicit execution contract with independent implementations and declared evidence boundaries. Software tag v0.1.0 pins commit ae649a244e28d7d4b3eb5df4e35215e14c1378c7. Evidence tag biorxiv-v1 pins a2f3f8301c7f5da10584aea121d500def38593be. The preprint itself is Zenodo v2; the evidence archive retains its existing tag. Original Atlas code is Apache-2.0 while Brian-derived material retains applicable upstream terms. Consult LICENSE_SCOPE.md.');
text(s,'One model, partitioned across machines\nAtlasIR constrains execution semantics',60,177,1140,132,46,C.navy,true);
text(s,'Preprint v2',60,361,247,41,29,C.green,true);text(s,'doi.org/10.5281/zenodo.23273257',344,361,840,45,29);
text(s,'Software v0.1.0',60,445,275,41,29,C.green,true);text(s,'github.com/RockLi/brian2-atlas\ndoi.org/10.5281/zenodo.23269098',344,439,840,81,26);
text(s,'Evidence archive',60,552,277,41,29,C.green,true);text(s,'github.com/RockLi/brian2-atlas-preprint\ndoi.org/10.5281/zenodo.23269145',344,546,860,81,26);

// 19 backup
s=slide('Backup: native and process-tree memory differ','Preprint v2, Section 5.4 and Figure 6. Historical single-observation 1,000 s cohort.',
 'Use for questions. Full recording versus one-second rolling window reduces native-child RSS, but the process tree can still exceed the C++ baseline because frontend and data representation dominate. This cohort predates the final parallel edge runner. Three fresh-process recovery replays match 41 fields and 11 trajectory segments, which is a distinct correctness result. Units are decimal GB.');
chart(s,'bar',['Rust full','Rust 1 s window','C++'],[
 {name:'Native-child RSS',values:[1.082,0.290,2.327],fill:C.green},
 {name:'Summed process-tree peak',values:[4.961,4.074,2.767],fill:C.blue}
],{left:55,top:188,width:1160,height:414},{barOptions:{direction:'column',grouping:'clustered',gapWidth:90},yAxis:{min:0,max:6,majorUnit:2,title:'Memory (decimal GB)',textStyle:{fontSize:21},majorGridlines:{fill:C.light,width:1}}});
text(s,'A rolling native buffer does not bound all memory in the workflow.',60,615,1150,34,25,C.green,true);

// 20 backup
s=slide('Backup: verified GPU calibration has a cost','Preprint v2, Section 5.6 and Table 3. Three selection samples per candidate.',
 'Use for questions. These are wide-case within-selection medians, not the later five winner replays. All selected outputs pass f32 checks. The wide f64 diagnostic remains failed. Total calibration includes preparation and selection overhead and can greatly exceed one replay. M3 retains baseline because no alternative clears the selection conditions. Decision reuse has separate exact-input evidence and includes compilation/buffer reuse.');
table(s,[['Wide case','Baseline (ms)','Selected (ms)','Calibration (s)'],['L4, prefix + bitset','122.26','79.86','24.36'],['A100, prefix + bitset','154.33','91.75','23.99'],['M3, baseline retained','1,313.67','1,313.67','16.87']],60,191,1160,322,[400,255,255,250]);
text(s,'Selection-round reductions: 34.7% on L4 and 40.6% on A100.',60,558,1150,43,29,C.green,true);
text(s,'Calibration can cost much more than a single replay.',60,611,1150,36,25,C.gray);

await fs.mkdir(path.join(BUILD,'render'),{recursive:true});
for(const [i,sl] of (process.env.SKIP_RENDER ? [] : p.slides.items.entries())){
 if(process.env.RENDER_ONLY && !process.env.RENDER_ONLY.split(',').map(Number).includes(i+1))continue;
 const b=await p.export({slide:sl,format:'png',scale:1.5});
 await fs.writeFile(path.join(BUILD,`render/slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
 console.log(`Rendered ${i+1}/${p.slides.items.length}`);
}
// PDF preview assembled from the inspected slide renders.
if(process.env.RENDER_ONLY && !process.env.FINALIZE) process.exit(0);
const candidate=path.join(BUILD,'candidate.pptx');await (await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(path.join(BUILD,'deck.json'),JSON.stringify(p.toProto()));
await fs.mkdir(path.join(BUILD,'final'),{recursive:true});
await finalizePresentation({workspaceDir:BUILD,candidatePath:candidate,finalPath:path.join(BUILD,'final/validated.pptx'),explicitTotalSlideCount:22,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...[8,10,12,14,18,22].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[8,10,12,14,18,22],requiredNativeChartOwnerSlides:[7,9,11,13,15,16,21],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
console.log('English PowerPoint and slide renders complete.');
