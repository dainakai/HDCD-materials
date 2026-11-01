const fs=require('fs'),vm=require('vm'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'../../../audit/revision-review-20261101');
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.handlers={};this._text='';this.dataset={};this.className='';}
  set textContent(t){this._text=String(t);this.children=[];}
  get textContent(){return this._text+this.children.map(x=>x.textContent).join(' ');}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this._text='';this.children=items;}
  setAttribute(k,v){this.attributes[k]=v;}
  addEventListener(k,f){this.handlers[k]=f;}
  querySelector(selector){const wanted=selector.slice(1);const queue=[...this.children];while(queue.length){const x=queue.shift();if(x.className.split(' ').includes(wanted))return x;queue.push(...x.children);}return null;}
  closest(){return this.parent||null;}
}
const nodes=Object.fromEntries(['print-report','comparison-output','full-progress','full-render-count'].map(k=>[k,new Element('div')]));
const buttons=['phase','rule','rank','z'].map(k=>{const e=new Element('button');e.dataset.comparison=k;return e;});
const mobile=new Element('a');mobile.parent={open:true};
let prints=0;const intervals=[];
const context={document:{querySelector:q=>nodes[q.slice(1)]||null,querySelectorAll:q=>q==='[data-comparison]'?buttons:q==='.mobile-toc a'?[mobile]:[],createElement:t=>new Element(t)},
  fetch:async name=>({ok:true,json:async()=>JSON.parse(fs.readFileSync(path.join(root,'public',name),'utf8'))}),
  window:{print:()=>prints++},setInterval:fn=>intervals.push(fn),console,Date,Number,Set,Error};
const code=fs.readFileSync(path.join(__dirname,'report.js'),'utf8');
(async()=>{
 vm.runInNewContext(code,context);await new Promise(setImmediate);await new Promise(setImmediate);
 for(const button of buttons){button.handlers.click();if(button.attributes['aria-pressed']!=='true')throw Error('button not selected');if(!nodes['comparison-output'].textContent.includes('固定入力対照'))throw Error('missing control disclosure');}
 nodes['print-report'].handlers.click();if(prints!==1)throw Error('print');mobile.handlers.click();if(mobile.parent.open)throw Error('mobile toc');
 const progress=JSON.parse(fs.readFileSync(path.join(root,'public/evidence/full-progress.json')));
 if(nodes['full-render-count'].textContent!==Number(progress.paired_holograms_verified).toLocaleString('ja-JP'))throw Error('progress counter');
 if(!nodes['full-progress'].textContent.includes('260745')||!nodes['full-progress'].textContent.includes('Depth-free'))throw Error('missing progress rows');
 if(intervals.length!==1)throw Error('missing refresh interval');await intervals[0]();
 const result={status:'PASS',comparison_buttons:4,print_handler:true,mobile_toc_closes:true,live_progress_sequences:6,auto_refresh_handler:true,
  scope:'DOM-stub JavaScript execution; not a browser rendering or mobile screenshot check.',script_sha256:crypto.createHash('sha256').update(code).digest('hex')};
 fs.writeFileSync(path.join(root,'interaction-verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
})().catch(e=>{console.error(e);process.exitCode=1;});
