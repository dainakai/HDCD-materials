// Handler checks without a browser; never writes real research judgments.
const fs=require('fs'),vm=require('vm');
class Element{
 constructor(){this._value='';this.children=[];this.hidden=false;this.disabled=false;this.checked=false;this.dataset={};this.classList={toggle(){}};}
 set value(v){this._value=String(v);}get value(){return this._value;}
 replaceChildren(...nodes){this.children=nodes;this.value=nodes[0]?.value||'';}
 scrollIntoView(){}
 getContext(){return new Proxy({}, {get:(o,k)=>o[k]||(()=>{}),set:(o,k,v)=>(o[k]=v,true)});}
}
const ids=['processing','scene','progress','bar','waiting','workspace','previous','candidate','next','title','meta','movie','projection','trajectory','current-label','note','advance','feedback','filter','filter-count','first-unreviewed','empty','event-detail','movie-status','open-movie','draft','undo','browse','gallery'];
const nodes=Object.fromEntries(ids.map(k=>[k,new Element()]));nodes.projection.value='xy';nodes.filter.value='all';nodes.advance.checked=true;
const buttons=['collision','non_collision','ambiguous'].map(label=>{const x=new Element();x.dataset.label=label;return x;});
const points=[{frame:0,x:500,y:500,z:400,diameter_px:10},{frame:1,x:501,y:499,z:401,diameter_px:10}];
const events=[0,1].map(i=>({candidate_id:`12:1:2:${i}`,review_rank:i+1,candidate_frame_raw:i,score:.99-i*.1,vanished_track_id:1,survivor_track_id:2,movie_url:'fixture.gif',tracks:[{points},{points}],review:null}));
let failSave=false;const writes=[];
const context={document:{getElementById:id=>nodes[id],querySelector:()=>({content:'fixture-csrf'}),addEventListener(){},createElement:()=>new Element(),querySelectorAll:q=>q==='[data-label]'?buttons:[...buttons,nodes.scene,nodes.candidate,nodes.projection,nodes.previous,nodes.next,nodes.undo,nodes.note,nodes.filter,nodes['first-unreviewed']]},
 fetch:async(path,opts)=>{if(opts.method==='POST'){if(failSave)return{ok:false,status:503};writes.push({path,body:JSON.parse(opts.body),headers:opts.headers});return{ok:true,json:async()=>({saved:true})};}
 return{ok:true,json:async()=>path.endsWith('/api/status')?{processing:{status:'PASS'},scenes:[{scene:12,ready:true,reviewed:writes.length}],budget_total:1120}:{events:structuredClone(events)}};},
 window:{location:{search:''}},sessionStorage:{getItem:()=>null,setItem(){}},URLSearchParams,setInterval:()=>1,console,Number,Math,Error};
(async()=>{
 vm.runInNewContext(fs.readFileSync(__dirname+'/static/review.js','utf8'),context);
 await new Promise(setImmediate);await new Promise(setImmediate);
 if(nodes.title.textContent!=='系列 12 · 候補 1 / 80')throw Error('first candidate not loaded');
 nodes.note.value='保存するメモ';await buttons[0].onclick();
 if(writes.length!==1||writes[0].body.note!=='保存するメモ'||writes[0].headers['X-CSRF-Token']!=='fixture-csrf')throw Error('incorrect save payload');
 if(nodes.title.textContent!=='系列 12 · 候補 2 / 80')throw Error('auto advance failed');
 failSave=true;nodes.note.value='失敗しても残すメモ';await buttons[2].onclick();
 if(nodes.note.value!=='失敗しても残すメモ')throw Error('unsaved note lost');
 if(!nodes.feedback.textContent.includes('メモは保持'))throw Error('save error not shown');
 failSave=false;nodes.previous.onclick();await nodes.undo.onclick();
 if(!writes.at(-1).path.endsWith('/undo')||nodes['current-label'].textContent!=='未判定')throw Error('undo failed');
 nodes.note.value='移動しても残すメモ';nodes.note.oninput();nodes.next.onclick();nodes.previous.onclick();
 if(nodes.note.value!=='移動しても残すメモ')throw Error('draft lost during navigation');
 nodes.filter.value='collision';nodes.filter.onchange();if(!nodes['event-detail'].hidden)throw Error('empty filter failed');
 nodes.filter.value='unreviewed';nodes.filter.onchange();if(nodes['event-detail'].hidden)throw Error('unreviewed filter failed');
 console.log(JSON.stringify({undo_preserves_navigation:true,draft_retention:true,label_filters:true,status:'PASS',candidate_navigation:true,csrf_payload:true,save_auto_advance:true,failed_save_preserves_note:true,scope:'JavaScript handlers with DOM stubs; no browser rendering and no real labels'}));
})().catch(e=>{console.error(e);process.exitCode=1;});
