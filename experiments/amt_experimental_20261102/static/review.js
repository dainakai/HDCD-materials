"use strict";
const base = '/experimental-review', $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="csrf-token"]').content;
const names = {collision:'衝突', non_collision:'非衝突', ambiguous:'曖昧'};
const player = new window.CollisionGifPlayer();
let allEvents = [], events = [], index = 0, currentScene = null, busy = false, loadId = 0, view = 'review';
const initialScene = Number(new URLSearchParams(window.location.search).get('scene')) || null;
const draftKey = 'amt-102p3mm-phase3-review-drafts';
let drafts = {};
try { drafts = JSON.parse(sessionStorage.getItem(draftKey) || '{}'); } catch (error) { drafts = {}; }
function persistDrafts() { try { sessionStorage.setItem(draftKey, JSON.stringify(drafts)); } catch (error) {} }
async function api(path, data) {
  const response = await fetch(base + path, {
    method:data === undefined ? 'GET' : 'POST', cache:'no-store',
    headers:data === undefined ? {} : {'Content-Type':'application/json','X-CSRF-Token':token},
    body:data === undefined ? undefined : JSON.stringify(data)
  });
  if (!response.ok) throw Error(`読み込み／保存に失敗しました（${response.status}）。`);
  return response.json();
}
function feedback(message, error = false) {
  $('feedback').textContent = message; $('feedback').hidden = !message;
  $('feedback').classList.toggle('error', error);
}
function controls() {
  ['scene','filter','candidate','first-unreviewed','previous','next','note','refresh','advance'].forEach(id => $(id).disabled = busy);
  document.querySelectorAll('[data-label]').forEach(button => button.disabled = busy || !events.length);
  $('previous').disabled = busy || index === 0 || !events.length;
  $('next').disabled = busy || index >= events.length - 1;
  $('undo').disabled = busy || !events[index]?.review;
  $('undo').hidden = !events[index]?.review;
  $('first-unreviewed').disabled = busy || !allEvents.some(event => !event.review);
}
async function status(refreshCurrent = false) {
  const data = await api('/api/status');
  const previous = $('scene').value || String(initialScene || '');
  const options = data.scenes.map(scene => {
    const option = document.createElement('option'); option.value = scene.scene; option.disabled = !scene.ready;
    option.textContent = `${String(scene.scene).padStart(4,'0')} (${scene.ready ? `${scene.reviewed} / 80 判定済み` : '処理中'})`;
    return option;
  });
  $('scene').replaceChildren(...options);
  const ready = data.scenes.filter(scene => scene.ready);
  const selected = ready.find(scene => String(scene.scene) === previous) || ready[0];
  if (selected) $('scene').value = selected.scene;
  const total = data.scenes.reduce((sum, scene) => sum + scene.reviewed, 0);
  $('progress').textContent = `判定済み ${total} / 1,120件 · レビュー可能 ${ready.length} / 14系列`;
  $('bar').value = total;
  $('processing').textContent = data.processing.status === 'PASS' ? '全系列のレビュー画像が完成しました。' :
    data.processing.status === 'FAILED' ? '処理エラーがあります。完了済み系列はレビューできます。' : '処理が終わった系列から自動で追加されます。';
  $('waiting').hidden = ready.length > 0;
  dashboard(data);
  if (selected && (currentScene !== Number(selected.scene) || refreshCurrent)) {
    await loadScene(Number(selected.scene), refreshCurrent ? events[index]?.candidate_id : null);
  }
}
async function loadScene(number, preferred = null) {
  if (busy) return;
  busy = true; controls(); const generation = ++loadId;
  try {
    const data = await api(`/api/scenes/${number}`);
    if (generation !== loadId) return;
    allEvents = data.events; currentScene = number; $('scene').value = number;
    window.history?.replaceState(null, '', base + '/?scene=' + number);
    refreshView(preferred || allEvents.find(event => !event.review)?.candidate_id || allEvents[0]?.candidate_id);
    feedback('');
  } finally { busy = false; controls(); }
}
function matches(event) {
  const filter = $('filter').value;
  return filter === 'all' || (filter === 'unreviewed' ? !event.review : filter === 'reviewed' ? Boolean(event.review) : event.review?.label === filter);
}
function refreshView(preferred, fallback = 0) {
  events = allEvents.filter(matches);
  const found = events.findIndex(event => event.candidate_id === preferred);
  index = found >= 0 ? found : Math.max(0, Math.min(fallback, events.length - 1));
  queue(); show(); gallery();
}
function queue() {
  const options = events.map((event, i) => {
    const option = document.createElement('option'); option.value = i;
    option.textContent = `#${event.review_rank} · ${event.review ? names[event.review.label] : '未判定'}`; return option;
  });
  $('candidate').replaceChildren(...options); $('candidate').value = index;
  $('queue-total').textContent = events.length; $('browse-total').textContent = events.length;
  $('empty').hidden = events.length > 0 || !currentScene;
  $('event-detail').hidden = events.length === 0;
  $('browse-summary').textContent = currentScene ? `系列 ${currentScene} · 上位80件のうち ${events.length}件を表示` : '完了した系列から表示します。';
}
function show() {
  const event = events[index];
  $('position').textContent = `${event ? index + 1 : 0} / ${events.length}`;
  if (!event) { player.clear(); controls(); return; }
  $('candidate').value = index;
  $('title').textContent = `${String(currentScene).padStart(4,'0')} / #${event.review_rank}`;
  $('title').title = `系列 ${currentScene} · 候補 ${event.review_rank} / 80`;
  $('candidate-id').textContent = event.candidate_id; $('candidate-id').title = event.candidate_id;
  $('score').textContent = Number(event.score).toFixed(5);
  $('frame').textContent = event.candidate_frame_raw;
  $('tracks').textContent = `${event.vanished_track_id} → ${event.survivor_track_id}`;
  $('open-movie').href = event.movie_url;
  player.load(event.movie_url, `系列 ${currentScene} · 候補 ${event.review_rank} / 80`);
  $('note').value = Object.hasOwn(drafts,event.candidate_id) ? drafts[event.candidate_id] : (event.review?.note || '');
  $('draft').hidden = $('note').value === (event.review?.note || '');
  $('current-label').textContent = event.review ? names[event.review.label] : '未判定';
  controls(); draw();
}
function gallery() {
  const container = $('gallery'); container.replaceChildren();
  if (view !== 'browse') return;
  events.forEach(event => {
    const button = document.createElement('button'); button.className = 'browse-tile';
    const media = document.createElement('div'); media.className = 'tile-gif';
    const image = document.createElement('img'); image.src = event.movie_url; image.loading = 'lazy'; image.alt = `候補${event.review_rank}の再構成画像`; media.append(image);
    const detail = document.createElement('div'); detail.className = 'tile-meta';
    const pill = document.createElement('span'); pill.className = 'status-pill ' + (event.review?.label || 'pending'); pill.textContent = event.review ? names[event.review.label] : '未判定';
    const title = document.createElement('strong'); title.className = 'tile-line'; title.textContent = `#${event.review_rank} · ${event.candidate_id}`;
    const score = document.createElement('span'); score.className = 'tile-line muted'; score.textContent = `Score ${Number(event.score).toFixed(5)}`;
    detail.append(pill,title,score); button.append(media,detail);
    button.onclick = () => { if (busy) return; index = events.indexOf(event); show(); setView('review'); };
    container.append(button);
  });
}
function dashboard(data) {
  const ready = data.scenes.filter(scene => scene.ready), reviewed = data.scenes.reduce((sum, scene) => sum + scene.reviewed, 0);
  const count = label => data.scenes.reduce((sum, scene) => sum + (scene.counts?.[label] || 0), 0);
  const cards = [['レビュー可能',ready.length * 80,''],['判定済み',reviewed,''],['衝突',count('collision'),'green'],['非衝突',count('non_collision'),'red'],['曖昧',count('ambiguous'),'']];
  $('stats').replaceChildren(...cards.map(([label,value,tone]) => {
    const card = document.createElement('div'); card.className = `stat ${tone}`;
    const name = document.createElement('span'); name.textContent = label;
    const number = document.createElement('strong'); number.textContent = value; card.append(name,number); return card;
  }));
  const cells = [];
  ['Scene','未判定','衝突','非衝突','曖昧'].forEach(text => { const cell = document.createElement('div'); cell.className = 'scene-head'; cell.textContent = text; cells.push(cell); });
  data.scenes.forEach(scene => {
    const cell = document.createElement('div');
    if (scene.ready) {
      const button = document.createElement('button'); button.className = 'scene-open'; button.textContent = `系列 ${scene.scene}`;
      button.onclick = async () => { if (busy) return; $('filter').value = 'all'; try { await loadScene(scene.scene); setView('review'); } catch(error) { feedback(error.message,true); } }; cell.append(button);
    } else { cell.textContent = `系列 ${scene.scene} · 処理中`; }
    cells.push(cell);
    [scene.ready ? 80 - scene.reviewed : '—', ...['collision','non_collision','ambiguous'].map(label => scene.counts?.[label] || 0)].forEach(value => { const item = document.createElement('div'); item.textContent = value; cells.push(item); });
  });
  $('scene-table').replaceChildren(...cells);
  $('dashboard-progress').textContent = `レビュー可能 ${ready.length} / 14系列 · 各系列上位80件、全1,120件。`;
}
function setView(next) {
  view = next; openFilters(false);
  ['review','browse','dashboard'].forEach(name => { $('tab-'+name).classList.toggle('active',name===view); $('tab-'+name).setAttribute('aria-current',name===view?'page':'false'); });
  $('workspace').hidden = view === 'dashboard'; $('workspace').className = view === 'browse' ? 'browse' : 'review-layout';
  $('filters').classList.toggle('browse-controls',view === 'browse');
  $('review-view').hidden = view !== 'review'; $('browse-view').hidden = view !== 'browse'; $('dashboard-view').hidden = view !== 'dashboard';
  player.setVisible(view === 'review'); gallery();
}
function openFilters(open) {
  $('filters').classList.toggle('open',open); $('filter-scrim').hidden = !open;
  document.body.classList.toggle('filters-open',open);
  $('open-filters').setAttribute('aria-expanded',String(open));
  $('browse-filters').setAttribute('aria-expanded',String(open));
  if (open) $('close-filters').focus();
}
async function save(label, undo = false) {
  if (busy || !events.length) return;
  busy = true; controls();
  const savedIndex = index, event = events[index], number = currentScene, note = $('note').value, next = events[index + 1]?.candidate_id;
  try {
    await api(`/api/scenes/${number}/events/${encodeURIComponent(event.candidate_id)}/${undo?'undo':'review'}`,undo ? {} : {label,note});
    event.review = undo ? null : {label,note}; delete drafts[event.candidate_id]; persistDrafts();
    feedback(undo ? '未判定に戻しました。変更履歴は保持しています。' : '保存しました。');
    const advance = $('advance').checked && !undo;
    if (!advance && !matches(event)) $('filter').value = 'all';
    refreshView(advance ? next : event.candidate_id,savedIndex);
    if (advance && next) $('review-view').scrollIntoView({behavior:'auto',block:'start'});
    if (allEvents.every(value => value.review)) feedback(`系列 ${number} の80件を判定しました。系列を切り替えて続けられます。`);
    await status().catch(() => feedback('判定は保存済みです。進捗の更新は次回に再試行します。'));
  } catch(error) { feedback(error.message + ' メモは保持しています。',true); }
  finally { busy = false; controls(); }
}
$('scene').onchange = () => loadScene(Number($('scene').value)).catch(error => feedback(error.message,true));
$('filter').onchange = () => refreshView(events[index]?.candidate_id);
$('first-unreviewed').onclick = () => { if (busy) return; $('filter').value = 'unreviewed'; refreshView(allEvents.find(event => !event.review)?.candidate_id); openFilters(false); };
$('candidate').onchange = () => { index = Number($('candidate').value); show(); };
$('previous').onclick = () => { if (!busy && index > 0) { index--; show(); } };
$('next').onclick = () => { if (!busy && index < events.length - 1) { index++; show(); } };
$('projection').onchange = draw;
$('note').oninput = () => { const event = events[index]; if (!event) return; if ($('note').value === (event.review?.note || '')) delete drafts[event.candidate_id]; else drafts[event.candidate_id] = $('note').value; persistDrafts(); $('draft').hidden = !Object.hasOwn(drafts,event.candidate_id); };
$('undo').onclick = () => save(null,true);
document.querySelectorAll('[data-label]').forEach(button => button.onclick = () => save(button.dataset.label));
['review','browse','dashboard'].forEach(name => $('tab-'+name).onclick = () => setView(name));
$('open-filters').onclick = $('browse-filters').onclick = () => openFilters(true);
$('close-filters').onclick = $('filter-scrim').onclick = () => { openFilters(false); (view === 'browse' ? $('browse-filters') : $('open-filters')).focus(); };
$('refresh').onclick = async () => { if (busy) return; player.clear(); try { await status(true); } catch(error) { feedback(error.message,true); } };
$('logout').onclick = async () => { try { await api('/api/logout',{}); window.location.assign(base+'/'); } catch(error) { feedback(error.message,true); } };
document.addEventListener('keydown',event => {
  if (event.key === 'Escape') { openFilters(false); return; }
  if (busy || view !== 'review' || ['INPUT','TEXTAREA','SELECT','BUTTON'].includes(event.target.tagName)) return;
  if (event.key === 'ArrowLeft') { $('previous').onclick(); event.preventDefault(); }
  else if (event.key === 'ArrowRight') { $('next').onclick(); event.preventDefault(); }
  else if (['1','2','3'].includes(event.key)) { save(['collision','non_collision','ambiguous'][Number(event.key)-1]); event.preventDefault(); }
});
status().catch(error => feedback(error.message,true));
setInterval(() => { if (!busy) status().catch(error => feedback(error.message,true)); },30000);
function draw(){const c=$('trajectory'),ctx=c.getContext('2d'),e=events[index];ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);if(!e)return;const axes=$('projection').value.split('');const units={x:.01,y:.01,z:.1};const val=(p,a)=>(p[a]-(a==='z'?1:0))*units[a];const all=e.tracks.flatMap(t=>t.points);if(!all.length)return;const bounds=axes.map(a=>{const v=all.map(p=>val(p,a));const lo=Math.min(...v),hi=Math.max(...v),pad=Math.max((hi-lo)*.1,.005);return[lo-pad,hi+pad];});const project=p=>[65+(val(p,axes[0])-bounds[0][0])/(bounds[0][1]-bounds[0][0])*500,395-(val(p,axes[1])-bounds[1][0])/(bounds[1][1]-bounds[1][0])*340];ctx.strokeStyle='#e0e8ed';ctx.font='18px system-ui';ctx.fillStyle='#526976';for(let i=0;i<=4;i++){const x=65+i*125,y=395-i*85;ctx.beginPath();ctx.moveTo(x,55);ctx.lineTo(x,395);ctx.moveTo(65,y);ctx.lineTo(565,y);ctx.stroke();ctx.fillText((bounds[0][0]+(bounds[0][1]-bounds[0][0])*i/4).toFixed(3),x-18,416);ctx.fillText((bounds[1][0]+(bounds[1][1]-bounds[1][0])*i/4).toFixed(3),8,y+4);}ctx.fillText(axes[0].toUpperCase()+' [mm]',280,444);ctx.fillText(axes[1].toUpperCase()+' [mm]',8,35);e.tracks.forEach((t,i)=>{ctx.strokeStyle=i===0?'#bc8100':'#0786ae';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;ctx.beginPath();t.points.forEach((p,j)=>{const [x,y]=project(p);if(j)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();t.points.forEach(p=>{const [x,y]=project(p);ctx.beginPath();ctx.arc(x,y,p.frame===e.candidate_frame_raw?6:2,0,Math.PI*2);ctx.fill();});});}