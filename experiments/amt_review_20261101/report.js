"use strict";
document.querySelector("#print-report")?.addEventListener("click", () => window.print());
document.querySelectorAll(".mobile-toc a").forEach(link => link.addEventListener("click", () => {
  const menu = link.closest("details");
  if (menu) menu.open = false;
}));
const box = document.querySelector("#comparison-output");
if (box) {
  fetch("evidence/comparison.json").then(response => {
    if (!response.ok) throw new Error("Comparison data unavailable");
    return response.json();
  }).then(rows => {
    const setups = {
      phase: {text: "ID優先を統一し、学習Z±15 mm、抑制前順位に固定。画像特徴の回数だけを変えた比較。", rows: [["strict_phase12_z15", "before", "画像特徴12回"], ["strict_phase3_z15", "before", "画像特徴3回"]]},
      rule: {text: "画像特徴3回、Z±15 mm、抑制前順位。IDを読む時点、attentionの教師、neutralと重みを合わせた変更。", rows: [["legacy_phase3_z15", "before", "旧教師規則"], ["strict_phase3_z15", "before", "ID優先を統一"]]},
      rank: {text: "同じモデル、同じ生存候補、同じ座標。順位罰則を決める時点だけを変えた比較。", rows: [["strict_phase3_z15", "before", "抑制前に順位固定"], ["strict_phase3_z15", "after", "抑制後に順位再計算"]]},
      z: {text: "ID優先統一、画像特徴3回、抑制前順位。値は学習時の絶対Z許容差。全条件の評価にはZゲートを使わない。", rows: [["strict_phase3_z5", "before", "±5 mm"], ["strict_phase3_z15", "before", "±15 mm"], ["strict_phase3_z30", "before", "±30 mm"], ["strict_phase3_znone", "before", "Z条件なし"]]}
    };
    const element = (tag, value) => { const item = document.createElement(tag); if (value !== undefined) item.textContent = value; return item; };
    function display(key) {
      const setup = setups[key];
      box.replaceChildren(element("p", setup.text));
      const wrap = element("div"); wrap.className = "table-wrap"; wrap.tabIndex = 0; wrap.setAttribute("role", "region"); wrap.setAttribute("aria-label", "比較結果。横スクロール可能");
      const table = element("table"), head = element("thead"), tr = element("tr"), body = element("tbody");
      ["条件", "AP ↑", "FP80 ↓", "FP90 ↓", "最大recall ↑"].forEach(label => { const th = element("th", label); th.scope = "col"; tr.append(th); });
      head.append(tr); table.append(head, body);
      setup.rows.forEach(([arm, rank, label]) => {
        const row = rows.find(x => x.arm === arm && x.rank === rank);
        const line = element("tr");
        [label, Number(row.event_ap).toFixed(6), row["fp_r0.8"], row["fp_r0.9"], Number(row.maximum_recall).toFixed(4)].forEach(value => line.append(element("td", value)));
        body.append(line);
      });
      wrap.append(table); box.append(wrap);
      const note = element("p", "旧102.4 mm画像を使った固定入力対照。最終版の性能値ではありません。"); note.className = "compare-note"; box.append(note);
      document.querySelectorAll("[data-comparison]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.comparison === key)));
    }
    document.querySelectorAll("[data-comparison]").forEach(button => button.addEventListener("click", () => display(button.dataset.comparison)));
    display("phase");
  }).catch(() => { box.textContent = "切替用データを読み込めませんでした。上の表で全比較を確認できます。"; });
}

const progressBox = document.querySelector('#full-progress');
if (progressBox) {
  const el = (tag, text) => { const x=document.createElement(tag); if(text!==undefined)x.textContent=text;return x; };
  const number = value => Number(value).toLocaleString('ja-JP');
  async function refreshProgress() {
    try {
      const response=await fetch('evidence/full-progress.json', {cache:'no-store'});
      if(!response.ok)throw new Error('progress unavailable');
      const data=await response.json();
      const headline=document.querySelector('#full-render-count');
      if(headline)headline.textContent=number(data.paired_holograms_verified);
      const status={running:'計算を実行中',PASS:'一連の計算が完了。図表と本文の最終確認へ',FAILED:'処理のエラーを確認する必要があります'}[data.status]||data.status;
      const current=el('p',`${status} · 更新 ${new Date(data.updated_at).toLocaleString('ja-JP')}`);current.className='progress-status';
      const totals=el('p',`ホログラム保存 ${number(data.paired_holograms_saved)}対 ／ 検証済み ${number(data.paired_holograms_verified)}対、MinIP ${number(data.minip_verified)}枚、計測完了 ${number(data.measurement_frames)}フレーム。`);
      progressBox.replaceChildren(current,totals);
      if(data.version)progressBox.append(el('small',`${data.version.branch} · ${data.version.commit.slice(0,12)}`));
      const tableWrap=el('div');tableWrap.className='table-wrap';tableWrap.tabIndex=0;
      const table=el('table'),head=el('thead'),header=el('tr'),body=el('tbody');
      ['系列','用途','ホログラム検証済み','MinIP','計測完了','3D / 2D追跡'].forEach(t=>{const th=el('th',t);th.scope='col';header.append(th);});head.append(header);
      data.sequences.forEach(s=>{const tr=el('tr');[s.seed,s.split,`${number(s.paired_holograms_verified)} / 6,000`,number(s.minip_verified),number(s.measurement_frames),`${s.tracking_complete?'完了':'待機'} / ${s.minip_tracking_complete?'完了':'待機'}`].forEach(x=>tr.append(el('td',x)));body.append(tr);});
      table.append(head,body);tableWrap.append(table);progressBox.append(tableWrap);
      const names={ours:'Ours',depth_free:'Depth-free',minip2d:'MinIP-based 2D'};
      const results=el('div');results.className='progress-results';
      data.methods.forEach(m=>{const card=el('div');card.append(el('strong',names[m.method]));
        if(m.evaluated){card.append(el('p',`AP ${Number(m.metrics.event_ap).toFixed(6)}`));card.append(el('small',`FP80 ${m.metrics.false_positives_at_recall_0p8 ?? '未到達'} / FP90 ${m.metrics.false_positives_at_recall_0p9 ?? '未到達'}`));}
        else card.append(el('p','再学習・評価の完了待ち'));
        results.append(card);});progressBox.append(results);
      if(data.stage_comparison?.length){
        progressBox.append(el('h3','新条件の段階比較'));
        const stageNames={trajectory:'軌跡の特徴',image:'画像特徴を追加',relation:'関係attentionを追加',suppression:'座標の重複抑制を追加'};
        const wrap=el('div');wrap.className='table-wrap';wrap.tabIndex=0;
        const t=el('table'),h=el('thead'),r=el('tr'),b=el('tbody');
        ['段階','AP','FP80','FP90'].forEach(v=>{const th=el('th',v);th.scope='col';r.append(th);});h.append(r);
        data.stage_comparison.forEach(s=>{const line=el('tr');[stageNames[s.stage]||s.stage,Number(s.event_ap).toFixed(6),s.false_positives_at_recall_0p8??'未到達',s.false_positives_at_recall_0p9??'未到達'].forEach(v=>line.append(el('td',v)));b.append(line);});
        t.append(h,b);wrap.append(t);progressBox.append(wrap,el('small','各段階には時間モード選択・最大5候補・抑制前の順位罰則・同じ学習済み位置補正器を適用。軌跡行の位置補正には画像特徴を渡さず、後続行では候補集合も絞られるため、分類スコアだけを変えた比較ではありません。最終行で座標抑制を追加します。'));
      }
      if(data.completion){
        if(data.completion.finalization_failed)progressBox.append(el('p','数値計算後の成果物整理でエラーが発生しました。結果の固定は未完了です。'));
        if(data.completion.finalized){
          progressBox.append(el('h3','数値検証の完了確認'));
          progressBox.append(el('p','3手法の入力・モデル・評価の照合と図表の整理が完了し、実験レビュー用のモデルを固定しました。'));
          const claims=data.completion.claim_checks||{};
          const titles={ours_exceeds_both_event_ap:'OursのAPが両比較法を上回る',ours_exceeds_both_maximum_f1:'Oursの最大F1が両比較法を上回る',stagewise_nondecreasing_event_ap:'段階追加でAPが単調に向上する',stagewise_nonincreasing_false_positives_at_recall_0p8:'段階追加でFP80が単調に減る',stagewise_nonincreasing_false_positives_at_recall_0p9:'段階追加でFP90が単調に減る'};
          const list=el('ul');Object.entries(titles).forEach(([key,title])=>list.append(el('li',`${title}：${claims[key]===true?'維持できる':claims[key]===false?'そのまま記載できない':'判定できない'}`)));progressBox.append(list);
          progressBox.append(el('p','図表と数値だけの差し替えでは完成しません。順位・教師対応・AP・段階比較・テスト利用履歴の本文修正と、実験由来のFig.3–5の更新が残ります。実験の再処理と新しい目視レビューは未実施です。'));
        }
      }
      if(data.failed_jobs.length)progressBox.append(el('p',`エラーを記録した処理：${data.failed_jobs.join('、')}`));
      const allowed=new Set(['assets/full-figure07.png','assets/full-figure08.png','assets/full-figure09.png','assets/full-comparison.png']);
      for(const item of data.figures){if(!allowed.has(item.path))continue;const fig=el('figure'),link=el('a'),img=el('img');link.href=item.path;link.target='_blank';link.rel='noopener';img.src=item.path;img.alt=item.label;img.loading='lazy';link.append(img);fig.append(link,el('figcaption',item.label+'（新条件の計算結果）'));progressBox.append(fig);}
      progressBox.append(el('small','旧版との比較表にある固定入力対照のAPと、この欄の最終版APは異なる評価です。'));
    } catch(error) {
      if(!progressBox.querySelector('.progress-status'))progressBox.textContent='進行状況を取得できませんでした。ページを再読み込みしてください。';
    }
  }
  refreshProgress();setInterval(refreshProgress,30000);
}
