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
