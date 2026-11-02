const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
async function exercise(rejectClipboard) {
  const handlers = {}, status = {textContent: ''};
  const block = {textContent: '\\caption{Saved evaluation}'};
  let printed = false, copied = null, selected = false;
  const button = {dataset: {copy: 'test'}, addEventListener: (_, fn) => {handlers.copy = fn;}};
  const context = {
    document: {
      querySelector: s => s === '#print' ? {addEventListener: (_, fn) => {handlers.print = fn;}} : status,
      querySelectorAll: () => [button], getElementById: () => block,
      createRange: () => ({selectNodeContents: b => {assert.equal(b, block); selected = true;}})
    },
    window: {print: () => {printed = true;}, setTimeout: () => {}, getSelection: () => ({removeAllRanges() {}, addRange() {}})},
    navigator: {clipboard: {writeText: async t => {if (rejectClipboard) throw new Error('disabled'); copied = t;}}}
  };
  vm.runInNewContext(fs.readFileSync(__dirname + '/paper.js', 'utf8'), context);
  handlers.print(); assert.equal(printed, true);
  await handlers.copy();
  if (rejectClipboard) {assert.equal(selected, true); assert.match(status.textContent, /端末/);}
  else {assert.equal(copied, block.textContent); assert.match(status.textContent, /コピーしました/);}
}
(async () => {await exercise(false); await exercise(true); console.log('PASS: print, clipboard copy, selection fallback. DOM stubs only; no browser-rendering claim.');})();
