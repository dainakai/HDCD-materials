"use strict";
document.querySelector('#print').addEventListener('click', () => window.print());
document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const block = document.getElementById(button.dataset.copy);
    const status = document.querySelector('#copy-status');
    try {
      await navigator.clipboard.writeText(block.textContent);
      status.textContent = 'LaTeXをコピーしました。';
    } catch (_) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(block);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = '文章を選択しました。端末のコピー操作を使ってください。';
    }
    window.setTimeout(() => { status.textContent = ''; }, 5000);
  });
});
