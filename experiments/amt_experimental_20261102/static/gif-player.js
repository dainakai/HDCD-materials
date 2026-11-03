/* Playback and GIF disposal composition ported from collisiongifserver/AnimatedGif. */
"use strict";
window.CollisionGifPlayer = class {
  constructor() {
    this.canvas = document.getElementById('movie');
    this.frames = []; this.index = 0; this.playing = true; this.speed = 1;
    this.visible = true; this.source = null; this.generation = 0;
    this.controls = ['play-pause', 'stop-playback', 'playback-speed', 'frame-back', 'frame-next'];
    document.getElementById('play-pause').onclick = () => { this.playing = !this.playing; this.render(); };
    document.getElementById('stop-playback').onclick = () => { this.playing = false; this.index = 0; this.render(); };
    document.getElementById('frame-back').onclick = () => this.step(-1);
    document.getElementById('frame-next').onclick = () => this.step(1);
    document.getElementById('playback-speed').onclick = () => {
      const speeds = [0.25, 0.5, 1, 2, 4];
      this.speed = speeds[(speeds.indexOf(this.speed) + 1) % speeds.length]; this.render();
    };
    document.addEventListener('visibilitychange', () => this.schedule());
    this.render();
  }
  clear() {
    clearTimeout(this.timer); this.controller?.abort(); this.generation++;
    for (const frame of this.frames) frame.canvas.width = frame.canvas.height = 0;
    this.frames = []; this.index = 0; this.source = null; this.canvas.width = this.canvas.height = 1;
    this.render();
  }
  async load(source, label) {
    if (source === this.source) return;
    this.clear(); this.source = source; this.playing = true;
    const generation = this.generation;
    const controller = this.controller = new AbortController();
    const status = document.getElementById('movie-status');
    status.hidden = false; status.textContent = 'GIFを読み込み中';
    this.canvas.setAttribute('aria-label', label);
    try {
      const response = await fetch(source, {signal: controller.signal, cache: 'no-store'});
      if (!response.ok) throw Error(`GIF load failed (${response.status})`);
      const buffer = await response.arrayBuffer();
      if (generation !== this.generation) return;
      const gif = Gifuct.parseGIF(buffer);
      const decoded = Gifuct.decompressFrames(gif, true);
      if (!decoded.length) throw Error('GIFにフレームがありません');
      this.frames = this.compose(decoded, gif.lsd.width, gif.lsd.height);
      status.hidden = true; this.render();
    } catch (error) {
      if (generation !== this.generation || error.name === 'AbortError') return;
      status.textContent = 'GIFを読み込めませんでした。Refreshから再読み込みしてください。';
      this.source = null; this.render();
    }
  }
  clone(source) {
    const canvas = document.createElement('canvas');
    canvas.width = source.width; canvas.height = source.height;
    canvas.getContext('2d').drawImage(source, 0, 0); return canvas;
  }
  compose(frames, width, height) {
    const composite = document.createElement('canvas');
    composite.width = width; composite.height = height;
    const context = composite.getContext('2d');
    const rendered = []; let previous = null, restoreBeforePrevious = null;
    for (const frame of frames) {
      if (previous?.disposalType === 2) {
        context.clearRect(previous.dims.left, previous.dims.top, previous.dims.width, previous.dims.height);
      } else if (previous?.disposalType === 3 && restoreBeforePrevious) {
        context.clearRect(0, 0, width, height); context.drawImage(restoreBeforePrevious, 0, 0);
      }
      const restoreForCurrent = frame.disposalType === 3 ? this.clone(composite) : null;
      const patch = document.createElement('canvas');
      patch.width = frame.dims.width; patch.height = frame.dims.height;
      const patchContext = patch.getContext('2d');
      const patchImage = patchContext.createImageData(frame.dims.width, frame.dims.height);
      patchImage.data.set(frame.patch); patchContext.putImageData(patchImage, 0, 0);
      context.drawImage(patch, frame.dims.left, frame.dims.top);
      rendered.push({canvas: this.clone(composite), delay: frame.delay || 100});
      patch.width = patch.height = 0;
      if (restoreBeforePrevious) restoreBeforePrevious.width = restoreBeforePrevious.height = 0;
      previous = frame; restoreBeforePrevious = restoreForCurrent;
    }
    if (restoreBeforePrevious) restoreBeforePrevious.width = restoreBeforePrevious.height = 0;
    composite.width = composite.height = 0;
    return rendered;
  }
  step(delta) {
    if (!this.frames.length) return;
    this.playing = false; this.index = (this.index + delta + this.frames.length) % this.frames.length; this.render();
  }
  setVisible(visible) { this.visible = visible; this.schedule(); }
  schedule() {
    clearTimeout(this.timer);
    if (!this.playing || !this.visible || document.hidden || this.frames.length < 2) return;
    this.timer = setTimeout(() => {
      this.index = (this.index + 1) % this.frames.length; this.render();
    }, Math.max(10, this.frames[this.index].delay) / this.speed);
  }
  render() {
    const frame = this.frames[this.index];
    if (frame) {
      this.canvas.width = frame.canvas.width; this.canvas.height = frame.canvas.height;
      this.canvas.getContext('2d').drawImage(frame.canvas, 0, 0);
    }
    this.controls.forEach(id => document.getElementById(id).disabled = !frame);
    document.getElementById('frame-position').textContent = `${frame ? this.index + 1 : 0} / ${this.frames.length}`;
    document.getElementById('playback-speed').textContent = `${this.speed}×`;
    document.getElementById('pause-icon').hidden = !this.playing;
    document.getElementById('play-icon').hidden = this.playing;
    const toggle = document.getElementById('play-pause');
    toggle.title = this.playing ? '一時停止' : '再生'; toggle.setAttribute('aria-label', toggle.title);
    this.schedule();
  }
};
