// Gives renderer.js the same window.inspector API the Electron preload did, backed by the Tauri commands in src-tauri.
{
  const { invoke, convertFileSrc } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;
  // The version query makes the window reload an annotated image after a correction redraws it.
  const withImage = (result) => (result && result.imagePath
    ? { ...result, imageUrl: `${convertFileSrc(result.imagePath)}?v=${result.version}` }
    : result);

  window.inspector = {
    status: () => invoke('status'),
    pick: (kind) => invoke('pick', { kind }),
    run: (options) => invoke('run', { options }),
    correct: (jsonPath, changes) => invoke('correct', { jsonPath, changes }).then(withImage),
    cancel: () => invoke('cancel'),
    open: (target) => invoke('open', { target }),
    on: (channel, callback) => listen(channel, (event) => callback(channel === 'result' ? withImage(event.payload) : event.payload)),
  };
}
