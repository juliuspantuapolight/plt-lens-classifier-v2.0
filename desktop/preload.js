const { contextBridge, ipcRenderer } = require('electron');

const CHANNELS = ['log', 'total', 'result', 'done', 'worker'];

contextBridge.exposeInMainWorld('inspector', {
  status: () => ipcRenderer.invoke('status'),
  pick: (kind) => ipcRenderer.invoke('pick', kind),
  run: (options) => ipcRenderer.invoke('run', options),
  correct: (jsonPath, changes) => ipcRenderer.invoke('correct', { jsonPath, changes }),
  cancel: () => ipcRenderer.invoke('cancel'),
  open: (target) => ipcRenderer.invoke('open', target),
  on: (channel, callback) => {
    if (CHANNELS.includes(channel)) ipcRenderer.on(channel, (_event, payload) => callback(payload));
  },
});
