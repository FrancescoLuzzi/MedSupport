import './style.css';

const btnEmbed = document.getElementById('btnEmbed') as HTMLButtonElement;
const btnCompare = document.getElementById('btnCompare') as HTMLButtonElement;
const btnQuery = document.getElementById('btnQuery') as HTMLButtonElement;
const inputText = document.getElementById('inputText') as HTMLTextAreaElement;
const modelSelect = document.getElementById('modelSelect') as HTMLSelectElement;
const topNInput = document.getElementById('topN') as HTMLInputElement;
const loadingSpan = document.getElementById('loading') as HTMLSpanElement;
const progressDiv = document.getElementById('progress') as HTMLDivElement;
const resultsDiv = document.getElementById('results') as HTMLDivElement;
const vectorDisplay = document.getElementById('vectorDisplay') as HTMLDivElement;
const similarityDisplay = document.getElementById('similarityDisplay') as HTMLDivElement;

let worker: Worker | null = null;
let currentRequestId = 0;
const pending = new Map<number, {resolve: (data: any)=>void; reject: (err: Error)=>void}>();

function showProgress(msg: string) {
  progressDiv.textContent = msg;
  progressDiv.style.display = 'block';
}

function showLoading(msg: string) {
  loadingSpan.textContent = msg;
  loadingSpan.style.display = 'inline';
}

async function initWorker() {
  if (worker) return;
  worker = new Worker(new URL('./onnx-worker.ts', import.meta.url), { type: 'module' });
  worker.onerror = (e) => console.error('Worker error', e);
  await new Promise<void>((res) => {
    worker!.onmessage = ({ data }) => {
      if (data.type === 'ready') res();
    };
  });
  worker.onmessage = ({ data }) => {
    const cb = pending.get(data.id);
    if (cb) {
      pending.delete(data.id);
      data.success ? cb.resolve(data) : cb.reject(new Error(data.error));
    } else if (data.type === 'progress') {
      showProgress(data.message);
    }
  };
}

function sendToWorker(type: string, data: any) {
  return new Promise<any>((resolve, reject) => {
    const id = ++currentRequestId;
    pending.set(id, { resolve, reject });
    worker!.postMessage({ type, id, data });
  });
}

async function generateEmbedding() {
  const text = inputText.value.trim();
  if (!text) { alert('Inserisci del testo!'); return; }
  btnEmbed.disabled = true;
  showLoading('Caricamento...');
  await initWorker();
  showProgress('Generazione embedding...');
  const { embedding, model } = await sendToWorker('generate-embedding', { text });
  vectorDisplay.innerHTML = `<h3>Modello:</h3><p><strong>${model}</strong></p>` +
    `<h3>Embedding (primi 10):</h3><div>${embedding.slice(0,10).map((v:number)=>v.toFixed(4)).join(', ')}</div>`;
  similarityDisplay.innerHTML = '';
  resultsDiv.style.display = 'block';
  btnEmbed.disabled = false;
}

async function compareExamples() {
  btnCompare.disabled = true;
  showLoading('Caricamento...');
  await initWorker();
  showProgress('Confronto esempi...');
  const { embeddings } = await sendToWorker('compare-examples', { examples: [] });
  // render table (omitted for brevity)
  btnCompare.disabled = false;
}

async function queryDB() {
  const topN = parseInt(topNInput.value) || 5;
  const text = inputText.value.trim();
  if (!text) { alert('Inserisci del testo!'); return; }
  btnQuery.disabled = true;
  showLoading('Caricamento...');
  await initWorker();
  showProgress('Query DB...');
  const { results } = await sendToWorker('query-db', { text, topN });
  vectorDisplay.innerHTML = '<h3>Risultati DB:</h3><ul>' +
    results.map((r: any)=>`<li>${r.value} <em>(${r.score.toFixed(4)})</em></li>`).join('') +
    '</ul>';
  resultsDiv.style.display = 'block';
  btnQuery.disabled = false;
}

btnEmbed.addEventListener('click', generateEmbedding);
btnCompare.addEventListener('click', compareExamples);
btnQuery.addEventListener('click', queryDB);
