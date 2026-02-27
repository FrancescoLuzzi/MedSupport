import { pipeline, env } from '@huggingface/transformers';
import { connect } from '@tursodatabase/database-wasm';

// --- CONFIGURATION ---
// 1. Point to your local model folder
env.localModelPath = '/models/';
env.allowRemoteModels = false;
env.allowLocalModels = true;

// 2. SERVICE WORKER CONSTRAINTS: disable multi-threading/proxy
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

// --- Cache setup ---
const MODEL_CACHE = 'model-cache-v1';
const DB_CACHE = 'db-cache-v1';
let lastCacheUpdate: string | null = null;
const supportsCache = self.isSecureContext && typeof caches !== 'undefined';

async function saveCacheTimestamp(ts: string): Promise<void> {
  if (!supportsCache) return;
  const cache = await caches.open(MODEL_CACHE);
  await cache.put('cache-metadata', new Response(ts));
}

async function loadCacheTimestamp(): Promise<string | null> {
  if (!supportsCache) return null;
  const cache = await caches.open(MODEL_CACHE);
  const resp = await cache.match('cache-metadata');
  return resp ? resp.text() : null;
}

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    if (!supportsCache) return;
    await caches.open(MODEL_CACHE);
    await caches.open(DB_CACHE);
    const ts = await loadCacheTimestamp();
    if (ts) {
      lastCacheUpdate = ts;
    } else {
      lastCacheUpdate = new Date().toISOString();
      await saveCacheTimestamp(lastCacheUpdate);
    }
  })());
});

async function cacheFirst(cacheName: string, request: Request): Promise<Response> {
  if (!supportsCache) {
    return fetch(request);
  }
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached && cached.ok) return cached;
  const response = await fetch(request);
  cache.put(request, response.clone());
  return response;
}

let vectorDb: any = null;
async function initVectorDb(): Promise<any> {
  if (vectorDb) return vectorDb;
  const resp = await cacheFirst(DB_CACHE, new Request('/db/vector.db'));
  const buffer = await resp.arrayBuffer();
  const root = await navigator.storage.getDirectory();
  const fileHandle = await root.getFileHandle('vector.db', { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(buffer);
  await writable.close();
  vectorDb = await connect('vector.db');
  return vectorDb;
}

class EmbeddingPipeline {
  static task = 'feature-extraction';
  static model = 'paraphrase-italian-mpnet-med-v2/final';
  private static instance: any = null;
  private static initializationPromise: Promise<any> | null = null;

  static async getInstance(): Promise<any> {
    if (this.instance) {
      return this.instance;
    }
    if (!this.initializationPromise) {
      this.initializationPromise = pipeline(this.task, this.model).then((m) => {
        this.instance = m;
        return m;
      });
    }
    return this.initializationPromise;
  }

  static async dispose(): Promise<void> {
    if (!this.instance) return;
    const toDispose = this.instance;
    this.instance = null;
    this.initializationPromise = null;
    await toDispose.dispose();
  }
}

async function generateEmbedding(text: string): Promise<number[]> {
  const pipe = await EmbeddingPipeline.getInstance();
  const output = await pipe(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Signal worker is ready
console.log('Worker: Starting...');
await EmbeddingPipeline.getInstance();
self.postMessage({ type: 'ready' });

self.onmessage = async (e: MessageEvent) => {
  const { type, id, data } = e.data;
  try {
    if (type === 'generate-embedding') {
      self.postMessage({ type: 'progress', id, message: 'Caricamento modello...' });
      const embedding = await generateEmbedding(data.text);
      self.postMessage({ type: 'result', id, success: true, embedding, model: 'local' });
    } else if (type === 'get-cache-info') {
      self.postMessage({ type: 'cache-info', id, timestamp: lastCacheUpdate });
    } else if (type === 'compare-examples') {
      const embeddings: any[] = [];
      const total = data.examples.length;
      for (let i = 0; i < total; i++) {
        const [t1, t2] = data.examples[i];
        self.postMessage({ type: 'progress', id, message: `Elaborazione ${i + 1}/${total}...` });
        embeddings.push({ emb1: await generateEmbedding(t1), emb2: await generateEmbedding(t2), text1: t1, text2: t2 });
      }
      self.postMessage({ type: 'result', id, success: true, embeddings, model: 'local' });
    } else if (type === 'query-db') {
      const { text, topN } = data;
      self.postMessage({ type: 'progress', id, message: 'Preparazione DB...' });
      const db = await initVectorDb();
      self.postMessage({ type: 'progress', id, message: 'Generazione embedding...' });
      const queryEmb = await generateEmbedding(text);
      self.postMessage({ type: 'progress', id, message: 'Ricerca nel DB...' });
      const sql = `SELECT value,vector_distance_cos(embedding,${queryEmb}) FROM positives LIMIT ${topN}`;
      const table = db.exec(sql);
      const results = table ? table.values.map(([v, s]) => ({ value: v, score: s })) : [];
      self.postMessage({ type: 'query-result', id, success: true, results });
    }
  } catch (err: any) {
    self.postMessage({ type: 'result', id, success: false, error: err.message });
  }
};
