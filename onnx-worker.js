import {
  pipeline,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

// --- CONFIGURATION ---
// 1. Point to your local model folder
env.localModelPath = './models/';
env.allowRemoteModels = false;
env.allowLocalModels = true;

// 2. REQUIRED FOR SERVICE WORKERS:
// Service Workers cannot handle the default multi-threading/proxying of ORT.
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

// ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
// ort.env.wasm.numThreads = 1; // Service Workers cannot spawn sub-workers
// ort.env.wasm.proxy = false;

// ---- Service Worker model caching ----
const MODEL_CACHE = 'model-cache-v1';
// Track cache update timestamp (persisted to cache storage)
let lastCacheUpdate = null;

/**
 * Persist cache update timestamp in cache storage.
 * @param {string} ts ISO timestamp string
 */
async function saveCacheTimestamp(ts) {
  const cache = await caches.open(MODEL_CACHE);
  await cache.put('cache-metadata', new Response(ts));
}

/**
 * Load persisted cache update timestamp from cache storage.
 * @returns {Promise<string|null>} previously saved timestamp or null
 */
async function loadCacheTimestamp() {
  const cache = await caches.open(MODEL_CACHE);
  const resp = await cache.match('cache-metadata');
  return resp ? resp.text() : null;
}

// Ensure model cache exists and record initial timestamp
self.addEventListener('install', event => {
  event.waitUntil((async () => {
    await caches.open(MODEL_CACHE);
    const ts = await loadCacheTimestamp();
    if (ts) {
      lastCacheUpdate = ts;
    } else {
      lastCacheUpdate = new Date().toISOString();
      await saveCacheTimestamp(lastCacheUpdate);
    }
  })());
});

// Cache model file requests under localModelPath
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  const modelPath = env.localModelPath.replace(/^\.\//, '/');
  if (url.pathname.startsWith(modelPath)) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          const clone = response.clone();
          // Cache new model files and record update timestamp
          caches.open(MODEL_CACHE)
            .then(cache => cache.put(event.request, clone))
            .then(() => {
              lastCacheUpdate = new Date().toISOString();
              return saveCacheTimestamp(lastCacheUpdate);
            });
          return response;
        });
      })
    );
  }
});

class EmbeddingPipeline {
  static task = 'feature-extraction';
  static model = 'paraphrase-italian-mpnet-med-v2/final';
  static instance = null;
  static initializationPromise = null

  static async getInstance() {
    if (this.instance) return this.instance;

    if (!this.promise) {
      this.promise = pipeline(this.task, this.model)
        .then(loadedModel => {
          this.instance = loadedModel;
          return loadedModel;
        });
    }
    return this.promise;
  }

  static async dispose() {
    const instanceCopy = this.instance
    this.instance = null
    await instanceCopy.dispose()
  }
}

async function loadModel() {
  return await EmbeddingPipeline.getInstance();
}

async function generateEmbedding(text) {
  const pipe = await loadModel()

  // Transformers.js handles padding and truncation automatically!
  // This prevents the "out of bounds" error caused by manual tokenizing.
  const output = await pipe(text, {
    pooling: 'mean',
    normalize: true
  });

  return Array.from(output.data)
}

// Signal worker is ready
console.log('Worker: Starting...');
await loadModel()
self.postMessage({ type: 'ready' });

// Message handler
// Expose cache info via messages
self.onmessage = async function (e) {
  const { type, id, data } = e.data;
  console.log('Worker received:', type);

  try {
    if (type === 'generate-embedding') {
      self.postMessage({ type: 'progress', id, message: 'Caricamento modello...' });
      const embedding = await generateEmbedding(data.text);
      self.postMessage({ type: 'result', id, success: true, embedding, model: 'local' });
    }
    else if (type === 'get-cache-info') {
      self.postMessage({ type: 'cache-info', id, timestamp: lastCacheUpdate });
    }
    else if (type === 'compare-examples') {
      const embeddings = [];
      const total = data.examples.length;

      for (let i = 0; i < total; i++) {
        const [text1, text2] = data.examples[i];
        self.postMessage({ type: 'progress', id, message: `Elaborazione ${i + 1}/${total}...` });
        embeddings.push({
          emb1: await generateEmbedding(text1),
          emb2: await generateEmbedding(text2),
          text1, text2
        });
      }
      self.postMessage({ type: 'result', id, success: true, embeddings, model: 'local' });
    }
  } catch (error) {
    console.error('Worker error:', error);
    self.postMessage({ type: 'result', id, success: false, error: error.message });
  }
};
