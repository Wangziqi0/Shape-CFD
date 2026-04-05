/**
 * BGE-large-en-v1.5 embedding 提取脚本
 * 用硅基流动 API 为 NFCorpus 的 corpus 和 queries 生成 1024d embeddings
 */

const fs = require('fs');
const path = require('path');

const API_URL = 'https://api.siliconflow.cn/v1/embeddings';
const API_KEY = 'sk-zfayhjgxrwhlfeczntxhlsmzubidqoradoklznnxmwunhttc';
const MODEL = 'BAAI/bge-large-en-v1.5';
const BATCH_SIZE = 32;
const CONCURRENCY = 4;
const MAX_CHARS = 512;
const QUERY_PREFIX = 'Represent this sentence for searching relevant passages: ';

const DATA_DIR = '/home/amd/HEZIMENG/legal-assistant/beir_data/nfcorpus';

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function embedBatch(texts, retries = 3) {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const resp = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
          model: MODEL,
          input: texts,
          encoding_format: 'float'
        })
      });

      if (resp.status === 429) {
        console.log(`  Rate limited, waiting 5s... (attempt ${attempt + 1})`);
        await sleep(5000);
        continue;
      }

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`API error ${resp.status}: ${errText}`);
      }

      const data = await resp.json();
      // 硅基流动返回 data.data[i].embedding
      return data.data.map(d => d.embedding);
    } catch (e) {
      if (attempt < retries - 1) {
        console.log(`  Error: ${e.message}, retrying in 3s...`);
        await sleep(3000);
      } else {
        throw e;
      }
    }
  }
}

async function embedWithConcurrency(batches, concurrency) {
  const results = new Array(batches.length);
  let idx = 0;

  async function worker() {
    while (idx < batches.length) {
      const myIdx = idx++;
      results[myIdx] = await embedBatch(batches[myIdx]);
    }
  }

  const workers = [];
  for (let i = 0; i < Math.min(concurrency, batches.length); i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  return results;
}

async function processCorpus() {
  console.log('=== Processing Corpus ===');
  const outPath = path.join(DATA_DIR, 'bge_corpus_vectors.jsonl');

  // 检查是否已经部分完成
  let existingIds = new Set();
  if (fs.existsSync(outPath)) {
    const lines = fs.readFileSync(outPath, 'utf8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      const obj = JSON.parse(line);
      existingIds.add(obj._id);
    }
    console.log(`  Found ${existingIds.size} existing embeddings, resuming...`);
  }

  // 读取 corpus — 只处理在 corpus_vectors.jsonl 中存在的 2473 个文档
  // corpus_vectors.jsonl 太大不能 readFileSync，用 readline 逐行读取 ID
  const corpusVecIds = new Set();
  const readline = require('readline');
  const rl = readline.createInterface({
    input: fs.createReadStream(path.join(DATA_DIR, 'corpus_vectors.jsonl')),
    crlfDelay: Infinity
  });
  for await (const line of rl) {
    if (!line.trim()) continue;
    // 只需要 _id，用正则提取避免解析整行 JSON
    const match = line.match(/"_id"\s*:\s*"([^"]+)"/);
    if (match) corpusVecIds.add(match[1]);
  }

  const corpusLines = fs.readFileSync(path.join(DATA_DIR, 'corpus.jsonl'), 'utf8').trim().split('\n');
  const docs = [];
  for (const line of corpusLines) {
    const obj = JSON.parse(line);
    if (corpusVecIds.has(obj._id) && !existingIds.has(obj._id)) {
      // 用 title + text, 截断到 MAX_CHARS
      const fullText = (obj.title ? obj.title + '. ' : '') + obj.text;
      docs.push({ _id: obj._id, text: fullText.slice(0, MAX_CHARS) });
    }
  }

  console.log(`  Need to embed ${docs.length} documents (${corpusVecIds.size} total, ${existingIds.size} done)`);

  if (docs.length === 0) {
    console.log('  All corpus embeddings already exist.');
    return;
  }

  // 分 batch
  const batches = [];
  const batchDocs = [];
  for (let i = 0; i < docs.length; i += BATCH_SIZE) {
    const slice = docs.slice(i, i + BATCH_SIZE);
    batches.push(slice.map(d => d.text));
    batchDocs.push(slice);
  }

  console.log(`  ${batches.length} batches of up to ${BATCH_SIZE}`);

  // 分组并发处理
  const outStream = fs.createWriteStream(outPath, { flags: 'a' });
  let done = existingIds.size;

  for (let g = 0; g < batches.length; g += CONCURRENCY) {
    const groupBatches = batches.slice(g, g + CONCURRENCY);
    const groupDocs = batchDocs.slice(g, g + CONCURRENCY);

    const results = await embedWithConcurrency(groupBatches, CONCURRENCY);

    for (let b = 0; b < results.length; b++) {
      const vectors = results[b];
      const docSlice = groupDocs[b];
      for (let j = 0; j < vectors.length; j++) {
        outStream.write(JSON.stringify({
          _id: docSlice[j]._id,
          vector: vectors[j],
          sentences: [vectors[j]]  // BGE 是单向量，sentences 就是 [vector] 以兼容格式
        }) + '\n');
        done++;
      }
    }

    console.log(`  Corpus: ${done}/${corpusVecIds.size} done`);

    if (g + CONCURRENCY < batches.length) {
      await sleep(200);  // 避免限流
    }
  }

  outStream.end();
  console.log(`  Corpus embedding complete: ${outPath}`);
}

async function processQueries() {
  console.log('=== Processing Queries ===');
  const outPath = path.join(DATA_DIR, 'bge_query_vectors.jsonl');

  let existingIds = new Set();
  if (fs.existsSync(outPath)) {
    const lines = fs.readFileSync(outPath, 'utf8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      const obj = JSON.parse(line);
      existingIds.add(obj._id);
    }
    console.log(`  Found ${existingIds.size} existing query embeddings, resuming...`);
  }

  const queryLines = fs.readFileSync(path.join(DATA_DIR, 'queries.jsonl'), 'utf8').trim().split('\n');
  const queries = [];
  for (const line of queryLines) {
    const obj = JSON.parse(line);
    if (!existingIds.has(obj._id)) {
      queries.push({ _id: obj._id, text: obj.text });
    }
  }

  console.log(`  Need to embed ${queries.length} queries (${existingIds.size} done)`);

  if (queries.length === 0) {
    console.log('  All query embeddings already exist.');
    return;
  }

  // BGE query prefix
  const batches = [];
  const batchQueries = [];
  for (let i = 0; i < queries.length; i += BATCH_SIZE) {
    const slice = queries.slice(i, i + BATCH_SIZE);
    batches.push(slice.map(q => QUERY_PREFIX + q.text.slice(0, MAX_CHARS)));
    batchQueries.push(slice);
  }

  console.log(`  ${batches.length} batches of up to ${BATCH_SIZE}`);

  const outStream = fs.createWriteStream(outPath, { flags: 'a' });
  let done = existingIds.size;

  for (let g = 0; g < batches.length; g += CONCURRENCY) {
    const groupBatches = batches.slice(g, g + CONCURRENCY);
    const groupQueries = batchQueries.slice(g, g + CONCURRENCY);

    const results = await embedWithConcurrency(groupBatches, CONCURRENCY);

    for (let b = 0; b < results.length; b++) {
      const vectors = results[b];
      const qSlice = groupQueries[b];
      for (let j = 0; j < vectors.length; j++) {
        outStream.write(JSON.stringify({
          _id: qSlice[j]._id,
          text: qSlice[j].text,
          vector: vectors[j]
        }) + '\n');
        done++;
      }
    }

    console.log(`  Queries: ${done}/${queries.length + existingIds.size} done`);

    if (g + CONCURRENCY < batches.length) {
      await sleep(200);
    }
  }

  outStream.end();
  console.log(`  Query embedding complete: ${outPath}`);
}

async function main() {
  console.log('BGE-large-en-v1.5 Embedding Extraction for NFCorpus');
  console.log(`Model: ${MODEL}, Batch: ${BATCH_SIZE}, Concurrency: ${CONCURRENCY}`);
  console.log();

  await processCorpus();
  console.log();
  await processQueries();

  console.log('\n=== Done ===');

  // 验证
  if (fs.existsSync(path.join(DATA_DIR, 'bge_corpus_vectors.jsonl'))) {
    const lines = fs.readFileSync(path.join(DATA_DIR, 'bge_corpus_vectors.jsonl'), 'utf8').trim().split('\n');
    const first = JSON.parse(lines[0]);
    console.log(`Corpus: ${lines.length} docs, vector dim: ${first.vector.length}`);
  }
  if (fs.existsSync(path.join(DATA_DIR, 'bge_query_vectors.jsonl'))) {
    const lines = fs.readFileSync(path.join(DATA_DIR, 'bge_query_vectors.jsonl'), 'utf8').trim().split('\n');
    const first = JSON.parse(lines[0]);
    console.log(`Queries: ${lines.length} queries, vector dim: ${first.vector.length}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
