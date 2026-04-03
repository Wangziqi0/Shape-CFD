#!/usr/bin/env node
'use strict';
const fs = require('fs'), path = require('path'), readline = require('readline');
const Database = require('better-sqlite3');

const dir = process.argv[2];
if (!dir) { console.error('Usage: node build_clouds.js <dataset_dir>'); process.exit(1); }

const corpusPath = path.join(dir, 'corpus_vectors.jsonl');
const cloudsPath = path.join(dir, 'clouds.sqlite');
const idMapPath = path.join(dir, 'id_map.json');

async function main() {
  console.log(`Building clouds.sqlite from ${corpusPath}`);
  if (fs.existsSync(cloudsPath)) fs.unlinkSync(cloudsPath);

  const db = new Database(cloudsPath);
  db.pragma('journal_mode = WAL');
  db.exec(`CREATE TABLE chunks (id INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, chunk_text TEXT, vector BLOB)`);
  db.exec('CREATE INDEX idx_file_id ON chunks(file_id)');

  const insert = db.prepare('INSERT INTO chunks (file_id, chunk_text, vector) VALUES (?, ?, ?)');
  const idMap = {};
  let intId = 0, totalChunks = 0;

  const rl = readline.createInterface({ input: fs.createReadStream(corpusPath), crlfDelay: Infinity });

  let batch = [];
  const flush = db.transaction((docs) => {
    for (const doc of docs) {
      const sents = doc.sentences || [];
      if (sents.length === 0) continue;
      idMap[doc._id] = intId;
      for (const sent of sents) {
        const buf = Buffer.alloc(sent.length * 4);
        for (let i = 0; i < sent.length; i++) buf.writeFloatLE(sent[i], i * 4);
        insert.run(intId, null, buf);
        totalChunks++;
      }
      intId++;
    }
  });

  for await (const line of rl) {
    if (!line.trim()) continue;
    try { batch.push(JSON.parse(line)); } catch(e) { continue; }
    if (batch.length >= 200) {
      flush(batch); batch = [];
      process.stdout.write(`\r  ${intId} docs, ${totalChunks} chunks`);
    }
  }
  if (batch.length > 0) flush(batch);

  console.log(`\n  Total: ${intId} docs, ${totalChunks} chunks`);
  fs.writeFileSync(idMapPath, JSON.stringify(idMap));
  console.log(`  id_map.json: ${Object.keys(idMap).length} entries`);
  db.close();
  console.log(`  clouds.sqlite: ${(fs.statSync(cloudsPath).size / 1e6).toFixed(1)} MB`);
}
main().catch(e => { console.error(e); process.exit(1); });
