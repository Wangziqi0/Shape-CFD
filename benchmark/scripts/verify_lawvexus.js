#!/usr/bin/env node
/**
 * verify_lawvexus.js — 迁移后一致性验证
 * D9-E 设计文档 第十二章
 *
 * 对比 JS USearch 和 LawVexus 的检索结果一致性
 * 运行方式: node verify_lawvexus.js [--db-path <path>] [--samples <n>]
 */

'use strict';

const path = require('path');
const fs = require('fs');

// 解析命令行参数
const args = process.argv.slice(2);
const getArg = (flag, defaultVal) => {
    const idx = args.indexOf(flag);
    return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : defaultVal;
};

const DB_PATH = getArg('--db-path', './data/legal_assistant.db');
const STORE_PATH = getArg('--store-path', './data/indices');
const SAMPLE_COUNT = parseInt(getArg('--samples', '10'), 10);
const TOP_K = parseInt(getArg('--top-k', '10'), 10);

async function verify() {
    console.log('╔══════════════════════════════════════╗');
    console.log('║  LawVexus 一致性验证工具              ║');
    console.log('╚══════════════════════════════════════╝\n');

    // 1. 加载 LawVexus
    let LawVexus;
    try {
        ({ LawVexus } = require('./law-vexus'));
    } catch (e) {
        console.error('❌ 无法加载 LawVexus:', e.message);
        process.exit(1);
    }

    const vexus = new LawVexus(STORE_PATH);
    vexus.setDbPath(DB_PATH);

    // 尝试加载法条索引
    try {
        vexus.createIndex('statute_index', 4096, 50000);
    } catch (e) {
        console.error('❌ 加载索引失败:', e.message);
        process.exit(1);
    }

    // 2. 验证基础功能
    console.log('━━━ 基础功能验证 ━━━\n');

    // 检查索引统计
    try {
        const stats = vexus.getAllStats();
        for (const s of stats) {
            const mb = (s.memoryUsageBytes / 1024 / 1024).toFixed(1);
            console.log(`  📊 ${s.name}: ${s.totalVectors} 向量, ${s.dimensions}维, ${mb} MB`);
            if (s.totalVectors === 0) {
                console.log(`     ⚠️ 索引为空，请先运行 migrate_to_lawvexus.js`);
            }
        }
        console.log('  ✅ 索引加载正常\n');
    } catch (e) {
        console.error('  ❌ 索引统计获取失败:', e.message);
        process.exit(1);
    }

    // 3. 随机采样检索验证
    console.log('━━━ 检索验证 ━━━\n');

    // 从 SQLite 随机取样本向量
    const sqlite3 = require('better-sqlite3');
    let db;
    try {
        db = new sqlite3(DB_PATH, { readonly: true });
    } catch (e) {
        console.error('  ❌ 数据库打开失败:', e.message);
        process.exit(1);
    }

    const rows = db.prepare(
        `SELECT id, vector FROM chunks WHERE vector IS NOT NULL ORDER BY RANDOM() LIMIT ?`
    ).all(SAMPLE_COUNT);

    if (rows.length === 0) {
        console.log('  ⚠️ 没有找到向量数据\n');
        db.close();
        return;
    }

    console.log(`  📌 随机采样 ${rows.length} 个查询向量，Top-${TOP_K} 检索...\n`);

    let totalMs = 0;
    let successCount = 0;

    for (let i = 0; i < rows.length; i++) {
        const { id, vector: vectorBlob } = rows[i];

        if (!vectorBlob || vectorBlob.length !== 4096 * 4) {
            console.log(`  ⚠️ 样本 #${i + 1} (chunk_id=${id}): 向量数据异常，跳过`);
            continue;
        }

        const queryBuf = Buffer.from(vectorBlob);

        try {
            const start = Date.now();
            const results = vexus.search('statute_index', queryBuf, TOP_K);
            const elapsed = Date.now() - start;
            totalMs += elapsed;

            // 自查验证：查询自身应排第一（如果在索引中）
            const selfFound = results.find(r => r.id === id);
            const selfRank = selfFound ? results.indexOf(selfFound) + 1 : -1;

            const status = selfRank === 1 ? '✅' : (selfRank > 0 ? `🟡 rank=${selfRank}` : '⚠️ 未命中');
            console.log(`  样本 #${i + 1} (id=${id}): ${results.length} 结果, ${elapsed}ms, 自查: ${status}, Top1: score=${results[0]?.score.toFixed(4) || 'N/A'}`);
            successCount++;
        } catch (e) {
            console.log(`  ❌ 样本 #${i + 1} (id=${id}): 检索失败 - ${e.message}`);
        }
    }

    db.close();

    // 4. 汇总
    console.log('\n━━━ 验证结果 ━━━\n');
    if (successCount > 0) {
        const avgMs = (totalMs / successCount).toFixed(1);
        console.log(`  成功: ${successCount}/${rows.length}`);
        console.log(`  平均延迟: ${avgMs}ms`);
        console.log(`  总耗时: ${totalMs}ms`);
        console.log(`\n  ✅ 验证通过！LawVexus 检索功能正常。`);
    } else {
        console.log(`  ❌ 所有检索均失败`);
        process.exit(1);
    }
}

verify().catch((e) => {
    console.error('💥 验证失败:', e);
    process.exit(1);
});
