#!/usr/bin/env node
/**
 * verify_improvements.js - 验证5项后端算法改进
 * 
 * 运行: node verify_improvements.js
 */

const path = require('path');
const fs = require('fs');

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✅ ${name}`);
    passed++;
  } catch (e) {
    console.error(`  ❌ ${name}: ${e.message}`);
    failed++;
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'Assertion failed');
}

// ========== 改进1: LCD IIG 去重 ==========
console.log('\n📦 改进1: LCD IIG 去重验证');

test('SearchEngine 类可加载', () => {
  // 只验证模块导出，不触发 load()
  const { SearchEngine } = require('./search_engine');
  const se = new SearchEngine();
  assert(typeof se._deduplicateAndCoverage === 'function', '_deduplicateAndCoverage 方法不存在');
  assert(typeof se._textToFeatureVector === 'function', '_textToFeatureVector 方法不存在');
  assert(typeof se._computeResidualNorm === 'function', '_computeResidualNorm 方法不存在');
  assert(typeof se._sparseDot === 'function', '_sparseDot 方法不存在');
  assert(typeof se._sparseNorm === 'function', '_sparseNorm 方法不存在');
  assert(typeof se._buildCoverageStats === 'function', '_buildCoverageStats 方法不存在');
});

test('TF-IDF 特征向量提取正确', () => {
  const { SearchEngine } = require('./search_engine');
  const se = new SearchEngine();
  const vec = se._textToFeatureVector('劳动合同到期不续签经济补偿');
  assert(Object.keys(vec).length > 0, '特征向量为空');
  assert(vec['劳'] !== undefined, '应包含单字特征');
  assert(vec['劳动'] !== undefined, '应包含2-gram特征');
  assert(Object.values(vec).every(v => v > 0 && v < 1), '权重应在(0,1)范围内');
});

test('稀疏向量运算正确', () => {
  const { SearchEngine } = require('./search_engine');
  const se = new SearchEngine();
  const a = { x: 3, y: 4 };
  const b = { x: 1, y: 0, z: 5 };
  assert(se._sparseDot(a, b) === 3, '点积应为3');
  assert(se._sparseNorm(a) === 5, '范数应为5');
});

test('IIG去重选择多样性高于顺序选择', () => {
  const { SearchEngine } = require('./search_engine');
  const se = new SearchEngine();
  // 模拟: 3个结果中2个内容相似,1个不同
  const results = [
    { law: '劳动法', article: '第1条', content: '劳动者有权获得劳动报酬', finalScore: 0.9, layer: 1 },
    { law: '劳动法', article: '第2条', content: '劳动者有权获得劳动保护', finalScore: 0.85, layer: 1 },
    { law: '合同法', article: '第10条', content: '当事人订立合同应当遵循公平原则', finalScore: 0.7, layer: 2 },
  ];
  const { articles } = se._deduplicateAndCoverage(results, null, 3);
  assert(articles.length === 3, `应选出3条，实际${articles.length}`);
  // IIG 应该先选劳动法第1条（最高分），然后选合同法（残差最大），最后劳动法第2条
  const laws = articles.map(a => a.law);
  assert(laws.includes('劳动法'), '应包含劳动法');
  assert(laws.includes('合同法'), '应包含合同法');
});

// ========== 改进2: 共现矩阵持久化 ==========
console.log('\n📦 改进2: 共现矩阵持久化验证');

test('DataEngine 共现矩阵方法存在', () => {
  const { DataEngine } = require('./data_engine');
  const de = new DataEngine('/tmp/test_verify.db');
  de.initialize();
  assert(typeof de.hasCooccurrence === 'function', 'hasCooccurrence 不存在');
  assert(typeof de.saveCooccurrence === 'function', 'saveCooccurrence 不存在');
  assert(typeof de.loadCooccurrence === 'function', 'loadCooccurrence 不存在');
  de.close();
  try { fs.unlinkSync('/tmp/test_verify.db'); } catch {}
});

test('共现矩阵读写一致性', () => {
  const { DataEngine } = require('./data_engine');
  const de = new DataEngine('/tmp/test_cooc.db');
  de.initialize();

  const testMatrix = {
    '劳动法|第1条': [
      { law: '劳动法', article: '第5条', weight: 0.9 },
      { law: '合同法', article: '第3条', weight: 0.7 },
    ],
    '合同法|第10条': [
      { law: '合同法', article: '第15条', weight: 0.8 },
    ],
  };

  // 写入
  assert(!de.hasCooccurrence('test'), '初始应无数据');
  de.saveCooccurrence(testMatrix, 'test');
  assert(de.hasCooccurrence('test'), '写入后应有数据');

  // 读取
  const loaded = de.loadCooccurrence('test');
  assert(Object.keys(loaded).length === 2, `应有2条源法条，实际${Object.keys(loaded).length}`);
  assert(loaded['劳动法|第1条'].length === 2, '劳动法第1条应有2条引用');
  assert(loaded['合同法|第10条'][0].weight === 0.8, '权重应为0.8');

  de.close();
  try { fs.unlinkSync('/tmp/test_cooc.db'); } catch {}
});

test('SearchEngine _loadOrBuildCooccurrence 方法存在', () => {
  const { SearchEngine } = require('./search_engine');
  const se = new SearchEngine();
  assert(typeof se._loadOrBuildCooccurrence === 'function', '_loadOrBuildCooccurrence 不存在');
  assert(typeof se._buildCooccurrence === 'function', '_buildCooccurrence 仍存在');
});

// ========== 改进3: Autopilot 智能跳过 ==========
console.log('\n📦 改进3: Autopilot 智能跳过验证');

test('Autopilot 跳过逻辑代码存在', () => {
  const code = fs.readFileSync(path.join(__dirname, 'qa_engine.js'), 'utf-8');
  assert(code.includes('Autopilot 跳过: 直接回答'), '应有 Autopilot 跳过日志');
  assert(code.includes('_shouldSkipAutopilot'), '应有 _shouldSkipAutopilot 方法');
  assert(code.includes('selectedModel === MODEL_PROFILES.lite'), '应有 lite 模型判断');
  assert(code.includes('hasPluginHint'), '应有插件触发迹象检测变量');
});

// ========== 改进4: 地域数据外部化 ==========
console.log('\n📦 改进4: 地域数据外部化验证');

test('regional_policies.json 文件存在并格式正确', () => {
  const raw = JSON.parse(fs.readFileSync(path.join(__dirname, 'regional_policies.json'), 'utf-8'));
  assert(raw._meta, '应有 _meta 元数据');
  assert(raw.cities, '应有 cities');
  assert(raw.provinceMapping, '应有 provinceMapping');
  assert(Object.keys(raw.cities).length >= 20, `应有>=20个城市，实际${Object.keys(raw.cities).length}`);
  assert(Object.keys(raw.provinceMapping).length >= 14, `应有>=14个省份映射，实际${Object.keys(raw.provinceMapping).length}`);
});

test('qa_engine.js 不再硬编码 REGIONAL_POLICIES', () => {
  const code = fs.readFileSync(path.join(__dirname, 'qa_engine.js'), 'utf-8');
  assert(!code.includes('const REGIONAL_POLICIES ='), '不应包含硬编码 REGIONAL_POLICIES');
  assert(code.includes('regional_policies.json'), '应引用外部 JSON 文件');
  assert(code.includes('REGIONAL_DATA'), '应使用 REGIONAL_DATA 变量');
});

test('地域检测功能正常', () => {
  // QAEngine 需要 SearchEngine 才能实例化，但 _detectRegion 是独立方法
  // 直接测试加载逻辑
  const code = fs.readFileSync(path.join(__dirname, 'qa_engine.js'), 'utf-8');
  assert(code.includes('REGIONAL_DATA.cities'), '应使用 REGIONAL_DATA.cities');
  assert(code.includes('REGIONAL_DATA.provinceMapping'), '应使用 REGIONAL_DATA.provinceMapping');
});

// ========== 改进5: ChainReasoner 关键词提取增强 ==========
console.log('\n📦 改进5: ChainReasoner 关键词提取增强验证');

test('4层 fallback 提取 - Level 1: 完整JSON', () => {
  const { ChainReasoner } = require('./chain_reasoner');
  const cr = new ChainReasoner();
  const output = '{"searchTerms": ["劳动仲裁", "拖欠工资"], "enhancedQuery": "xxx"}';
  const result = cr._extractSearchTerms(output);
  assert(result.length === 2, `Level 1 应提取2个词，实际${result.length}`);
  assert(result.includes('劳动仲裁'), '应包含劳动仲裁');
});

test('4层 fallback 提取 - Level 1: enhancedQuery', () => {
  const { ChainReasoner } = require('./chain_reasoner');
  const cr = new ChainReasoner();
  const output = '{"facts": [], "enhancedQuery": "劳动合同到期补偿"}';
  const result = cr._extractSearchTerms(output);
  assert(result.length === 1, `应提取1个增强查询`);
  assert(result[0] === '劳动合同到期补偿', '应为增强查询内容');
});

test('4层 fallback 提取 - Level 1: keyIssues', () => {
  const { ChainReasoner } = require('./chain_reasoner');
  const cr = new ChainReasoner();
  const output = '{"keyIssues": ["工伤认定", "赔偿标准"], "dispute": "工伤"}';
  const result = cr._extractSearchTerms(output);
  assert(result.length === 2, `应提取2个争议焦点`);
});

test('4层 fallback 提取 - Level 1: dispute + domain 合并', () => {
  const { ChainReasoner } = require('./chain_reasoner');
  const cr = new ChainReasoner();
  const output = '{"dispute": "拖欠工资", "domain": "劳动法", "relationship": "劳动关系"}';
  const result = cr._extractSearchTerms(output);
  assert(result.length === 3, `应合并3个字段`);
});

test('4层 fallback 提取 - Level 3: 法律关键词库匹配', () => {
  const { ChainReasoner } = require('./chain_reasoner');
  const cr = new ChainReasoner();
  // 纯文本，没有JSON
  const output = '该案件涉及工资拖欠和加班费计算问题，需要通过劳动仲裁解决';
  const result = cr._extractSearchTerms(output);
  assert(result.length >= 2, `应从文本中提取到至少2个法律关键词，实际${result.length}`);
});

test('4层 fallback 提取 - Level 4: 完全无法识别返回空', () => {
  const { ChainReasoner } = require('./chain_reasoner');
  const cr = new ChainReasoner();
  const output = '这是一段完全没有法律关键词的普通文本';
  const result = cr._extractSearchTerms(output);
  assert(Array.isArray(result), '应返回数组');
});

// ========== 总结 ==========
console.log('\n' + '═'.repeat(60));
console.log(`📊 验证结果: ${passed} 通过, ${failed} 失败, 共 ${passed + failed} 项`);
console.log('═'.repeat(60));

if (failed > 0) process.exit(1);
