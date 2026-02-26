// Unified LLM backend proxy
// Supports provider: gemini | openai | github (stub)
// IMPORTANT: GitHub Copilot models are NOT publicly exposed via a simple REST API.
// The 'github' provider here is a placeholder returning 501 until you wire an actual endpoint.
// Keep this server private; never expose raw API keys client-side.

import express from 'express';
import fetch from 'node-fetch';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { fileURLToPath } from 'url';
import cors from 'cors';
import { Pool } from 'pg';

const app = express();
app.use(express.json({ limit: '20kb' }));
app.use(cors());

// Basic rate limiting (tune for production)
const limiter = rateLimit({ windowMs: 60 * 1000, max: 60 });
app.use('/api/llm', limiter);

// Utility: safe JSON extraction
function safeJsonParse(str){ try{ return JSON.parse(str); }catch{return null;} }
function normalizeJsonField(value, fallback){
  if(value === undefined) return undefined;
  if(value === null) return null;
  if(typeof value === 'string'){
    const parsed = safeJsonParse(value);
    if(parsed !== null) return parsed;
    return fallback;
  }
  return value;
}

function roundSetupPriceFields(setup){
  if(!setup || typeof setup !== 'object') return setup;
  const out = { ...setup };
  const keys = ['entry','sl','tp1','tp2','tp3','tp4','tp5','tp6'];
  for(const key of keys){
    const n = Number(out[key]);
    if(Number.isFinite(n)) out[key] = Number(n.toFixed(2));
  }
  return out;
}

function applySymbolTpProfile(sym, setup, cid){
  if(!setup || !sym) return setup;
  const out = { ...setup };
  const symbol = String(sym).toUpperCase();
  let entry = Number(out.entry ?? out.entryPrice ?? out.entry_ts ?? NaN);
  if(!Number.isFinite(entry) && cid && typeof cid === 'string'){
    const parts = cid.split('|');
    const maybe = Number(parts[parts.length - 1]);
    if(Number.isFinite(maybe)) entry = maybe;
  }
  if(!Number.isFinite(entry)) return out;

  const side = String(out.side || 'BUY').toUpperCase();
  const round2 = (value) => Math.round(value * 100) / 100;

  if(symbol === 'PAXGUSDT' || symbol === 'PAXGUSD'){
    const offsets = side === 'BUY' ? [5, 10, 15, 20, 30] : [-5, -10, -15, -20, -30];
    out.tp1 = round2(entry + offsets[0]);
    out.tp2 = round2(entry + offsets[1]);
    out.tp3 = round2(entry + offsets[2]);
    out.tp4 = round2(entry + offsets[3]);
    out.tp5 = round2(entry + offsets[4]);
    out.entry = round2(entry);
    return out;
  }

  if(symbol.startsWith('ETH')){
    const offsets = side === 'BUY' ? [15,30,45,60,75] : [-15,-30,-45,-60,-75];
    for(let i=0;i<5;i++) out['tp'+(i+1)] = round2(entry + offsets[i]);
    if(!out.entry) out.entry = round2(entry);
  }

  return out;
}

// Build prompt from task + snapshots
function buildPrompt(task, snapshot, fundamentals){
  return [
    'You are an institutional-grade market structure & forward catalyst analyst.',
    `Task: ${task || 'trade-analysis'}`,
    'Return ONLY valid JSON. Keys: recommendation (BUY|STRONG BUY|SELL|STRONG SELL|HOLD), probability (0-100), reasons (string[] up to 8), analysis_summary (<=280 chars).',
    'Calibrate probability: actionable (BUY/SELL variants) 70-95; HOLD 45-60.',
    'Focus forward: catalysts in next horizon; minimize recap of past.',
    'Snapshot JSON:', JSON.stringify(snapshot||{}),
    'Fundamentals JSON:', JSON.stringify(fundamentals||{}),
    'Output strictly one JSON object.'
  ].join('\n');
}

// Normalize upstream model response to app format
function normalizeModelJson(j){
  if(!j) return null;
  const recRaw = String(j.recommendation||'HOLD').toUpperCase();
  let rec = ['BUY','STRONG BUY','SELL','STRONG SELL','HOLD'].includes(recRaw) ? recRaw : 'HOLD';
  let prob = Number(j.probability||50);
  if(['BUY','STRONG BUY','SELL','STRONG SELL'].includes(rec)) prob = Math.min(95, Math.max(70, prob));
  else prob = Math.min(60, Math.max(45, prob));
  const reasons = Array.isArray(j.reasons) ? j.reasons.slice(0,8) : [];
  return { recommendation: rec, probability: Math.round(prob), reasons, source: 'LLM' };
}

async function callGemini(model, prompt){
  const endpoint = `https://generativelanguage.googleapis.com/v1/models/${encodeURIComponent(model)}:generateContent?key=${process.env.GEMINI_KEY}`;
  const body = { contents: [{ parts: [{ text: prompt }]}], generationConfig:{ temperature:0.25, maxOutputTokens:700, responseMimeType:'application/json' } };
  const r = await fetch(endpoint,{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  if(!r.ok) throw new Error('Gemini error '+r.status);
  const j = await r.json();
  const txt = j.candidates?.[0]?.content?.parts?.map(p=>p.text).join(' ') || '';
  return txt;
}

async function callOpenAI(model, prompt){
  const r = await fetch('https://api.openai.com/v1/chat/completions',{ method:'POST', headers:{ 'Authorization':`Bearer ${process.env.OPENAI_KEY}`, 'Content-Type':'application/json' }, body: JSON.stringify({ model, messages:[{role:'system', content:'Return ONLY JSON as specified.'},{role:'user', content:prompt}], temperature:0.2, max_tokens:700 }) });
  if(!r.ok) throw new Error('OpenAI error '+r.status);
  const j = await r.json();
  return j.choices?.[0]?.message?.content || '';
}

async function callOllama(model, prompt){
  const base = process.env.OLLAMA_URL || 'http://localhost:11434';
  const r = await fetch(`${base}/api/generate`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ model, prompt, stream:false }) });
  if(!r.ok) throw new Error('Ollama error '+r.status);
  const j = await r.json();
  return j.response || ''; // Ollama returns {response: '...'}
}

app.post('/api/llm', async (req,res) => {
  const { provider, model, task, snapshot, fundamentals } = req.body || {};
  // Optional bearer auth
  const authToken = process.env.LLM_AUTH_TOKEN;
  if(authToken){
    const header = req.get('Authorization') || '';
    if(header !== `Bearer ${authToken}`) return res.status(401).json({ error:'unauthorized' });
  }
  if(!provider || !model) return res.status(400).json({ error:'provider and model required'});
  const prompt = buildPrompt(task, snapshot, fundamentals);
  try{
    let raw='';
  if(provider === 'gemini') raw = await callGemini(model, prompt);
  else if(provider === 'openai') raw = await callOpenAI(model, prompt);
  else if(provider === 'ollama') raw = await callOllama(model, prompt);
  else if(provider === 'github') return res.status(501).json({ error:'GitHub provider stub: no public Copilot inference API. Implement when available.'});
    else return res.status(400).json({ error:'Unknown provider'});

    const cleaned = raw.replace(/```json|```/gi,'').trim();
    let parsed = safeJsonParse(cleaned);
    if(!parsed){
      const match = cleaned.match(/\{[\s\S]*\}/);
      if(match) parsed = safeJsonParse(match[0]);
    }
    if(!parsed) return res.status(422).json({ error:'Model output not JSON', raw: cleaned.slice(0,500) });
    const norm = normalizeModelJson(parsed);
    if(!norm) return res.status(422).json({ error:'Normalization failed', parsed });
    return res.json({ ...norm, raw: parsed });
  }catch(e){
    console.error('[LLM ERROR]', e);
    return res.status(500).json({ error:'LLM failure', detail:e.message });
  }
});

app.get('/health', (req,res)=> res.json({ ok:true, ts:Date.now() }));

// Postgres pool (enabled when DATABASE_URL is set)
let db = null;
if(process.env.DATABASE_URL){
  db = new Pool({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized:false } });
  db.connect().then(c=>{ c.release(); console.log('Postgres pool connected'); }).catch(e=>{ console.error('Postgres connection failed', e.message); db=null; });
}

function setupsAuth(req){
  const token = process.env.SETUPS_API_TOKEN;
  if(!token) return true;
  const header = (req.get('Authorization')||'').replace(/^Bearer\s*/i,'');
  return header === token;
}

// Always register setups API routes. If DB isn't configured the routes will return 503
app.get('/api/setups', async (req,res)=>{
  if(!setupsAuth(req)) return res.status(401).json({ error:'unauthorized' });
  if(!db) return res.status(503).json({ error:'database not configured', message:'Set DATABASE_URL env var on the server and redeploy.' });
  try{
    // Only return PAXGUSDT setups (system now restricted to PAXGUSD)
    const q = await db.query("SELECT canonical_id, symbol, setup, status, origin, entry_ts, hit_history, created_at, updated_at FROM setups WHERE upper(symbol) = 'PAXGUSDT' ORDER BY created_at DESC LIMIT 1000");
    return res.json({ setups: q.rows });
  }catch(e){ console.error(e); return res.status(500).json({ error:'db error' }); }
});

app.post('/api/setups', async (req,res)=>{
  if(!setupsAuth(req)) return res.status(401).json({ error:'unauthorized' });
  if(!db) return res.status(503).json({ error:'database not configured', message:'Set DATABASE_URL env var on the server and redeploy.' });
  const payload = req.body || {};
  const canonical_id = payload.canonical_id || payload.setup?.canonical_id;
  const symbol = payload.symbol || payload.setup?.symbol;
  const setup = payload.setup || payload;
  const status = payload.status || setup.status || 'PENDING';
  const origin = payload.origin || setup.origin || null;
  const entry_ts = payload.entry_ts || setup.entryTs || null;
  if(!canonical_id || !symbol) return res.status(400).json({ error:'canonical_id and symbol required' });
  // Enforce whitelist: only allow PAXGUSDT setups going forward
  if(String(symbol).toUpperCase() !== 'PAXGUSDT'){
    return res.status(403).json({ error:'Only PAXGUSDT setups are accepted by this service' });
  }
  try{
    const normalized = applySymbolTpProfile(symbol, setup, canonical_id);
    if(normalized) Object.assign(setup, normalized);
    Object.assign(setup, roundSetupPriceFields(setup));
  }catch(e){ console.error('normalize error', e); }
  try{
    await db.query(
      `INSERT INTO setups (canonical_id, symbol, setup, status, origin, entry_ts, hit_history) VALUES ($1,$2,$3,$4,$5,$6,$7)
       ON CONFLICT (canonical_id) DO UPDATE SET
         setup = EXCLUDED.setup,
         status = EXCLUDED.status,
         origin = EXCLUDED.origin,
         entry_ts = EXCLUDED.entry_ts,
         hit_history = EXCLUDED.hit_history,
         updated_at = now()`
      ,[canonical_id, symbol, setup, status, origin, entry_ts, setup.hitHistory || []]
    );
    return res.json({ ok:true });
  }catch(e){ console.error('upsert failed', e); return res.status(500).json({ error:'db error', detail: e.message, code: e.code }); }
});

app.patch('/api/setups/:canonical', async (req,res)=>{
  if(!setupsAuth(req)) return res.status(401).json({ error:'unauthorized' });
  if(!db) return res.status(503).json({ error:'database not configured', message:'Set DATABASE_URL env var on the server and redeploy.' });
  const canonical = req.params.canonical;
  const body = req.body || {};
  const fields = [];
  const values = [];
  let idx = 1;
  if(body.status){ fields.push(`status = $${idx++}`); values.push(body.status); }
  const hitHistory = normalizeJsonField(body.hitHistory, []);
  if(hitHistory !== undefined){ fields.push(`hit_history = $${idx++}`); values.push(hitHistory); }
  const setupRaw = normalizeJsonField(body.setup, {});
  const setupObj = roundSetupPriceFields(applySymbolTpProfile(body.symbol || 'PAXGUSDT', setupRaw, canonical));
  if(setupObj !== undefined){ fields.push(`setup = $${idx++}`); values.push(setupObj); }
  if(body.entry_ts){ fields.push(`entry_ts = $${idx++}`); values.push(body.entry_ts); }
  if(fields.length===0) return res.status(400).json({ error:'nothing to update' });
  values.push(canonical);
  const sql = `UPDATE setups SET ${fields.join(',')}, updated_at = now() WHERE canonical_id = $${idx}`;
  try{ await db.query(sql, values); return res.json({ ok:true }); }catch(e){ console.error('patch failed', e); return res.status(500).json({ error:'db error', detail: e.message, code: e.code }); }
});

// Admin cleanup: delete setups for symbol patterns (protected by setupsAuth)
app.delete('/api/admin/cleanup', async (req,res)=>{
  if(!setupsAuth(req)) return res.status(401).json({ error:'unauthorized' });
  if(!db) return res.status(503).json({ error:'database not configured' });
  const q = req.query || {};
  const pattern = (q.pattern || '').toUpperCase();
  if(!pattern) return res.status(400).json({ error:'pattern query required, e.g. ?pattern=ETH' });
  try{
    const sql = `DELETE FROM setups WHERE upper(symbol) LIKE $1`;
    const r = await db.query(sql, [pattern + '%']);
    return res.json({ ok:true, deleted: r.rowCount });
  }catch(e){ console.error('cleanup failed', e); return res.status(500).json({ error:'db error' }); }
});

// Admin cleanup: delete known legacy SELL entries by exact entry text (protected by setupsAuth)
app.delete('/api/admin/delete-legacy-sells', async (req,res)=>{
  if(!setupsAuth(req)) return res.status(401).json({ error:'unauthorized' });
  if(!db) return res.status(503).json({ error:'database not configured' });
  const entries = Array.isArray(req.body?.entries) && req.body.entries.length
    ? req.body.entries.map(String)
    : ['5133.33','5124.4349999999995','5112.945'];
  try{
    const sql = `
      DELETE FROM setups
      WHERE upper(symbol) = 'PAXGUSDT'
        AND upper(coalesce(setup->>'side','')) = 'SELL'
        AND coalesce(setup->>'entry','') = ANY($1::text[])
    `;
    const r = await db.query(sql, [entries]);
    return res.json({ ok:true, deleted: r.rowCount, entries });
  }catch(e){
    console.error('delete-legacy-sells failed', e);
    return res.status(500).json({ error:'db error' });
  }
});

// Static serving of frontend (optional). Place crttrade.html in same directory.
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(__dirname, {
  setHeaders: (res, filePath) => {
    if(String(filePath).toLowerCase().endsWith('.html')){
      res.set('Content-Type', 'text/html; charset=utf-8');
    }
  }
}));
app.get('/', (req,res)=> { res.set('Content-Type', 'text/html; charset=utf-8'); res.sendFile(path.join(__dirname, 'crttrade.html')); });

const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=> console.log('LLM proxy + static frontend listening on', PORT));

