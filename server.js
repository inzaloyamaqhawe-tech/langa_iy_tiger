// Unified LLM backend proxy
// Supports provider: gemini | openai | github 
// IMPORTANT: GitHub Copilot models are NOT publicly exposed via a simple REST API.
// The 'github' provider here is a placeholder returning 501 until you wire an actual endpoint.
// Keep this server private; never expose raw API keys client-side.

import express from 'express';
import fetch from 'node-fetch';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { fileURLToPath } from 'url';
import { Pool } from 'pg';

const app = express();
app.use(express.json({ limit: '20kb' }));

// Basic rate limiting (tune for production)
const limiter = rateLimit({ windowMs: 60 * 1000, max: 60 });
app.use('/api/llm', limiter);

// Utility: safe JSON extraction
function safeJsonParse(str){ try{ return JSON.parse(str); }catch{return null;} }

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

// If DATABASE_URL is set, enable setups API backed by Postgres
let pool = null;

(async function initDb(){
  if(!process.env.DATABASE_URL) return;
  try{
    const pg = await import('pg');
    const Pool = pg.Pool;
    pool = new Pool({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
    // simple test connect
    await pool.connect().then(c=>{ c.release(); console.log('Postgres pool connected'); }).catch(e=>{ console.warn('Postgres connect failed', e.message); pool = null; });
  }catch(e){ console.warn('Postgres init failed', e && e.message ? e.message : e); pool = null; return; }

  // Register setups API routes now that pool is available
  app.get('/api/setups', async (req,res)=>{
    try{
      const q = `SELECT canonical_id, symbol, setup, status, entry_ts, hit_history, created_at FROM setups WHERE upper(symbol) = 'PAXGUSDT' ORDER BY created_at DESC LIMIT 500`;
      const r = await pool.query(q);
      return res.json({ setups: r.rows });
    }catch(e){ console.error('/api/setups error', e); return res.status(500).json({ error: 'db error' }); }
  });

  app.post('/api/setups', async (req,res)=>{
    try{
      const { canonical_id, symbol, setup, status, origin, entry_ts } = req.body || {};
      if(!canonical_id) return res.status(400).json({ error:'canonical_id required' });
      const q = `INSERT INTO setups (canonical_id, symbol, setup, status, origin, entry_ts, created_at, updated_at)
                 VALUES ($1,$2,$3::jsonb,$4,$5,$6, now(), now())
                 ON CONFLICT (canonical_id) DO UPDATE SET setup = EXCLUDED.setup, status = EXCLUDED.status, origin = EXCLUDED.origin, entry_ts = EXCLUDED.entry_ts, updated_at = now()
                 RETURNING canonical_id, symbol, setup, status, entry_ts, hit_history, created_at`;
      const vals = [canonical_id, symbol || null, setup || {}, status || 'PENDING', origin || null, entry_ts || null];
      const r = await pool.query(q, vals);
      return res.json({ ok:true, row: r.rows[0] });
    }catch(e){ console.error('POST /api/setups error', e); return res.status(500).json({ error:'db error' }); }
  });

  app.patch('/api/setups/:canonical', async (req,res)=>{
    try{
      const canonical = req.params.canonical;
      const body = req.body || {};
      const parts = [];
      const vals = [];
      let idx = 1;
      if(typeof body.status !== 'undefined'){ parts.push(`status = $${idx++}`); vals.push(body.status); }
      if(typeof body.hitHistory !== 'undefined'){ parts.push(`hit_history = $${idx++}`); vals.push(body.hitHistory); }
      if(typeof body.setup !== 'undefined'){ parts.push(`setup = $${idx++}::jsonb`); vals.push(body.setup); }
      if(typeof body.entry_ts !== 'undefined'){ parts.push(`entry_ts = $${idx++}`); vals.push(body.entry_ts); }
      if(parts.length === 0) return res.status(400).json({ error:'no updatable fields' });
      const q = `UPDATE setups SET ${parts.join(', ')}, updated_at = now() WHERE canonical_id = $${idx} RETURNING canonical_id, symbol, setup, status, entry_ts, hit_history, created_at`;
      vals.push(canonical);
      const r = await pool.query(q, vals);
      if(!r.rows.length) return res.status(404).json({ error:'not found' });
      return res.json({ ok:true, row: r.rows[0] });
    }catch(e){ console.error('PATCH /api/setups error', e); return res.status(500).json({ error:'db error' }); }
  });

  app.delete('/api/admin/cleanup', async (req,res)=>{
    try{
      const pattern = (req.query.pattern||'').toString().toUpperCase();
      if(!pattern) return res.status(400).json({ error:'pattern required' });
      const like = `%${pattern}%`;
      await pool.query('BEGIN');
      const r = await pool.query(`DELETE FROM setups WHERE upper(symbol) LIKE $1 RETURNING canonical_id`, [like]);
      await pool.query('COMMIT');
      return res.json({ ok:true, deleted: r.rows.length });
    }catch(e){ try{ await pool.query('ROLLBACK'); }catch(_){} console.error('DELETE /api/admin/cleanup error', e); return res.status(500).json({ error:'db error' }); }
  });

})();

// Static serving of frontend (optional). Place crttrade.html in same directory.
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(__dirname, { setHeaders: (res) => { res.set('Content-Type', 'text/html; charset=utf-8'); } }));
app.get('/', (req,res)=> { res.set('Content-Type', 'text/html; charset=utf-8'); res.sendFile(path.join(__dirname, 'crttrade.html')); });

const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=> console.log('LLM proxy + static frontend listening on', PORT));

