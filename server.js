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

// Static serving of frontend (optional). Place tigeriy.html in same directory.
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(__dirname, { setHeaders: (res) => { res.set('Content-Type', 'text/html; charset=utf-8'); } }));
app.get('/', (req,res)=> { res.set('Content-Type', 'text/html; charset=utf-8'); res.sendFile(path.join(__dirname, 'tigeriy.html')); });

const PORT = process.env.PORT || 3000;
app.listen(PORT, ()=> console.log('LLM proxy + static frontend listening on', PORT));

