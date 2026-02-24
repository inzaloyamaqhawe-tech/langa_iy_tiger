#!/usr/bin/env node
import fs from 'fs';

const DATABASE_URL = process.env.DATABASE_URL || process.argv[2];
if(!DATABASE_URL){
  console.error('Usage: set DATABASE_URL env or provide as first arg');
  console.error('Example (Linux/mac): DATABASE_URL="postgres://..." node export_setups.mjs');
  console.error('Example (Windows PowerShell): $Env:DATABASE_URL = "postgres://..."; node export_setups.mjs');
  process.exit(2);
}

try{
  const pg = await import('pg');
  const Pool = pg.Pool;
  const pool = new Pool({ connectionString: DATABASE_URL, ssl: { rejectUnauthorized: false } });
  const q = `SELECT canonical_id, symbol, setup, status, entry_ts, hit_history, created_at, updated_at FROM setups ORDER BY created_at DESC`;
  const r = await pool.query(q);
  const rows = r.rows || [];
  const outFile = `setups_export_${new Date().toISOString().replace(/[:.]/g,'-')}.json`;
  fs.writeFileSync(outFile, JSON.stringify(rows, null, 2), 'utf8');
  console.log('Exported', rows.length, 'rows to', outFile);
  await pool.end();
}catch(e){
  console.error('Export failed:', e && e.message ? e.message : e);
  process.exit(1);
}
