import { readFileSync, readdirSync, writeFileSync, mkdirSync } from 'fs';
import { join, relative } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const publicDocs = join(__dirname, '..', 'public', 'docs');
const outDir = join(__dirname, '..', 'src', 'generated');
const outFile = join(outDir, 'docsMap.json');

function collectMd(dir, base = '') {
  const entries = readdirSync(dir, { withFileTypes: true });
  const out = {};
  for (const e of entries) {
    const full = join(dir, e.name);
    const rel = base ? `${base}/${e.name}` : e.name;
    if (e.isDirectory()) {
      Object.assign(out, collectMd(full, rel));
    } else if (e.name.endsWith('.md')) {
      const key = '/docs/' + rel.replace(/\\/g, '/');
      out[key] = readFileSync(full, 'utf-8');
    }
  }
  return out;
}

mkdirSync(outDir, { recursive: true });
const map = collectMd(publicDocs);
writeFileSync(outFile, JSON.stringify(map), 'utf-8');
console.log('Generated docsMap.json with', Object.keys(map).length, 'docs');
