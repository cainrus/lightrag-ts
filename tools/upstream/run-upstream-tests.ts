// Run upstream LightRAG's own pytest files against lightrag-ts (d11).
//
//   node tools/upstream/run-upstream-tests.ts [pytest args...]
//
// The test files come from an upstream git checkout at the manifest's tag
// (LIGHTRAG_UPSTREAM, default ~/vendors/lightrag), exported into a temp dir with
// upstream's pytest settings; tools/upstream/lightrag_ts_adapter.py swaps the
// ported functions for TS ones. The Python side needs lightrag-hku at that tag
// plus pytest and pytest-asyncio (LIGHTRAG_TEST_PY), e.g.
//   uv venv --python 3.12 ~/.cache/lightrag-ts/upstream-venv
//   uv pip install --python ~/.cache/lightrag-ts/upstream-venv/bin/python \
//     lightrag-hku==1.4.16 'pytest>=8.4.2' 'pytest-asyncio>=1.2.0'
import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '../..');
const MANIFEST = join(ROOT, 'tools/upstream/upstream-tests.json');
const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8')) as { upstream: string; files: Record<string, unknown> };
const UPSTREAM = process.env.LIGHTRAG_UPSTREAM ?? join(homedir(), 'vendors/lightrag');
const PY = process.env.LIGHTRAG_TEST_PY ?? join(homedir(), '.cache/lightrag-ts/upstream-venv/bin/python');

if (!existsSync(PY)) {
  console.error(`no python at ${PY}; create the venv (see the header of this file) or set LIGHTRAG_TEST_PY`);
  process.exit(2);
}
const git = (...args: string[]) => execFileSync('git', ['-C', UPSTREAM, ...args], { encoding: 'utf8', maxBuffer: 1 << 26 });
const version = execFileSync(PY, ['-I', '-c', 'import lightrag; print(lightrag.__version__)'], { encoding: 'utf8' }).trim();
if (`v${version}` !== manifest.upstream) {
  console.error(`python has lightrag ${version}, the manifest pins ${manifest.upstream}`);
  process.exit(2);
}

const dir = mkdtempSync(join(tmpdir(), 'lightrag-upstream-tests-'));
for (const f of ['tests/__init__.py', 'tests/conftest.py', ...Object.keys(manifest.files)]) {
  const out = join(dir, f);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, git('show', `${manifest.upstream}:${f}`));
}
// upstream's [tool.pytest.ini_options]
const pyproject = git('show', `${manifest.upstream}:pyproject.toml`);
const ini = /\[tool\.pytest\.ini_options\]\n([\s\S]*?)(?:\n\[|$)/.exec(pyproject)?.[1] ?? '';
writeFileSync(join(dir, 'pytest.ini'), `[pytest]\n${ini.replace(/^(\w+)\s*=\s*\[(.*)\]$/gm, (_m, k: string, v: string) => `${k} = ${v.replace(/[",]/g, ' ').trim()}`).replace(/"/g, '')}\n`);

const files = Object.keys(manifest.files).map((f) => join(dir, f));
const r = spawnSync(PY, ['-m', 'pytest', '-p', 'lightrag_ts_adapter', '-rs', ...process.argv.slice(2), ...files], {
  cwd: dir,
  stdio: 'inherit',
  env: {
    ...process.env,
    PYTHONPATH: join(ROOT, 'tools/upstream'),
    LIGHTRAG_TS_ROOT: ROOT,
    LIGHTRAG_TS_MANIFEST: MANIFEST,
    LIGHTRAG_TS_NODE: process.execPath,
  },
});
process.exit(r.status ?? 1);
