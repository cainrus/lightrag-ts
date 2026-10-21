// Count nodes / edges / docs / vectors of a LightRAG working dir, read-only.
//   node scripts/store-stats.ts <working_dir> [--workspace W] [--json out.json]
import { writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { storeStats } from '../src/store.ts';

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: { workspace: { type: 'string', default: '' }, json: { type: 'string' } },
});
const dir = positionals[0];
if (!dir) {
  console.error('usage: store-stats.ts <working_dir> [--workspace W] [--json out.json]');
  process.exit(2);
}
const t0 = performance.now();
const stats = storeStats(dir, values.workspace, (s) => console.log(`  ${s}  (${((performance.now() - t0) / 1000).toFixed(1)}s)`));
const result = {
  dir,
  seconds: Number(((performance.now() - t0) / 1000).toFixed(1)),
  peakRssMb: Math.round(process.resourceUsage().maxRSS / 1024),
  ...stats,
};
console.log(JSON.stringify(result, null, 2));
if (values.json) writeFileSync(values.json, JSON.stringify(result, null, 2) + '\n');
