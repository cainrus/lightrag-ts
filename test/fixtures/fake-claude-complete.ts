#!/usr/bin/env node
// Stand-in for the claude-complete bridge in tests: never calls a model.
// Appends {argv, stdin} to $FAKE_CLAUDE_LOG (one JSON line per call) and
// answers by $FAKE_CLAUDE_MODE:
//   reply (default)  stdout: $FAKE_CLAUDE_REPLY, else "  reply:<md5 of stdin>\n"
//   fail             stderr: "boom", exit 3
//   limit            stderr: the CLI's session-limit text, exit 1
//   hang             never answers
//   describe         --describe-model prints {"provider","model"}
// A file named by $FAKE_CLAUDE_REPLIES (JSON: md5 of stdin -> reply) wins over the default reply.
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, readFileSync } from 'node:fs';

const argv = process.argv.slice(2);
const i = argv.indexOf('--describe-model');
if (i >= 0) {
  process.stdout.write(JSON.stringify({ provider: 'anthropic', model: argv[i + 1] }) + '\n');
  process.exit(0);
}
const stdin = readFileSync(0, 'utf8');
const log = process.env.FAKE_CLAUDE_LOG;
if (log) appendFileSync(log, JSON.stringify({ argv, stdin }) + '\n');
const md5 = createHash('md5').update(stdin).digest('hex');
const mode = process.env.FAKE_CLAUDE_MODE ?? 'reply';
if (mode === 'fail') {
  process.stderr.write('  boom\n');
  process.exit(3);
}
if (mode === 'limit') {
  process.stderr.write("claude -p failed (rc=1): You've hit your session limit · resets 11pm\n");
  process.exit(1);
}
if (mode === 'hang') setInterval(() => {}, 1000);
else {
  const table = process.env.FAKE_CLAUDE_REPLIES;
  const replies = table && existsSync(table) ? (JSON.parse(readFileSync(table, 'utf8')) as Record<string, string>) : {};
  process.stdout.write(replies[md5] ?? process.env.FAKE_CLAUDE_REPLY ?? `  reply:${md5}\n`);
}
