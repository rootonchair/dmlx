// Builds the docs pages (src/content/docs/docs/*.md) from the repo's README.md and docs/*.md.
// The README stays the one source; these generated pages are not committed. Runs before `dev` and `build`.
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const root = join(web, '..');
const out = join(web, 'src/content/docs/docs');
const repo = 'https://github.com/rootonchair/dmlx/blob/main/';
// From src/content/docs/docs/ back to the repo root.
const up = '../../../../../';

/** Splits a Markdown file into its "## " sections, keyed by heading. */
function sections(md) {
  const map = new Map();
  let name = '_intro';
  for (const line of md.split('\n')) {
    const m = line.match(/^## (.+)$/);
    if (m) { name = m[1].trim(); map.set(name, []); continue; }
    if (!map.has(name)) map.set(name, []);
    map.get(name).push(line);
  }
  return new Map([...map].map(([k, v]) => [k, v.join('\n').trim()]));
}

/** Makes README Markdown work as a docs page: images, links, heading levels. */
function adapt(md, { from = '' } = {}) {
  return md
    // <p align="center"><img …><br><sub>caption</sub></p> and plain <img …> become Markdown images.
    .replace(/<p align="center">\s*([\s\S]*?)<\/p>/g, '$1')
    .replace(/<br>\s*<sub>([^<]*)<\/sub>/g, '\n\n*$1*')
    .replace(/<img\s+src="([^"]+)"[^>]*?alt="([^"]*)"[^>]*>/g, (_, src, alt) => `![${alt.replace(/[[\]]/g, '')}](${up}${from}${src})`)
    .replace(/<br>/g, '\n\n')
    // Repo-relative links go to GitHub; links to other docs pages stay on the site.
    .replace(/\]\((?!https?:|#|\.\.\/\.\.\/)([^)]+)\)/g, (_, path) => {
      const p = (from + path).replace(/^\.\//, '');
      if (p === 'docs/RELEASE_NOTES.md') return '](/docs/release-notes/)';
      if (p === 'docs/KNOWN_ISSUES.md') return '](/docs/known-issues/)';
      return `](${repo}${p})`;
    })
    // Each page has its own title, so "###" sections become "##".
    .replace(/^### /gm, '## ');
}

function page(slug, title, description, body) {
  writeFileSync(join(out, `${slug}.md`), `---\ntitle: ${JSON.stringify(title)}\ndescription: ${JSON.stringify(description)}\n---\n\n${body.trim()}\n`);
}

const readme = sections(readFileSync(join(root, 'README.md'), 'utf8'));
const get = (name) => {
  if (!readme.has(name)) throw new Error(`README.md has no "## ${name}" section`);
  return adapt(readme.get(name));
};
const status = (readme.get('_intro').match(/> \*\*Status:[\s\S]*?(?=\n\n|\n<p|$)/) || [''])[0];

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

page('getting-started', 'Getting started', 'Requirements and how to install dMLX.',
  `${adapt(status)}\n\n## Requirements\n\n${get('Requirements')}\n\n## Install\n\n${get('Install')}`);
page('using-dmlx', 'Using dMLX', 'Make and edit images with chat, with examples and a tour of the app.',
  `${get('How to use dMLX')}\n\n## What you can make\n\n${get('What you can make')}\n\n## Tour of the app\n\n${get('Tour of the app')}`);
page('crop-and-draw', 'Crop / Draw', 'Show the part of an image to change, then say what to do.', get('Edit with Crop / Draw'));
page('agent', 'The agent', 'How the agent on your Mac plans each message.', get('The agent'));
page('models', 'Models', 'The image models dMLX runs, and how to add more.', get('Image models'));
page('memory', 'How dMLX controls memory', 'How the engine keeps your Mac usable while it works.', get('How dMLX controls memory'));
page('api', 'Engine API', 'The local, OpenAI-compatible images API.', get('Engine API'));
page('build-from-source', 'Build from source', 'Build and test the engine and the app.',
  `${get('Build from source')}\n\n## Project layout\n\n${get('Project layout')}\n\n## License\n\n${get('License')}`);

for (const [file, slug, title, description] of [
  ['KNOWN_ISSUES.md', 'known-issues', 'Known issues', 'Problems we know about in this release.'],
  ['RELEASE_NOTES.md', 'release-notes', 'Release notes', 'What changed in each release.'],
]) {
  const md = readFileSync(join(root, 'docs', file), 'utf8').replace(/^# .+\n/, '');
  page(slug, title, description, adapt(md, { from: 'docs/' }).replace(/\]\(https:\/\/github\.com\/rootonchair\/dmlx\/blob\/main\/docs\/\.\.\//g, `](${repo}`));
}
console.log(`sync-docs: wrote ${out}`);
