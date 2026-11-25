// The download the website links to: the newest published release's .dmg, read from GitHub at build time.
// If GitHub can't be reached, the link is built from the version in engine/pyproject.toml, in the same
// form as tools/bundle_app.sh names the file and the release tags it (0.1.0b2 -> v0.1.0-beta2).
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const repo = 'rootonchair/dmlx';
export const releasesPage = `https://github.com/${repo}/releases`;

export interface Download {
  url: string;
  tag: string;
  sizeMB?: number;
  prerelease: boolean;
}

function fromVersion(): Download {
  // Resolved from the working directory (web/), not import.meta.url: once built, this module runs from
  // web/dist/.prerender/chunks/, where a relative URL no longer reaches engine/.
  const toml = readFileSync(resolve(process.cwd(), '../engine/pyproject.toml'), 'utf8');
  const version = toml.match(/^version\s*=\s*"([^"]+)"/m)?.[1] ?? '';
  const m = version.match(/^(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?$/);
  const names: Record<string, string> = { a: 'alpha', b: 'beta', rc: 'rc' };
  const tag = m ? `v${m[1]}${m[2] ? `-${names[m[2]]}${m[3]}` : ''}` : `v${version}`;
  return {
    url: `${releasesPage}/download/${tag}/dMLX-${version}-macos-arm64.dmg`,
    tag,
    prerelease: Boolean(m?.[2]),
  };
}

async function fromGitHub(): Promise<Download | undefined> {
  const res = await fetch(`https://api.github.com/repos/${repo}/releases?per_page=10`, {
    headers: {
      Accept: 'application/vnd.github+json',
      // Unauthenticated calls from shared build machines (Vercel) often hit the rate limit and get a 403.
      ...(process.env.GITHUB_TOKEN ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` } : {}),
    },
    signal: AbortSignal.timeout(8000),
  });
  if (!res.ok) throw new Error(`GitHub API ${res.status}`);
  const releases = (await res.json()) as {
    tag_name: string; draft: boolean; prerelease: boolean;
    assets: { name: string; size: number; browser_download_url: string }[];
  }[];
  for (const r of releases) {
    const dmg = r.assets.find((a) => a.name.endsWith('-macos-arm64.dmg'));
    if (!r.draft && dmg) {
      return { url: dmg.browser_download_url, tag: r.tag_name, sizeMB: Math.round(dmg.size / 1e6), prerelease: r.prerelease };
    }
  }
  return undefined;
}

let cached: Promise<Download> | undefined;

export function latestDownload(): Promise<Download> {
  cached ??= fromGitHub()
    .then((d) => d ?? fromVersion())
    .catch((error) => {
      console.warn(`release: ${error}; linking to the version in engine/pyproject.toml`);
      return fromVersion();
    });
  return cached;
}
