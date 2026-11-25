// The dMLX website: a custom homepage (src/pages/index.astro) and Starlight docs under /docs/.
// Screenshots and logo files are read from the repo's docs/ folder, so they have one source.
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://www.dmlx.app',
  vite: { server: { fs: { allow: ['..'] } } },
  redirects: { '/docs': '/docs/getting-started/' },
  integrations: [
    starlight({
      title: 'dMLX',
      disable404Route: true,
      description: 'Make and edit images with chat, fully on your Mac.',
      logo: { light: '../docs/brand/dmlx-mark.svg', dark: '../docs/brand/dmlx-mark-dark.svg' },
      favicon: '/favicon.svg',
      components: {
        ThemeSelect: './src/components/ThemeSelect.astro',
        SocialIcons: './src/components/SocialIcons.astro',
        SkipLink: './src/components/SkipLink.astro',
      },
      head: [
        { tag: 'link', attrs: { rel: 'icon', href: '/favicon.ico', sizes: '48x48' } },
        { tag: 'link', attrs: { rel: 'apple-touch-icon', href: '/apple-touch-icon.png' } },
        { tag: 'meta', attrs: { property: 'og:site_name', content: 'dMLX' } },
        { tag: 'meta', attrs: { property: 'og:image', content: 'https://www.dmlx.app/og.jpg' } },
        { tag: 'meta', attrs: { property: 'og:image:width', content: '1200' } },
        { tag: 'meta', attrs: { property: 'og:image:height', content: '630' } },
        { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } },
      ],
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/rootonchair/dmlx' }],
      customCss: [
        '@fontsource/ibm-plex-sans/400.css',
        '@fontsource/ibm-plex-sans/600.css',
        '@fontsource/ibm-plex-mono/400.css',
        '@fontsource/figtree/600.css',
        './src/styles/docs.css',
      ],
      sidebar: [
        { label: 'Start here', items: ['docs/getting-started', 'docs/using-dmlx', 'docs/crop-and-draw'] },
        { label: 'How it works', items: ['docs/agent', 'docs/models', 'docs/memory'] },
        { label: 'Reference', items: ['docs/api', 'docs/build-from-source', 'docs/known-issues', 'docs/release-notes'] },
      ],
    }),
  ],
});
