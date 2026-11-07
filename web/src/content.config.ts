import { defineCollection } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
  // Empty: the site is English only. Declared so Starlight doesn't warn that it's missing.
  i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() }),
};
