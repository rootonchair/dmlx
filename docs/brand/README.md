# dMLX brand

The logo is **Still Water**: a half sun over its reflection in still water. The reflection breaks into
ripples. It stands for something quiet that happens on your Mac.

| File | Use |
|---|---|
| `dmlx-mark.svg` | The mark on light backgrounds |
| `dmlx-mark-dark.svg` | The mark on dark backgrounds |
| `dmlx-mark-black.svg`, `dmlx-mark-white.svg` | One color only |
| `dmlx-mark-small.svg` | 32 px and smaller (favicon, menu bar). It has two thicker lines |
| `dmlx-app-icon.svg` | The macOS app icon (superellipse tile on the 1024 grid) |

The app icon (`app/Resources/AppIcon.icns`) and `docs/icon.png` are made by
`swift tools/dev/make_icon.swift`. The script uses the same shapes as these files. If you change the
mark, change both.

## Colors ("Lake")

| Token | Light | Dark |
|---|---|---|
| Accent (the sun) | `#2f6fa3` | `#7db7e8` |
| Text (the water) | `#18212b` | `#e8eef3` |
| Background | `#f4f6f8` | `#11161c` |
| App icon tile | `#1d2a36` | `#1d2a36` |
| App buttons and links (`Theme.accent`) | `#2f6fa3` | `#3a84cf` |

In the app, the logo uses `Theme.logo` (the accent row above). Buttons and links use `Theme.accent`.
In dark mode it is deeper than the logo blue so that white text on it stays readable. Put white text on
the accent color, never black.

## Wordmark

Write "dMLX" in **Figtree SemiBold** (Google Fonts), with letter spacing −2%. Put the mark to the left
of the word. The mark is as tall as the cap height plus about 40%, and the gap is about one third of
the mark's width.
