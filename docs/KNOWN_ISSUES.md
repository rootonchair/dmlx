# Known issues — v0.1 beta

| Area | Issue | Workaround / plan |
|---|---|---|
| Install | The app is ad-hoc signed, not notarized. macOS blocks the first launch. | Right-click → Open, or `xattr -dr com.apple.quarantine /Applications/dMLX.app`. Notarized builds need an Apple Developer ID. |
| Hardware | The 16 GB target is validated on a loaded 24 GB Mac and with the 16 GB tier emulated, **not yet on a real 16 GB machine**. | Please report results with the bug template (chip + memory). |
| Speed | Edits are ~4× slower than new images (reference tokens double the sequence): ~50 s at 1024² on a base M5. | Use Landscape/Portrait or smaller sources; Pro/Max chips are proportionally faster. |
| Quality | Below 32 GB the decoder always runs tiled to avoid an ~8 GB memory spike. FLUX.2 Klein shows a slight color-temperature shift vs. the exact decode (no seams). Z-Image is unaffected. | Exact decode is used automatically on 32 GB+ Macs with headroom. |
| Previews | No live preview frames yet; the card shows progress (and the blurred source image while editing). | Planned: cheap latent previews. |
| Features | No video, no brush/mask edits, no upscaling, no LoRA in this beta. | On the roadmap (`dmlx-spec.md` §13). |
| History | A job that is running when you quit the app is not resumed; it shows as stopped. | Retry from the turn. |
| Cache | Prompt embeddings are cached in memory only and are cleared when the engine restarts. | Planned SSD tier (spec F6). |
| Network | Models download from Hugging Face once; downloads need network access. | After that dMLX works offline. |
| Memory signal | "Available memory" follows macOS's own accounting (free + file cache). On a Mac that is already swapping heavily, dMLX may refuse jobs. | "Try anyway" runs in low-memory mode and stops automatically if the Mac starts to thrash. |
