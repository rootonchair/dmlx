# Known issues — v0.1 beta

| Area | Issue | Workaround / plan |
|---|---|---|
| Install | The app is ad-hoc signed, not notarized. macOS blocks the first launch. | Right-click → Open, or `xattr -dr com.apple.quarantine /Applications/dMLX.app`. Notarized builds need an Apple Developer ID. |
| Hardware | The 16 GB target is validated on a loaded 24 GB Mac and with the 16 GB tier emulated, **not yet on a real 16 GB machine**. | Please report results with the bug template (chip + memory). |
| Speed | Edits take about twice as long as new images (the reference image doubles the sequence): ~37 s at 1024² on a base M5, ~45 s in lowest-memory mode. | Pro/Max chips are proportionally faster. |
| Speed | Lowest-memory mode (used only when memory is very tight) reads the model from disk block by block: about 25% slower. | Close a few apps; the engine picks a faster mode by itself when there is room. |
| Quality | Below 32 GB, Z-Image decodes tiled (visually identical). FLUX.2 Klein uses an exact banded decoder that matches the one-piece decode up to rounding. | — |
| Reproducibility | A fixed seed reproduces the same image within a memory mode. Lowest-memory mode runs the model uncompiled, so its image can differ very slightly (≈1/255 on average) from the same seed in other modes. | — |
| Previews | No live preview frames yet; the card shows progress (and the blurred source image while editing). | Planned: cheap latent previews. |
| Features | No video, no brush/mask edits, no upscaling, no LoRA in this beta. | On the roadmap (`dmlx-spec.md` §13). |
| History | A job that is running when you quit the app is not resumed; it shows as stopped. | Retry from the turn. |
| Cache | Prompt embeddings are cached in memory only and are cleared when the engine restarts. | Planned SSD tier (spec F6). |
| Network | Models download from Hugging Face once; downloads need network access. | After that dMLX works offline. |
| Memory signal | "Available memory" follows macOS's own accounting (free + file cache). On a Mac that is already swapping heavily, dMLX may refuse jobs. | "Try anyway" runs in low-memory mode and stops automatically if the Mac starts to thrash. |
