# Known issues — v0.1 beta 2

| Area | Issue | Workaround / plan |
|---|---|---|
| Install | The app is ad-hoc signed, not notarized. macOS blocks the first launch. | Right-click → Open, or `xattr -dr com.apple.quarantine /Applications/dMLX.app`. Notarized builds need an Apple Developer ID. |
| Hardware | The 16 GB target is tested on a loaded 24 GB Mac and on a simulated 16 GB Mac, **not on a real 16 GB machine**. | The simulation holds back the missing memory and applies the same gate (no critical pressure, swap-in ≤ 64 MB/s). |
| Speed | Edits take about twice as long as new images (the reference image doubles the sequence): ~37 s at 1024² on a base M5, ~45 s in lowest-memory mode. | Pro/Max chips are proportionally faster. |
| Speed | Lowest-memory mode (used only when memory is very tight) reads the model from disk block by block: about 25% slower. | Close a few apps; the engine picks a faster mode by itself when there is room. |
| Quality | Below 32 GB, Z-Image decodes tiled (visually identical). FLUX.2 Klein uses an exact banded decoder that matches the one-piece decode up to rounding. | — |
| Reproducibility | A fixed seed reproduces the same image within a memory mode. Lowest-memory mode runs the model uncompiled, so its image can differ very slightly (≈1/255 on average) from the same seed in other modes. | — |
| Previews | No live preview frames yet; the card shows progress (and the blurred source image while editing). | Planned: cheap latent previews. |
| Features | No video, no mask-based inpainting (Crop / Draw paints visible marks that the edit model reads), no upscaling, no LoRA in this beta. | On the roadmap (`dmlx-spec.md` §13). |
| Models | Only FLUX.2 Klein, FLUX.3 Klein, Z-Image, Qwen Image 2.1, ERNIE Image Turbo, MOSS Image, Boogu Image Turbo, Iris 3B Turbo and Krea 2 Turbo models in mflux format can be added; other mflux families need a table entry and a memory measurement (task 15). ERNIE, MOSS, Boogu and Krea 2 are slow (minutes per 1024 px image). Memory estimates for added 9B and base models are derived, not measured on every variant. | Pick an mflux-community version; report variants that misbehave. |
| History | A job that is running when you quit the app is not resumed; it shows as stopped. | Retry from the turn. |
| Cache | Prompt embeddings are cached in memory only and are cleared when the engine restarts. | Planned SSD tier (spec F6). |
| Network | Models download from Hugging Face once; downloads need network access. | After that dMLX works offline. |
| Memory signal | "Available memory" follows macOS's own accounting (free + file cache). On a Mac that is already swapping heavily, dMLX may refuse jobs. | "Try anyway" runs in low-memory mode and stops automatically if the Mac starts to thrash. |
| Agent | Planning a message takes about 3–10 s (more when the agent looks at older images). On 16 GB the agent is freed before each image job, so every message loads it again. | Pick a smaller agent model in Settings → Intelligence → Agent. |
| Agent | The agent can still pick the wrong image or workflow, especially the smaller models (Qwen3-VL 4B is the weakest). It sometimes asks a question again instead of taking a new request. | Name the image ("image 2"), or pick the workflow from the buttons above the message box. |
| Qwen Image 2.1 | Needs about 9.5 GB free and takes about 4 minutes per image; on 16–24 GB Macs it is often refused until other apps are closed. Research license, non-commercial. | Use FLUX.2 Klein for everyday edits. |
