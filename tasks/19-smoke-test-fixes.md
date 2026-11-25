# 19 · Fixes from the smoke test

**Status:** Done (2026-11-10): every item fixed; 7 test rounds, the last found no issues (the app side was
clean from round 5). A smoke test of the basic app features after task 17 (commit 9ed4e22),
by two test workers (image flows; everything else), found no crashes or hangs and the issues below.

## Engine
- [x] **A stopped job leaves the image model loaded.** In the `sequential` / `streamed` plans the main weights
      are freed when denoise ends (`memory.py` `after_phase`); a Stop mid-denoise skips that, so 3.23 GB of
      Z-Image stayed resident. The next job (Klein 4B) then ran streamed instead of clearing the idle model,
      because the planner evicts only when no strategy fits. Free what the plan would have freed on any abort,
      and let the planner evict idle models before it falls back to a slower strategy.
      Fixed in two parts. (1) Any stop (cancel, memory pressure, invalid numbers, a crash) now frees what the plan
      frees at the end of a job (`JobGuard.release_after_stop`): encoder and denoise parts in sequential /
      streamed, the encoder in phase_free; eager stays warm. It runs after the `except` block, as in task 17;
      a crash in a job is now caught there too, so its traceback no longer holds the weights. (2) `plan` tries
      each strategy, fastest first, with what evicting idle parts of other unpinned models could give back. It
      evicts least recently used first and measures the room again after each eviction, so it never plans past
      the room. It doesn't evict when that can't help. Tests in `test_stop_frees.py` (stop in each strategy,
      invalid numbers, crash, evict before a slower plan, pinned model kept, no useless eviction, a free that
      gives nothing back). Real check (scratch home, GPU lock): Klein 4B 512² sequential, DELETE at step 2:
      cancelled in 0.9 s, "freed flux2-klein-4b/transformer (2.03 GB)", nothing resident, headroom 4.42 → 6.66
      GB; the next Z-Image job loaded its full 3.23 GB transformer. Z-Image sequential itself didn't fit then
      (needs 5.4 GB, 4.7 usable on the shared Mac); it ran streamed and its stop freed it too. Eviction:
      Z-Image phase_free left 3.23 GB warm (room 3.19 GB), then Klein evicted it and loaded normally (2.11 /
      2.03 GB) instead of streaming beside it (degraded mode allowed only sequential / streamed at the time).
- [x] **HEIC photos can't be attached** (the engine's Pillow has no HEIF support; HTTP 500). The app converts
      HEIC to PNG before upload (app item). The engine answers HEIC with a 415 that says so ("IMG_0001.HEIC is a
      HEIC photo, which the engine can't open. Convert it to PNG or JPEG first."). It knows HEIC by name or by
      its `ftyp` bytes. No new dependency.
- [x] **Files that are not images get HTTP 500** and a traceback; they should get a clear 4xx error naming the file.
      `/v1/uploads` and `/v1/images/edits` now answer 415 `unsupported_image` ("notes.txt is not an image dMLX
      can open. Use a PNG, JPEG or WebP image."), logged at info level with no traceback. Test
      `test_files_that_are_not_images_get_a_clear_415` (.txt, corrupt .png, HEIC by name and by bytes, edits).
- [x] **An interrupted download starts again from zero**, leaves the old `.incomplete` file (970 MB) on disk,
      and the progress shows the largest `.incomplete` file, not the current one.
      Cause: huggingface_hub 1.33 writes each attempt to a new `<etag>.<random>.incomplete` (hf_xet and plain
      HTTP alike). It deletes that file on an error but not on a quit, and never resumes it. dMLX now downloads
      the files itself over plain HTTP (`downloads.Downloader`) into the same cache layout. Each file goes to
      `blobs/<etag>.incomplete`, a later pull continues it with a Range request, LFS files are checked against
      their sha256, and a server that ignores the Range starts the file again. Orphans (huggingface_hub's names,
      or other revisions') are deleted when the engine starts and when a download starts; after success no
      `.incomplete` is left; Remove deletes the repo folder and counts a partial download's bytes in `freed_gb`
      (it said 0.00 GB); a repo folder that is a link is never cleaned.
      Progress counts this model's files only and starts at what is on disk. Tests in
      `test_resumable_downloads.py` (fake Hugging Face). Real check with an empty scratch cache, Klein 4B (4.62
      GB): cancel at 2.76 GB (the poll missed 20 %), `partial_bytes` 2,764,185,358, pull again: `done_bytes`
      went on from 2.76 GB and the cut file grew from 489.7 MB, finished at 4,619,704,206 bytes with no
      `.incomplete` left (4.5 GiB on disk, not 5.2), then Remove left the cache empty. The own downloader is
      there to resume, cancel and check files, not for speed: on the dev network plain HTTP and hf_xet both
      ran at about 5 MB/s (test round 4 timed xet at 4.75 MB/s to the end of a 2.15 GB file). Earlier short
      samples made xet look stalled because it writes almost nothing to disk until late in a file.
- [x] **Downloads can't be cancelled** (no endpoint). `DELETE /admin/models/{id}/pull`, see "Download API". Real
      check: 200 `{"cancelled": true}` in 0.38 s, then 404 `not_downloading`. Tests as above.
- [x] **Disk sizes in GiB** (coordinator's addition). `size_gb` in `/v1/models`, the catalog's `size_gb`, the
      Add Model sizes and `freed_gb` are now decimal GB (bytes / 1e9); memory figures stay in GiB. Klein 4B now
      reads 4.62 GB. Saved model files keep `download_gb` in GiB (read back to bytes, never shown). The agent
      models' sizes were checked against Hugging Face and set to their exact decimal GB (8B: 5.78).
- [x] **Refusal says "pick a smaller size" with no smaller size** (coordinator's addition, engine part of "Wrong
      text"). It suggests a smaller size only above 1.1 MP; at the app's ~1 MP sizes it says "Close some apps,
      or choose a lighter model." Test `test_the_refusal_suggests_a_smaller_size_only_when_there_is_one`.

## App
- [x] **Quit does nothing while the first-run sheet is open.** AppKit ignores Quit while a sheet is open
      (`applicationShouldTerminate` is never called). Quit (menu and ⌘Q) is now the app's own command: it closes
      any sheet, then quits. Checked on a scratch home: ⌘Q and dMLX → Quit dMLX both quit with the welcome
      sheet open, and the engine stopped with the app.
- [x] **The welcome sheet comes back after every engine restart** while no image model exists. It now opens by
      itself once per app launch (`OnboardingRule`); an engine restart leaves it closed, and sending with no model
      still opens the setup step. Checked: Set Up Later, then two `kill -9` of my engine: no sheet. Test
      `testWelcomeOpensOncePerLaunch`.
- [x] **The setup sheet's buttons are off-screen on a 14" display** (sheet 772 pt tall). The sheet is now at most
      as tall as the room from the window's title bar down to the Dock (`SheetFit`, updated when the window moves
      or resizes); the model lists scroll, the title and Back / Set Up Later / Download stay. Checked on this
      1512×982 screen with the Dock visible: window at its default place gives a 655 pt sheet with the buttons at
      y = 855–879; window moved down to y = 380 gives a 476 pt sheet, buttons still in view.
- [x] **An image sent before its upload finishes is dropped**; dropping 5 images attaches 3 and drops 2 silently.
      Uploads in flight now show as "Attaching …" chips and hold their place among the 3; a send waits for them
      ("Sends when the image is attached.") and goes out with the image. If that upload fails, the message is not
      sent and goes back to the box. A drop hands all files over at once and keeps their order; the extra ones get
      "2 files were not attached: this model accepts up to 3 images." HEIC/HEIF become PNG with ImageIO
      (upright, full size) before upload. Unsupported files show the engine's message ("Couldn't attach:
      notes.txt is not an image dMLX can open. …"). Checked: a 63 MB PNG attached and sent at once became an
      edit of that image (Klein 4B, 43 s); 5 files gave 3 chips and the toast; a .heic attached. Tests
      `testAttachPlanSaysHowManyWereLeftOut`, `testUploadsInFlightHoldTheirPlace`, `testHEICIsConvertedToAnUprightPNG`.
- [x] **Download cancel and a partial download** shown in the Model Library (with the engine items). Cancel next
      to the progress; a partial download shows "Resume download · 0.2 of 4.6 GB" (from `partial_bytes`) and a
      trash button that deletes it. Quitting during a download asks "A download is running. Quit anyway?".
      Checked against the real engine with an empty cache: download, Cancel (engine kept 0.20 GB), relaunch
      (Resume still shown), Resume (went on from 0.2), Cancel, trash (cache empty again); ⌘Q mid-download showed
      the question and its Cancel kept the app running.
- [x] **Wrong text:** the refusal says "pick a smaller size" when no smaller size exists; "Hugging Face cache"
      in onboarding and the Remove dialog (models are in dMLX's own folder); the no-model composer hint asks
      "edit or create new" instead of saying no model is downloaded. The refusal is the engine worker's fix
      (`memory.py`, see Engine); the app adds no filter. Onboarding now says "Models are saved in dMLX's models folder." (no longer cut
      off); the Remove dialog says "This deletes 4.6 GB from dMLX's models folder on this Mac." With no image
      model the hint says "No image model downloaded" (or "Image model still downloading").
- [x] **GB vs GiB on the same model card** (4.3 GB size, 4.6 GB progress). App side done: the app shows the engine's
      `size_gb` as it is and works out progress and Resume from bytes / 1e9 (`DiskSize`); only RAM figures stay in
      GiB. Test `testDiskSizesAreDecimal`. Engine side done too: `size_gb` is decimal GB (see Engine), so
      the card reads 4.62 GB next to "0.2 of 4.6 GB".
- [x] **A toast covers the attachment chips.** The toast now sits just above the composer, whatever its height.
      Checked with 3 attachments and a 4th: the toast is above the composer. Also fixed: each attachment chip
      stretched to 240 pt, far from its thumbnail; chips now take their own width.

## Fix round 1: app
From test round 1 (app). Checked on a scratch home with my own bundle; the screen was locked, so the checks
used the DevHarness log, process samples and the engine log instead of real-window screenshots.
- [x] **Quit from the Dock, a logout or a shut down was refused while the welcome sheet was open.** That quit
      is an Apple Event, which AppKit refuses with a sheet open before `applicationShouldTerminate` runs. The app
      now handles the quit event itself (`AppDelegate.handleQuit`) and takes the Quit command's path: ask
      during a download, close the sheet, quit. Checked: welcome sheet open (harness: 1 sheet attached),
      `tell application id … to quit` returned 0, the app and both engine processes were gone within 3 s.
- [x] **Files dropped on the text area pasted their paths.** A transparent view over the message field
      (`FileDropCatcher`) takes only file drops and hands them to `attach`, so images attach and other files
      get the engine's message; clicks and plain text drops still reach the field. Checked with a real Finder
      drag in test round 5 (screen unlocked).
- [x] **Quitting during an agent model download did not ask.** The app now tracks the agent's download
      (`agentDownload`, from the poll and right after Onboarding or Settings starts it) and the question names
      it (`QuitQuestion`). Checked with an empty cache: agent download started (`agent=Qwen3-VL 8B`), a quit
      event left the app running with the question open (sample: `handleQuit` → `confirmQuit` →
      `NSAlert runModal`). Test `testQuitAsksDuringAnAgentDownloadToo`.
- [x] **An upload started in one chat landed in the next.** Changing chat now stops the uploads in flight
      and forgets a held message; nothing of them lands anywhere (the composer of the old chat is gone).
      Checked: attach sysorange.heic, New Creation 0.2 s later: no chip, no attachment 15 s later, and the
      engine got no upload for it.
- [x] **A held send had no way out while the engine hung.** "Sends when the image is attached." now has a
      Cancel (uploads stop, the message goes back to the box), and an upload gives up after 120 s
      (`UploadWait.timeout`; the client's limit was an hour). Checked with the engine stopped (SIGSTOP):
      Cancel gave back "make it deep blue" with "Not sent: attaching was cancelled. …"; a second held send
      came back to the box by itself within 125 s, unsent.
- [x] **The "N files were not attached" toast was replaced at once.** Notes from one drop are merged into
      one toast, and long toasts stay longer (3 to 10 s, `ToastText`). Checked: img1, img2, notes.txt, img4,
      img5 gave 2 attachments and "2 files were not attached: … Couldn't attach: notes.txt is not an image …",
      still shown 7 s later. Test `testOneDropsNotesShareTheToast`.
- [x] **HEIC kept its Display P3 profile.** Converted images are drawn into 8-bit sRGB, and P3 (or other
      non-sRGB) JPEG and PNG files are converted too. Checked: portrait.heic (P3) stored as pixels averaging
      216/38/38, the same as the photo shown in sRGB (the P3 numbers read as sRGB would be 200/62/53). Test
      `testWideColourPhotosBecomeSRGB`.
- [x] **"Will generate" showed while images were attaching.** The hint says "Attaching an image · decided
      when it is attached" until the uploads end. Checked in the harness log. Test
      `testHintWaitsForImagesStillAttaching`.
- Note: in one harness run the 48 MP HEIC uploaded in under a second, so a send meant to be held went out and
  ran a 35 s Klein 4B edit without the GPU lock (11:39; the lock was free when I
  looked right after; later runs took it).

## Fix round 2: app
From test round 2 (app). The screen was still locked (`CGSSessionScreenIsLocked` = 1 at the start and at the
end), so the real Finder drag, the setup sheet on the 982 pt screen, cards and toasts on screen and light/dark
were not run again.
- [x] **A second quit while the download question was open stacked a second alert.** One question at a time
      (`QuitGate`): while it is open, another quit (Dock, ⌘Q, a logout) is turned down and the app comes to
      the front. Checked on a scratch home with an empty cache, Klein 4B downloading: two quit events 3 s
      apart; the second returned at once, the first waited on the question, and the process sample shows one
      `handleQuit → confirmQuit → NSAlert runModal` (round 2 had two nested). Test `testOneQuitQuestionAtATime`.
- [x] **An image attached while a sent message was planned vanished.** When a planned message goes out,
      only its own images leave the composer (`SentAttachments.kept`); images attached or still attaching
      meanwhile stay for the next message. Changing chat still drops everything. Test
      `testImagesAttachedWhileAMessageIsPlannedStay`. Not run with the agent (needs the GPU and the agent model).
- [x] **"This model accepts up to 3 images" was the app's fixed limit.** The limit is now the edit model's
      `max_references` from `/v1/models`, at most the app's 3 (`AttachLimit`). The toast names the limit that
      applies: "FLUX.2 Klein 4B accepts up to 3 images.", or for Qwen Image 2.1 (10) "dMLX attaches up to 3
      images to a message.". The paperclip's tooltip and the Crop / Draw check use the same limit. Test
      `testAttachLimitIsTheModelsOwnOrTheApps`.
- [x] **The route hint said "loads 4.6 GB" with the download size.** It now uses the engine's `memory_gb`
      (agreed with the coordinator): "loads FLUX.2 Klein 4B · about X GB", or "loads FLUX.2 Klein 4B" when the
      engine sends none. Test `testTheLoadHintShowsMemoryNotDiskSize`. Not seen on screen (locked).
- [x] **Checked on a real screen in test round 5:** the Finder drag onto the text field, the setup sheet's
      buttons at the default window place, cards, toasts, light/dark.

## Fix round 1: engine
From test round 1 (engine). Real checks on a scratch home and scratch caches, GPU lock held for model runs.
- [x] **Qwen Image 2.1's text encoder was downloaded by mflux inside the first job** (2.1 GB of
      AiArtLab/zen-image-edit with huggingface_hub and hf_xet: not resumable, not cancellable, not in the
      progress). A model now lists the extra repos it needs (`downloads.EXTRA_REPOS`, by adapter, so models of
      that family added from the app get them too). The pull downloads them with the same resumable downloader,
      and also the 0.32 GB text_fusion tensors, which mflux takes as one byte range of a 14.5 GB file and keeps
      in its own cache. The model is `downloaded` only when all of it is there. `total_bytes`, progress,
      `partial_bytes`, `size_gb` (now 7.6 GB, before and after) and Remove cover the extra repo. Remove keeps
      the extra repo while another model still has it on disk. In a job the adapter reads that copy and cannot
      download (`_local_zen_image_edit`); if it is missing, the job says "download the model again".
      A test checks the repo, revision, files and tensor path against mflux's own, so an mflux update that
      changes them fails the test. Tests in `test_resumable_downloads.py` (download with the model, downloaded
      only when complete, a cut tensor file resumes with a Range, cancel inside the extra repo, Remove keeps a
      shared repo, size before = after). Real check with an empty cache: pull Qwen Image 2.1, total
      7,600,482,867 B (repo 5.51 GB + encoder files 1.77 GB + text_fusion 0.32 GB). Cancel at 5.91 GB, in the
      encoder file (0.05 s, `partial_bytes` = `done_bytes`, one `.incomplete` of 376 MB). Resume went on from
      5.91 GB and finished in 25 min with no `.incomplete` left. `scan_cache_dir`: 2 repos, no warnings. Then
      `HF_HUB_OFFLINE=1` and the first Qwen job (512 px, sequential, 11 s) loaded the encoder (2.37 GB) and
      added no file to the caches. A 20-step job drew "HELLO" correctly. Other paths that could fetch at run time:
      none left. Tokenizers and the agent load from local folders; Klein's small text encoder and PID decode
      are never switched on. `dmlx pull` (dev) now uses the same downloader.
- [x] **The evict-first planner re-planned before the freed memory came back** (50 ms), so it evicted a warm
      Klein and still went streamed. After each eviction it now polls the room until about what was freed
      (90 %) has come back or the job's need fits, for at most 2 s (`memory._evict`; `make_room` too). It
      still evicts one part at a time. Each eviction is logged with the room before and after. Tests with memory
      that comes back 0.4 s late: it still plans eager and evicts the same parts as when memory comes back at
      once; when one part is enough it evicts one (both tests failed with the old 50 ms); the wait is bounded.
      Real check (16 GB emulated, Klein 4B warm, then Z-Image 512², 4 runs): every eviction was needed and its
      memory came back (room 4.31 → 6.39 → 8.18 GB, phase_free; 4.62 → 6.59 → 7.97 GB, eager). No run
      evicted and then ran streamed. Pinned Klein was never evicted. Headroom never went below 1.7 GB.
- [x] **Cancel answered `{"cancelled": true}` while the download still ran.** The status is now `idle` at
      once and the old thread writes no more state; a pull right after waits for that thread, so the Resume is
      not lost. The thread also checks the cancel during the sha256 re-hash and between the listing calls.
      Remove waits for the thread before deleting. Test
      `test_a_cancel_answers_at_once_and_a_resume_right_after_is_not_lost`.
- [x] **`size_gb` went from 4.62 to 4.61 GB after the download.** After the download it is every file on
      disk (not only the weights), and before it, built-in models use their exact bytes (`download_bytes`;
      Klein 9B and Z-Image were 0.01 GB off too). Test `test_the_size_is_the_same_before_and_after_the_download`.
- [x] **A failure while saving an upload was reported as "not an image".** Only decoding gives 415; a save
      error gives 507 `upload_failed` ("Couldn't save photo.png: the disk is full.") or 500. Test
      `test_an_upload_that_cannot_be_saved_is_not_called_not_an_image`.
- [x] **Raw download errors reached the user.** `error` is a plain sentence (full disk, refused, not found,
      rate limited, server trouble, connection lost, damaged file). A full disk or a 4xx is not retried, so a
      full disk errors at once instead of after 60 s. Test `test_download_errors_read_as_plain_sentences`.
- [x] **`unknown_model` text had stray quotes.** It now uses the KeyError's message ("Unknown model 'nope'.
      Known: …"), as does the API's 404 handler. Test `test_unknown_model_message_has_no_stray_quotes`.
- Also: with several partial files, the progress could go back while one of them downloaded; it now counts
  the later ones too. Test `test_progress_never_goes_back_with_several_partial_files`.

## Fix round 2: engine
From test round 2 (engine). Real check on a scratch home, a scratch cache linking the real models, GPU lock held.
- [x] **An eager plan for a cached prompt loaded the text encoder it did not count** (older than task 19). The
      plan left the encoder out of its need (the prompt's embedding was cached), then `JobGuard.start` loaded
      every part of an eager plan; on 16 GB swap grew 8.26 → 11.0 GB and the job stopped for memory pressure.
      The plan now carries what it skipped (`Plan.skip`) and the guard loads exactly what was counted, in every
      strategy. If the pipeline needs the skipped encoder after all (its cache had the prompt with another
      negative prompt, or MOSS's plain-sentence retry), the guard first frees the transformer unless there is
      clearly room, frees the encoder after the encode, and reports `unplanned_loads`. Qwen Image 2.1 always
      enters the encode phase (it needs the processor that comes with its encoder), so it never says the
      prompt is cached. The job report lists `skipped`. Tests in `test_stop_frees.py`: eager loads only
      transformer + vae and never more than the need, every strategy skips the encoder, an unplanned encode
      never has encoder and transformer together. All three failed before the fix. Real check (16 GB
      emulated, b3 repro): the second Z-Image job with the same prompt planned eager at 5.38 GB, loaded
      `vae, transformer` only, max active 6.38 GB, swap 10.5 → 10.5 GB, done.
- [x] **Two models sharing an extra repo, downloading at once, damaged a shared file.** Each file now has
      one writer (a lock per file); the other download waits, then reuses the finished file or continues
      the part left behind. The check now reads the file on disk: its size, then its sha256 (before, the
      hash covered the bytes received, so another writer's bytes passed). The tensor file's header is checked
      too. Snapshot links and refs are replaced in one step, and the end-of-download clean-up skips a part
      another download is writing. Tests `test_two_models_sharing_an_extra_repo_download_at_once` (4 start
      offsets: both done, every file right, each file fetched once) and
      `test_a_file_changed_on_disk_while_downloading_is_not_accepted`. The tester's probe passes 6 of 6
      (it failed 5 of 6).
- [x] **The wait for evicted memory stalled every HTTP request** (3.7 s for a Stop). Plans are now kept one
      at a time by their own lock (`_plan_lock`); the bookkeeping lock is only held for short reads, so
      readers never wait for an eviction. `GET /v1/models`, `GET /admin/memory` and `DELETE /v1/jobs/{id}` run
      in a thread, off the event loop. A Stop during planning now ends the evictions (the job is cancelled
      before the next one). Test `test_requests_answer_during_an_eviction_wait_and_a_stop_ends_the_evictions`
      (uvicorn, 2 s per eviction: every request under 0.5 s, the warm model keeps a part). It failed before.
- [x] **The app showed the disk size as "loads X GB".** `GET /v1/models` now sends `memory_gb`, see
      "Download API". Agreed with the coordinator; the app uses it. Test `test_models_report_the_memory_a_job_needs`.
      Real values here: Klein 4B 5.76, Klein 9B 10.76, Qwen Image 2.1 8.89, Z-Image 7.58 GB. The estimates
      before download are within 0.01 GB of these.

## Fix round 3
From test round 3 (engine and app). One worker.
- [x] **Engine: a download cancel took 10 s on a stalled connection.** The cancel joined the download thread
      for up to 10 s, and the thread only checked between chunks. Now the cancel answers without waiting and
      shuts the socket the download is reading, so a stalled read ends at once; a cancel during a retry's
      pause ends it too. A pull right after still waits for the old thread (fix round 1). Test
      `test_a_cancel_on_a_stalled_connection_answers_at_once_and_the_thread_ends` (a real HTTP server that
      sends one chunk, then nothing; read through the real httpx client): fails without the fix (thread
      still reading). Real check: a pull of Klein 4B into an empty cache through a local proxy that stops
      passing bytes. Stalled at 62.9 MB (no bytes for 4 s), the DELETE answered 200 in 0.003 s and the
      thread logged "cancelled" the same moment. Resume went on from 62.9 MB, a second cancel took 0.006 s,
      and Remove cleaned up.
- [x] **Engine: a Stop during planning ended the job only after the eviction wait and the eager loads.**
      The Stop is now checked before each eviction, during the wait for memory (it stops the wait), when
      the plan is made, and before each part `JobGuard.start` loads. A job stopped before it ran frees every
      part it loaded itself (parts that were warm before stay). Tests in `test_stop_frees.py`:
      `test_a_stop_during_an_eviction_wait_ends_the_job_at_once` (under 0.5 s; was the full 2 s),
      `test_a_stop_while_an_eager_plan_loads_ends_the_job_and_frees_what_it_loaded`,
      `test_a_job_stopped_before_it_ran_keeps_parts_that_were_warm_before`. All three fail without the fix.
      Real check (GPU lock held): Z-Image eager at 512², DELETE at its first "loading" event. The job said
      `cancelled` 0.17 s later; it had loaded only the VAE (0.15 GB) and freed it. The eviction-wait case
      was checked with fake adapters only.
- [x] **App: a partial download under 0.1 GB read "Resume download · 0.0 of 4.6 GB".** Amounts under 0.1 GB
      now show in MB ("Resume download · 40 MB of 4.6 GB", trash tooltip "Delete the 40 MB downloaded so
      far"), never "0 MB" for something kept. Test `testASmallPartialDownloadShowsMegabytes`. Not seen on
      screen.

## Fix round 4
From test round 4 (engine and app). One worker.
- [x] **Engine: a half-downloaded agent model counted as installed.** The downloader links each file into the
      snapshot as soon as it is done, and `local_vlm.locate()` only asked for `config.json` and the weight
      files present. After a quit inside Qwen3-VL 8B's shard 2 it said installed, with shard 2 and the
      tokenizer missing. Each agent model now lists every file of its pinned revision it needs (all but
      README.md and .gitattributes; checked against Hugging Face's listing for all four); it is installed only
      when each is in the snapshot, each weight file is complete, and no `.incomplete` is in its blobs. The
      index can't be used: it names shards that don't exist. Agent pulls go through the same resumable
      `Downloader` as image models, so the next pull continues the cut file. Tests
      `test_a_half_downloaded_agent_is_not_installed` and
      `test_an_agent_download_stopped_in_shard_2_is_not_installed_and_the_pull_resumes_it` (fake Hugging Face);
      both fail with the old `locate()`. The real agent folders on this Mac still count as installed. Real
      check, scratch home and empty cache: Qwen3-VL 8B pull, engine stopped (SIGTERM) at 5.40 of 5.78 GB with
      shard 2 at 46 MB. The snapshot had config.json and shard 1 but no shard 2 or tokenizer (old `locate()`:
      installed). After the restart: `installed: false`. The pull answered `done_bytes` 5.40 GB at once and
      finished in 85 s: `installed: true`, all 17 files, no `.incomplete` left.
- [x] **Docs: the line above said hf_xet stalls on this network.** Reworded: the own downloader is there to
      resume, cancel and check files, not for speed; both ran at about 5 MB/s here.
- [x] **App: trashing a partial under 0.1 GB toasted "freed 0.0 GB".** The toast now uses the same MB rule as
      the Resume label ("Removed FLUX.2 Klein 4B · freed 10 MB"), `DiskSize.removedMessage`. Test
      `testTheRemoveToastShowsMegabytesForASmallPartial`; fails with the old "%.1f GB" format. Not seen on
      screen.

## Fix round 5
From test round 5 (engine). One worker.
- [x] **Engine: `POST /admin/models/{id}/load` evicted a running job's model.** The planner now evicts to get a
      faster plan, and its rule "no other model is running" holds for jobs, not for a load. The memory manager
      now remembers the model of the running job, from its plan to its end; a load never evicts it or a pinned
      model. When the load can't fit without them it answers 507 `insufficient_memory` with the usual "needs
      about X GB but only Y GB can be used" text (it was a 500), and a refused load pins nothing. A load beside
      a job also leaves the job's MLX memory limit alone. The app never calls this endpoint. Test
      `test_a_load_never_evicts_the_running_jobs_model` (fake models, a job held in its denoise): the load
      plans a slower strategy beside it, then a 507 when only evicting would fit, and evicts again once the
      job ends. It fails with the fix undone (the load planned eager and freed the job's parts).
- [x] **Engine: nothing guarded the agents' file lists.** Tests now check every catalog agent: a full pinned
      commit, a non-empty `files` with `config.json`, the tokenizer and a complete set of weight files (one
      `model.safetensors`, or every `model-k-of-N` shard with the index). The list itself is the source: the
      repos' own index files name shards that don't exist (Qwen3-VL 8B's names four, Qwen3-VL 4B's two). Each
      agent's revision and a fingerprint of its file list are stored in the test, as checked against Hugging
      Face's listing; a revision bump, a new agent or a list change without updating them fails with a
      message saying what to check. Offline. Tests `test_every_agent_has_a_pinned_revision_and_a_complete_file_list`,
      `test_an_agents_revision_and_files_were_checked_together`, and
      `test_the_file_list_checks_catch_a_missing_shard_or_an_unchecked_revision` (a bumped revision and an
      empty list both fail).
- [x] **Not a bug: Qwen Image 2.1 shows 12.73 GB on this Mac.** That is the real disk use. The copy here was
      downloaded before Oct 25, when the downloader still fetched its unused `text_encoder/` (5.1 GB); a
      clean download is 7.60 GB. Left as is: Remove and download again clears it.

## Fix round 6
From test round 6 (engine). One worker.
- [x] **Engine: a load beside a running job could take the job's working memory.** Round 5's load planned
      against the room it saw, but the job still had parts and working memory to take, so the two plans
      together could pass the budget (real: 5.38 + 3.73 GB planned against 7.94 GB of room). Counting what the
      job still needs isn't exact, so `POST /admin/models/{id}/load` now refuses while a job runs: 409
      `job_running`, "dMLX is busy with a job or the assistant; load the model after it finishes." It loads nothing and pins nothing. The
      load holds the accelerator lock while it runs, so a job also can't start halfway through it. It also
      refuses while the agent is on the accelerator. The app never calls this endpoint. Test
      `test_a_load_during_a_job_is_refused` (replaces round 5's test; fake models, a job held in its denoise):
      409 with nothing loaded or pinned, then 200 eager once the job ends. It fails with the fix undone (200
      instead of 409). `uv run pytest -q`: 407 passed. No real run.

## Download API
Settled by the engine worker (2026-11-10); the app builds to this.
- `POST /admin/models/{id}/pull` starts a download, or continues it from the files already on disk. It answers
  the download state: `{"model", "status", "total_bytes", "done_bytes", "progress", "error"}`.
  `status` is `idle | downloading | done | error`. A cancelled download goes back to `idle`, not a new status.
- `DELETE /admin/models/{id}/pull` stops a running download and keeps its files. It answers 200
  `{"cancelled": true}` at once, without waiting for the download's thread: from then on the status is `idle`
  and the download writes no more progress. The connection it reads is shut, so the thread ends at once even
  on a stalled network; a pull (or a Remove) sent right after waits for that thread, so a Resume right after
  a Cancel is never lost. 404 `{"error": {"code": "not_downloading", ...}}` when none is
  running (`unknown_model` for a bad id).
- A model downloads every repo it needs: Qwen Image 2.1 also gets its small text encoder
  (AiArtLab/zen-image-edit, 2.09 GB). `total_bytes`, `done_bytes`, `partial_bytes` and `size_gb` include it,
  and the model is `downloaded` only when all of it is there. No job downloads anything.
- `error` is a plain sentence for people ("The disk is full. Free some space, then download again; it
  continues from where it stopped.", "Hugging Face refused the download. ..."), not `[Errno 28] ...`.
- Each model in `GET /v1/models` has `partial_bytes`: the bytes already on disk for an unfinished download
  (0 when the model is downloaded or nothing is there). It survives an engine restart, so the app should use
  `!downloaded && partial_bytes > 0` to show a partial download ("Resume"), not `download.status`.
- `done_bytes` counts only this model's files. A resumed download starts at the bytes already on disk, not
  at 0. `total_bytes` is the whole model. All are bytes; show them as decimal GB (bytes / 1e9).
- `DELETE /admin/models/{id}/files` (Remove) also works on a partial download and deletes every file of it.
  It answers 409 `model_busy` while the download runs, so cancel first. It keeps an extra repo that another
  model still has on disk (downloaded, downloading or partly downloaded).
- Each model in `GET /v1/models` also has `memory_gb`: the memory its default plan needs at its default
  size, with every part loaded plus working memory, in GB of 1024^3 bytes (the unit of the memory refusal
  messages and `need_gb`; not the decimal GB of disk sizes). Before the download it is an estimate (the
  download's size plus working memory) and `memory_estimated` is true. It is null when nothing is known.
  `memory_min_gb` is the leanest plan's need, which a refusal compares against (null before the download).
- `POST /v1/uploads`: 415 `unsupported_image` when the file is not an image; 507 `upload_failed` ("Couldn't
  save photo.png: the disk is full.") or 500 `upload_failed` when the image can't be saved.

## Not changed
- Settings → Intelligence → "Let a model on this Mac decide" asks for a raw model folder. Left as is.

## Done when
Every item is fixed with a test where one can be written, and a test round finds no issues.
