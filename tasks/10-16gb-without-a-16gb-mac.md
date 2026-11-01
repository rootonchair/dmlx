# 10 · Show 16 GB works by simulation only, then notarize

**Status:** Open (from the competitor comparison, 2026-11-02). No testing on a real 16 GB Mac (none bought,
rented or borrowed): the simulation is the test, so it must be as strict as a real 16 GB Mac.

## Why
Keeping the Mac usable on 16 GB is the headline, and no competitor publishes anything like it (ComfyUI's
low-memory claims are written for GPUs; the usual Mac tip is to lift the MPS memory limit, with a warning
that it "may cause system failure"). So far it is tested on a loaded 24 GB Mac and a simulated 16 GB one
(`DMLX_EMULATE_TOTAL_GB`, `DMLX_EMULATE_STRICT`, task 08). The beta is also not notarized, while Draw
Things installs from the App Store.

## Today
- The simulation tells the engine the Mac has 16 GB and holds the rest in a ballast
  (`python -m dmlx.nohang ballast`): random bytes, so they cannot be compressed, but macOS can still swap
  them out. A real 16 GB Mac cannot do that, so the simulation is gentler than the real thing.
- Install needs right-click → Open or `xattr -dr com.apple.quarantine`.

## To do
- [ ] **A wired ballast:** `nohang ballast --wired` holds the memory as MLX buffers under
      `mx.set_wired_limit`, so macOS can neither compress nor swap it, much like RAM that is not there.
      Check with `vm_stat` that the wired count grows by the ballast size and stays there.
- [ ] Re-run the task 08 phases (`docs/benchmarks/raw/mem16_phases.py`) and a dev-harness session
      (Ministral 3B as the agent, Klein generate and edit) with the wired ballast and everyday apps open.
      Gate: swap-in ≤ 64 MB/s, pressure never critical, no visible hang. Record in `docs/benchmarks/`.
- [ ] Notarize: needs an Apple Developer ID ($99/year; the user's decision and account). Then add
      `notarytool` submit + staple to `tools/bundle_app.sh --zip`, with hardened runtime for the bundled
      Python.

## Done when
The wired-ballast simulation passes the gate, and the release zip opens without a Gatekeeper warning.
