# 10 · Prove 16 GB on a real Mac, then notarize

**Status:** Open (from the competitor comparison, 2026-11-02).

## Why
"Never hangs your Mac, even on 16 GB" is the headline, and no competitor publishes anything like it
(ComfyUI's low-memory claims are written for GPUs; the usual Mac tip is to lift the MPS memory limit, with
a warning that it "may cause system failure"). It is still validated only on a loaded 24 GB Mac and the
emulated 16 GB tier (`DMLX_EMULATE_TOTAL_GB`, task 08). The beta is also not notarized, while Draw Things
installs from the App Store.

## Today
- `docs/KNOWN_ISSUES.md`: "not yet on a real 16 GB machine".
- Install needs right-click → Open or `xattr -dr com.apple.quarantine`.

## To do
- [ ] Run the task 08 phases (`docs/benchmarks/raw/mem16_phases.py`, gitignored: copy it along) and a dev-harness session (agent + Klein generate/edit,
      Ministral 3B as the agent) on a real 16 GB Apple Silicon Mac with everyday apps open. Gate: swap-in
      ≤ 64 MB/s, pressure never critical, no visible hang.
- [ ] Record the results in `docs/benchmarks/` and update the README and KNOWN_ISSUES.
- [ ] Notarize: needs an Apple Developer ID (the user's decision and account). Then add `notarytool`
      submit + staple to `tools/bundle_app.sh --zip`, and use hardened runtime for the bundled Python.

## Done when
A real 16 GB Mac passes the gate, and the release zip opens without a Gatekeeper warning.
