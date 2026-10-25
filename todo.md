# TethysPath — Remaining Work Items

Tracking items for the TethysPath movement-ecology package. Larval-biology notes that
used to live here have been archived to `docs/HISTORICAL_LARVAL_NOTES.md` (they belong
to the ParticleTracking project).

## Phase 6 — Notebook migration (in progress)
- [x] Move dashboard/Leaflet export logic into `notebooks/TethysPathNotes.jl`; keep
  `export_dashboards` as a no-op shim in `src/pipeline.jl`.
- [x] Document the figure conventions (`FIGURE_CONVENTIONS.md`, "degrade loudly" rule).
- [x] Migrate JLD2 checkpoints to the `TethysPath` module name
  (`scripts/migrate_checkpoint.jl`, run 2026-10-23).
- [x] Delete `src/dashboards.jl` (superseded by the notebook; a byte-exact receipt
  copy is kept at `src/dashboards.jl.backup`, gitignored). The former placeholder
  `.backup` file was empty (0 bytes), so it was replaced by the real copy before
  deletion on 2026-10-23.

## Numerics & validation
- [x] Property-test invariants for coordinate-space helpers and transition kernels
  (`test/test_numerics.jl`, `:numerics` segment).
- [x] NaN/Inf robustness tests for A* / circuit diagnostics (disconnected components,
  unreachable sources).
- [x] Notebook field-mapping regression tests (kernel row sums, `#DATA-CHECK` anchors).
- [ ] Deeper posterior-predictive coverage: verify `brier_*`/`kl_*` CIs are
  bootstrap-consistent with the sample sizes in `validation.summary` (currently trusted
  from `pipeline.jl`).
- [ ] Joint survey density likelihood vs. telemetry fit: sensitivity of the Negative
  Binomial dispersion `r` to survey zero-inflation (Phase 1 follow-up).
- [ ] Verify that `synthesize_adr_results`'s propagator inversion
  (`M_prop = I - v*A - D*L`, inverted to obtain a "transition matrix") is the
  intended model, or construct `Gamma_mean` directly from the advection/diffusion
  operators — flagged by an embedded NOTE in `src/movement.jl`.

## Maintainability (Phase 7 — queued, see `docs/PHASE7_IMPROVEMENT_PLAN.md`)
- [ ] Split `src/movement.jl` (8.6k lines) into focused sub-modules behind the same API
  (telemetry ingestion / kernel & simulation / A* / posterior vis).
- [ ] Type-stabilise hot kernels (avoid `Vector{Any}` accumulation in
  `simulate_trajectories` path ensemble code).
- [ ] Cache expensive spatial transforms alongside mesh construction (`kstep_transition_cache`
  covers kernels; extend to stratum polygons and coordinate scaling factors).

## Documentation
- [x] `docs/movement_analysis.md` Phase 6 rewritten for the notebook workflow;
  `render_html` documented as default-off.
- [x] Coordinate-space assumptions documented at API boundaries (`haversine_distance`,
  `bearing_deg`, `lonlat_to_xy_km`, `effective_resistance_matrix`).
- [ ] Build `docs/` with Documenter and promote inline docstrings to a reference manual
  (currently `docs/movement_analysis.md` is the only user-facing page).
