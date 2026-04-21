# Vision — DeepFake Detection & Investigation Suite

> North star of the project. Every feature, doc, and code change must defend the claims made here.

---

## 1. North star

Build a **web-accessible deepfake investigation suite** that, given a face-centric video, returns a forensically useful verdict:

1. A calibrated **REAL / FAKE** score grounded in spatial (XceptionNet) and temporal cues.
2. If FAKE, a **four-way manipulation attribution** — `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures` — with per-class confidence.
3. **Dual visual evidence** — Grad-CAM++ heatmaps from the RGB stream and from the frequency/noise (SRM+FFT) stream, showing *where* the model looked.
4. A **forensic report** (JSON + PDF) with per-frame scores, attribution distribution, technical metadata (device, timing), and the engine/model versions used.
5. All of the above exposed through a **public website** (primary surface for end users) and a **Streamlit research console** (internal).

We are not building a magic truth-detector. We are building a **convergent evidence tool** that a journalist, moderator, student, or analyst can use to form a defensible opinion.

---

## 2. Why this is feasible

- **Task is bounded.** FaceForensics++ (FF++) c23 provides 1 000 real + 4 000 manipulated videos with public splits. Detection at AUC > 0.94 and attribution at > 85 % macro-F1 on identity-safe splits are reproducible targets in the literature (Rossler et al. 2019; Afchar et al. 2018; Dolhansky et al. 2020).
- **All pieces exist as code.** Spatial, temporal, DSAN v3 attribution, Grad-CAM++, fusion LR, Flask API, Streamlit dashboard, and report generator are all in `src/` and `app/` today. What is missing is GPU training runs, real benchmark numbers, website surface, and productionisation.
- **The hard research is done.** Eight audit rounds (v2.2 → v10.2) in `PROJECT_PLAN_v10.md` have killed the obvious foot-guns (SRM in DataLoader, StandardScaler on `[Ss, Ts]`, `global_pool=''` on EfficientNet, warm-up LR init, identity-safe splits, gated fusion input, Grad-CAM target layer, thread safety note).
- **A one-person team can ship.** The full scope fits one student + Cursor agents + one GPU box (L4 via college / cloud credit).

---

## 3. Product philosophy

| Principle | What it means in practice |
|-----------|--------------------------|
| **Evidence over verdict** | Never show just a label. Always show scores, heatmaps, per-frame plot, and the method fingerprint. |
| **Calibrated honesty** | Report that AUC is on FF++ c23 identity-safe, and that generalisation to in-the-wild content is weaker. Never claim 100 %. |
| **Determinism** | Same video + same engine version ⇒ byte-identical JSON. Cache by `sha256(video) ⊕ engine_version`. |
| **Reproducibility** | Pinned Python (3.10) and library versions per `requirements.txt`. Every release tagged with its engine version on the dashboard. |
| **Privacy-first** | Uploaded videos are processed and discarded unless the user opts in to retention. No third-party video is ever used for training. |
| **Explainability is non-negotiable** | Every FAKE verdict ships with at least one spatial + one frequency heatmap. A verdict without evidence is a bug. |
| **Web-first** | The public surface is the Next.js website. Streamlit is a *research* console for us. Mobile is out of scope for V1. |
| **Single source of truth** | `docs/PROJECT_PLAN_v10.md` for technical decisions, `docs/IMPLEMENTATION_PLAN.md` for phasing, `docs/REQUIREMENTS.md` for acceptance, this file for *why*. |

---

## 4. Who we are building for

| Tier | Persona | Primary need |
|------|---------|-------------|
| Public (anonymous) | Curious student, journalist's intern | Upload a short clip, get a verdict + evidence within 30 s |
| Signed-in user | Journalist, moderator, content reviewer | Upload history, downloadable PDF reports, run up to *N* analyses per day |
| Investigator (paid / research) | Fact-checker org, academic | Batch upload, API access, longer videos, exportable evidence bundle |
| Admin (us) | Project owner + maintainers | Dataset health, model versions, queue status, abuse review |
| Researcher (us) | Cohort validation | Streamlit console, raw per-frame JSON, ablation dashboards |

---

## 5. What the user experiences

1. **Landing page.** Clear value proposition, live demo with a bundled sample, "How it works" (one diagram, no jargon), legal disclaimer.
2. **Upload.** Drag-and-drop a video (≤ 100 MB for free tier). A 30 s server-side budget; longer jobs go to a queue and surface progress.
3. **Results page.** Verdict card (colour-coded, confidence band), per-frame score line, manipulation-method bar chart (if FAKE), two Grad-CAM tiles (spatial + frequency), and a "Download PDF" button.
4. **Report.** A deterministic, paginated PDF: metadata, scores, heatmaps, caveats, engine version — the kind of artefact that could be attached to a ticket or a story.
5. **History.** Past analyses indexed by upload time, with the ability to re-open any report. Free tier retains 7 days.

---

## 6. Prediction contract (honesty clauses)

- Every verdict prints **engine version + model version + dataset it was trained on**.
- Confidence bands: `High ≥ 0.80`, `Moderate 0.55–0.80`, `Indicative 0.50–0.55`, `Uncertain < 0.50`. Indicative and Uncertain never show a definitive label; they say "insufficient signal".
- Per-frame analysis is shown so the user can judge consistency themselves.
- A prominent disclaimer states that the system is **a technical tool, not legal or journalistic evidence**, and that **generalisation to unseen manipulation methods is not guaranteed**.
- Cross-dataset evaluation (Celeb-DF v2, DFDC preview) is reported in `docs/TESTING.md` so users understand the drop when leaving FF++.

---

## 7. Success metrics

| Horizon | Metric | Target |
|---------|--------|--------|
| V1 (engine closed) | FF++ identity-safe test AUC | ≥ 0.94 |
| V1 | DSAN v3 attribution macro-F1 | ≥ 0.83 |
| V1 | p95 inference latency on L4, 10 s clip, 10 frames, no Grad-CAM | ≤ 2 s |
| V1 | p95 inference latency with dual Grad-CAM on 3 frames | ≤ 5 s |
| V2 (website live) | Public Lighthouse performance | ≥ 90 |
| V2 | WCAG 2.1 AA on all public pages | Pass |
| V3 (scale) | Free-tier LCP on 4G | ≤ 2.5 s |
| V3 | System uptime excluding scheduled maintenance | ≥ 99 % |
| V3 | Cross-dataset generalisation drop (FF++ → Celeb-DF v2) | ≤ 15 pp AUC drop (reported honestly) |

---

## 8. What we are NOT building (scope discipline)

- **No** realtime video-call deepfake detection.
- **No** deepfake *generation* of any kind, not even for data augmentation. We only consume public academic datasets.
- **No** face / identity recognition or matching. We analyse authenticity, not identity.
- **No** audio-only analysis in V1. Audio-visual fusion is a V3 stretch goal.
- **No** training on user uploads. Ever.
- **No** mobile app in V1 or V2. Capacitor wrap is V3+.
- **No** Docker / Kubernetes orchestration unless hosting clearly requires it (student-scale deployment targets Fly.io / Railway / Modal).
- **No** claims of 100 % accuracy anywhere in UI, marketing, or docs.

---

## 9. Failure modes we accept and document

- **Unseen manipulation method**: GAN-only face-swap on in-the-wild data may produce weak signals. We report the calibration tier honestly.
- **Heavy compression (FF++ c40)**: detection AUC drops. We surface this in the About page.
- **Low-resolution faces**: a face-quality gate will reject / warn under a minimum bounding-box size.
- **No face detected**: pipeline returns `N/A` verdict with a clear explanation, not a silent 0.5.
- **Concurrent Grad-CAM requests** on a single API instance can corrupt SRM state (BUG-001). We mitigate by making each request spawn a fresh wrapper and by serialising CAM work behind a queue.

---

## 10. Relationship to other living docs

- `docs/PROJECT_PLAN_v10.md` — technical design (stable).
- `docs/REQUIREMENTS.md` — formal FR/NFR (stable; extended in V2 for website).
- `docs/IMPLEMENTATION_PLAN.md` — phase-by-phase execution.
- `docs/ROADMAP.md` — strategic horizon.
- `docs/WEBSITE_PLAN.md` — public website.
- `docs/ARCHITECTURE.md` — systems view.
- `docs/AUDIT_REPORT.md` — current defects vs the vision in this file.
- `Agent_Instructions.md` — entry point for AI agents working on the repo.

Any PR that changes the answer to "what does this product do, for whom, with what guarantees?" must update this file.
