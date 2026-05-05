# SGIC – Secure Generative Image Communication (Demo)

## Original Problem Statement
Build a full-stack web demo for the research project "SGIC" with a SENDER page (cover-image upload, secret message + key, embed pipeline, metrics PSNR/SSIM/BPP/Chi-Square/RS, side-by-side cover/stego/diff×50, RGB histograms, comparison table with PROPOSED highlighted in red, robustness testing, ablation study, training graphs) and a RECEIVER page (stego upload, key, message recovery, verification badge, BER, original-vs-extracted diff). Backend must connect to the simplified SGIC pipeline (`/embed`, `/extract`, `/metrics` and friends).

## User Choices
- Backend: simplified faithful pipeline (zlib + key-derived stream cipher + key-seeded LSB).
- Comparison table: pre-computed reference values shown instantly.
- Training graphs: static reference data; ablation runs live per upload.
- Theme: dark academic / cyber-security lab aesthetic, Signal-Red accent.
- Auth: Emergent-managed Google social login (no role separation).

## Architecture
- **Backend (FastAPI + Motor)** under `/api`
  - Auth: `/api/auth/session`, `/api/auth/me`, `/api/auth/logout` (Emergent OAuth, httpOnly cookie, Bearer fallback)
  - SGIC: `/api/embed`, `/api/extract`, `/api/robustness`, `/api/ablation`, `/api/comparison`, `/api/graphs`
  - Pipeline (`sgic.py`): SHA-256 keyed permutation LSB, integrity tag, zlib, scikit-image PSNR/SSIM, chi-square + RS analysis, attacks (Gaussian/JPEG/resize)
- **Frontend (React + Tailwind + Recharts)**
  - Routes: `/login`, `/sender`, `/receiver`, OAuth callback (`#session_id=…`)
  - Components: `ProtectedShell`, `Nav`, `MetricCard`/`Panel`/`StatusPill`, recharts wrappers
  - 6 Sender tabs: Metrics · Visualizations · Comparison · Robustness · Ablation · Graphs

## Implemented (2026-01)
- ✅ Emergent Google OAuth + cookie session storage (7-day TTL)
- ✅ Sender flow: upload → embed → metrics + visualizations + histograms
- ✅ Receiver flow: upload → extract with verification + BER
- ✅ Live robustness (Gaussian σ=5 / JPEG q=75 / Resize ×0.5) — all return PSNR/SSIM/BER/PASS-FAIL
- ✅ Live ablation (No-Encryption / No-Diffusion / Full-SGIC)
- ✅ Static comparison table with PROPOSED row highlighted in red
- ✅ Reference graphs (PSNR vs Epoch, SSIM vs Epoch, Timing, Steganalysis, Loss curves)
- ✅ Dark Swiss/cyber theme, IBM Plex Mono headings, Signal-Red accent
- ✅ Tested: 15/15 backend tests, 9/9 frontend E2E checkpoints PASS

## Backlog / P1
- Replace simplified LSB with the full DDIM diffusion pipeline (GPU job queue + WS progress)
- Add per-image cached comparison values (run all 4 baselines live for the uploaded image)
- Persist embed history per user (Mongo) and add a "Sessions" tab
- Real bit-level diff visualizer between original/recovered messages
- File-format independence: support TIFF / WebP / BMP

## P2 / Future
- Multi-image batch run + dataset-level metrics export (CSV)
- Advanced attacks: cropping, rotation, compression chain
- Steganalysis classifier (CNN) inline detection score
- Researcher collaboration (share a stego URL via signed token)

## Personas
- **Research student** demoing the project at viva — needs an instant, visually impressive walkthrough.
- **Reviewer / professor** validating round-trip correctness and metric claims.
- **Curious tinkerer** uploading their own image to inspect SGIC behavior.
