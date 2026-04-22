# Website Plan — Next.js 15 public site (V2)

> Primary user surface for the DeepFake Detection & Investigation Suite.
> Pairs with: [`VISION.md`](VISION.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`REQUIREMENTS.md`](REQUIREMENTS.md), [`ROADMAP.md`](ROADMAP.md), [`FREE_STACK.md`](FREE_STACK.md).
>
> **Free-tier discipline.** Every service named below must stay on a free / free-tier / self-hosted plan. **No payments, no subscriptions, no premium tier, no `/pricing` page, no Stripe, no Razorpay — ever.** BTech academic project.

---

## 1. Goals

1. Let an anonymous visitor understand the product in ≤ 30 seconds.
2. Let a visitor try a **bundled sample** without signing up.
3. Let a signed-in user **upload a real video**, receive a verdict + evidence, and download a PDF.
4. Surface honesty: engine version, dataset disclaimers, caveats.
5. Be fast: LCP ≤ 2.5 s on 4G mobile; Lighthouse ≥ 90.
6. Be accessible: WCAG 2.1 AA.

Non-goals (V2):

- Real-time / live-camera detection.
- Video calls.
- Collaboration / team workspaces.
- Native mobile app (V4).

---

## 2. Tech stack (pinned)

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **Next.js 15** (App Router, RSC) | Best-in-class DX, Vercel-native, good for SEO + auth patterns |
| Language | **TypeScript 5 strict** | No `any` in production code (QR-07) |
| Styling | **Tailwind CSS 4** + CSS variables | Matches ssm_calender + jyotish ai patterns; fast theming |
| UI primitives | **shadcn/ui** (Radix) | Accessible out of the box |
| Data fetching | **Native `fetch` with React `use()` + SWR for polling** | Simple, no over-abstraction |
| Auth | **Auth.js (NextAuth v5)** with email OTP | Invite-code-friendly |
| Forms | **React Hook Form** + **Zod** | Share schemas with API |
| Charts | **Recharts** | Simple line/bar; bundled small |
| PDF preview | Lazy-loaded `<iframe>` pointing at API `.../report.pdf` | Zero JS PDF lib |
| i18n (V3) | **next-intl** | EN / HI / MR at launch |
| Analytics | **Umami** (self-hosted on Vercel Hobby — free) | Privacy-friendly, DPDP-safe; stays free at our volume |
| Errors | **Sentry — free Developer plan (5 k events/mo)** | Standard; sample aggressively to stay under quota |
| Deploy | **Vercel Hobby (free)** primary; **Cloudflare Pages (free)** as fallback | PR previews on either |
| E2E | **Playwright** | Matches Jyotish AI pattern |
| Unit | **Vitest** + **React Testing Library** | Fast |
| Lint | **ESLint** (Next config) + **Prettier** | Single source of truth for style |

Package manager: **pnpm**. Node 20 LTS pinned in `.nvmrc`.

---

## 3. Information architecture

### 3.1 Public (anonymous)

| Route | Page | Purpose |
|-------|------|---------|
| `/` | Home | Hero, 3-step how-it-works, live demo card, FAQ, CTA to sign up |
| `/how-it-works` | Explainer | Pipeline diagram, per-module deep dive, links to `RESEARCH.md` |
| `/demo` | Demo | Pick from 3 bundled clips (1 real, 2 fakes); read-only result page |
| `/about` | About | Team, thesis context, non-goals, acknowledgements |
| `/privacy` | Privacy policy | DPDP + GDPR posture; what we store, for how long |
| `/terms` | Terms of service | Acceptable use, disclaimer ("not legal evidence") |
| `/research` | Research notes | Summary + link to GitHub `docs/RESEARCH.md` |
| `/contact` | Contact | Form → transactional email |
| `/legal/disclaimer` | Disclaimer | Inline on every result, plus a standalone page |

### 3.2 Onboarding

| Route | Purpose |
|-------|---------|
| `/signup` | Email magic-link signup (no phone / SMS — all SMS providers are paid); invite code gate in V2-beta |
| `/signup/verify` | OTP entry |
| `/signin` | Magic-link sign-in |
| `/onboarding` | One-screen: accept terms + consent scope + pick default language |

### 3.3 Authenticated

| Route | Purpose |
|-------|---------|
| `/dashboard` | Card: "Start new analysis" + last 5 analyses |
| `/analyses` | Full list, paginated |
| `/analyses/new` | Upload widget + progress |
| `/analyses/[id]` | Results page (verdict gauge, per-frame plot, heatmaps, method-bar, PDF button) |
| `/analyses/[id]/report.pdf` | Redirect → API pre-signed URL |
| `/settings/profile` | Name, email, language |
| `/settings/security` | Active sessions, 2FA (V3+) |
| `/settings/data` | Export + delete my data (DPDP) |

> **`/settings/billing` was removed on the free-tier pivot.** Single free tier only; nothing to bill, nothing to cancel.

### 3.4 Admin (role-gated)

| Route | Purpose |
|-------|---------|
| `/admin` | Dashboard: DAU, queue depth, error rate |
| `/admin/users` | User list, activity, role, ban/unban |
| `/admin/analyses` | Global analysis queue with status filters |
| `/admin/abuse` | Reported content review |
| `/admin/invites` | Generate / revoke invite codes (V2-beta) |
| `/admin/audit-log` | Read-only; every admin access to upload data |

---

## 4. Core UX flows

### 4.1 Anonymous demo

```
/  →  click "Try Demo"  →  /demo
  →  pick sample clip (bundled)
  →  POST /api/demo/analyses  (server-side proxy, no auth)
  →  200 with canned JSON (or live cache-backed re-analysis)
  →  render Results UI with a "Sign up to run your own" CTA
```

Latency target: ≤ 1 s (canned). The demo never calls the inference worker.

### 4.2 Authenticated upload → verdict

```
/analyses/new
  →  drag file (MP4 / MOV / AVI) up to the single free-tier limit (100 MB, 60 s)
  →  client validates (MIME, size, length via browser API)
  →  POST /v1/jobs  (FastAPI) (multipart) with auth cookie
  ←  202 { id, status: "queued" }
  →  router.replace(`/analyses/${id}`)
  (page renders skeleton + "Queued — expected ~30 s")
  →  SWR polls GET /v1/jobs/{id} every 2 s
  ←  status → "running" → "done" (or "failed")
  →  render Results
  →  "Download PDF" calls GET /v1/jobs/{id}/report.pdf
```

Status tokens for UI:

- `queued` — spinner + estimated wait (queue length * avg).
- `running` — progress bar (approximate; API returns `progress: 0..1`).
- `done` — full Results render.
- `failed` — error card with taxonomy code + retry button.

### 4.3 PDF report download

Route: `/analyses/[id]/report.pdf` is a thin server handler that redirects to a pre-signed S3 URL (5 min TTL). No PDF library in the client bundle.

### 4.4 Sign-up (invite-only during V2-beta)

```
/signup
  →  enter email + invite code
  →  POST /api/auth/signup/magic-link  →  magic-link email via Resend (free) or Brevo (free)
  →  click link in email (single-use, 15 min TTL)
  →  /onboarding (accept terms, set consent scope, pick language)
  →  /dashboard
```

V2-launch removes the invite-code gate; **sign-up remains free and tier-less**. There is no pricing page.

---

## 5. Design system

### 5.1 Visual language

- **Mood:** Forensic, serious, calm. Not "AI-magic" glitz.
- **Palette (dark-first):**
  - Background: `#0B1020` (deep slate)
  - Surface: `#131B30`
  - Text primary: `#E6EAF2`
  - Text muted: `#8A93A6`
  - Accent (primary, verdict FAKE): `#F25F5C` (coral red)
  - Accent (secondary, verdict REAL): `#4ECDC4` (teal)
  - Accent (uncertain): `#FFD166` (amber)
  - Info: `#6FB7FF` (soft blue)
- **Light mode:** invert luminance, keep accents.
- **Typography:**
  - UI: `Inter` variable, weights 400/500/600/700.
  - Mono (scores, timings): `JetBrains Mono`.
  - Headings: Inter, `font-feature-settings: 'ss01','cv01'`.

### 5.2 Component library

Built on shadcn/ui + Radix:

- `Button`, `Input`, `Select`, `Textarea`, `Checkbox`, `Switch`
- `Dialog`, `DropdownMenu`, `Tooltip`, `Popover`, `Sheet`
- `Table`, `Tabs`, `Toast`, `Alert`, `Banner`
- Domain components (in `website/src/components/analysis/`):
  - `VerdictGauge` — circular gauge with confidence band colour.
  - `PerFrameChart` — Recharts line, shaded by threshold.
  - `AttributionBarChart` — 4-bar horizontal chart with animated fill.
  - `HeatmapPair` — side-by-side spatial + frequency tiles with frame slider.
  - `ScoreCard` — metric with label + tooltip explaining the formula.
  - `EvidenceList` — per-frame top-3 frames with heatmap thumbnails.
  - `DisclaimerBanner` — legally reviewed copy; appears on every result page.
  - `EngineVersionFooter` — engine + model checksum + dataset tag.

### 5.3 Accessibility

- WCAG 2.1 AA baseline. All interactive elements keyboard-navigable.
- Every chart has an accessible summary (e.g. `aria-label="Per-frame fake probability for 30 sampled frames, minimum 0.12 maximum 0.91"`).
- Heatmaps have alt text explaining the model's attention.
- Verdict colour is never the only signal (always paired with an icon + label).
- Prefers-reduced-motion disables gauge animations.

---

## 6. Internationalisation (V3)

- `next-intl`, JSON message files in `website/messages/`.
- Launch locales: `en`, `hi`, `mr`. Others deferred.
- Routing strategy: prefix (`/hi/...`), default `en` stays at root.
- Dates/times: `Intl.DateTimeFormat`; no third-party locale lib.
- Copy sourced from a single `website/content/*.json` per locale; reviewed by native speaker before merging.

---

## 7. SEO strategy

- Static HTML for all marketing pages (Next.js RSC with no client boundary on that tree).
- JSON-LD on `/`, `/how-it-works`, `/demo` (`SoftwareApplication`, `Article`).
- `sitemap.xml` auto-generated via `next-sitemap`; `robots.txt` authored.
- OpenGraph + Twitter card images per marketing page (generated once, stored in `website/public/og/`).
- `next/font` to inline critical font for LCP.
- No external tracker on public pages (**Umami self-hosted** only, first-party domain).

---

## 8. Performance budget

| Metric | Budget |
|--------|--------|
| Initial JS (homepage) | ≤ 120 KB gzipped |
| LCP on 4G mobile | ≤ 2.5 s |
| Lighthouse Performance | ≥ 90 |
| Lighthouse Accessibility | ≥ 95 |
| Time-to-Interactive on `/demo` | ≤ 3 s |
| PDF page bundle | ≤ 20 KB (no PDF lib) |

Guardrails:

- No third-party widgets on public pages.
- No image that isn't `next/image`.
- No blocking font.
- No `"use client"` at layout root.

---

## 9. Security posture

See [`SECURITY.md`](../SECURITY.md) for the root policy. Website specifics:

- HTTPS only; HSTS with `max-age=63072000; includeSubDomains; preload`.
- CSP: nonce-based `script-src 'self' 'nonce-XYZ'`; `default-src 'self'`; `connect-src` allowlist the API origin.
- Cookies: `HttpOnly`, `Secure`, `SameSite=Lax`. Refresh-token rotation.
- CSRF: double-submit cookie pattern on mutation endpoints.
- Rate limit on `/api/auth/*` and `/api/demo/*`.
- Bot protection on signup (Cloudflare Turnstile).
- Content upload: client-side size + MIME check; server-side re-check and magic-byte sniff.
- Uploaded videos stored in a private S3-compatible bucket (**Cloudflare R2 free 10 GB** or **Backblaze B2 free 10 GB**; MinIO on the L4 box in dev) with **private** ACL, pre-signed download URLs only, and a **24-hour lifecycle delete** for every upload (single free tier).

---

## 10. CI/CD

- `.github/workflows/web-ci.yml`: on push/PR to `main` — `pnpm install` → `pnpm lint` → `pnpm typecheck` → `pnpm test` → `pnpm build` → Lighthouse budget assertion against the built site.
- `.github/workflows/web-e2e.yml`: on PR — Playwright against Vercel preview URL.
- Dependabot weekly for `website/`.
- Changeset-free (single-package repo for the website).

---

## 11. Folder layout

```
website/
├── src/
│   ├── app/                     # Next.js App Router
│   │   ├── (marketing)/         # route group, no auth
│   │   │   ├── page.tsx
│   │   │   ├── how-it-works/page.tsx
│   │   │   ├── demo/page.tsx
│   │   │   ├── about/page.tsx
│   │   │   ├── privacy/page.tsx
│   │   │   ├── terms/page.tsx
│   │   │   └── contact/page.tsx
│   │   ├── (auth)/              # signup / signin
│   │   ├── (app)/               # protected
│   │   │   ├── dashboard/page.tsx
│   │   │   ├── analyses/
│   │   │   │   ├── page.tsx
│   │   │   │   ├── new/page.tsx
│   │   │   │   └── [id]/page.tsx
│   │   │   └── settings/
│   │   ├── (admin)/             # admin routes (role-gated)
│   │   ├── api/                 # Next.js route handlers (thin proxies to FastAPI)
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── ui/                  # shadcn primitives
│   │   ├── marketing/
│   │   ├── analysis/            # VerdictGauge, PerFrameChart, …
│   │   └── layout/
│   ├── lib/
│   │   ├── api.ts               # Generated from OpenAPI snapshot
│   │   ├── auth.ts
│   │   ├── format.ts
│   │   └── i18n.ts
│   ├── hooks/
│   ├── styles/
│   └── types/
├── messages/                    # i18n (V3)
├── public/
│   ├── og/
│   ├── icons/
│   └── samples/                 # bundled demo clip thumbnails
├── tests/
│   ├── unit/
│   └── e2e/
├── .env.example
├── next.config.mjs
├── tailwind.config.ts
├── tsconfig.json
├── package.json
└── README.md
```

---

## 12. Launch checklist

Before flipping open signups (V2-launch):

- [ ] Homepage LCP ≤ 2.5 s on 4G throttling.
- [ ] Lighthouse Performance ≥ 90, Accessibility ≥ 95, Best Practices ≥ 95.
- [ ] Privacy + terms + disclaimer pages reviewed by a human (ideally a lawyer-friend).
- [ ] Consent banner (DPDP + cookie).
- [ ] `/demo` works without JS (server-rendered sample JSON → HTML).
- [ ] All 4 verdict states (REAL, FAKE-high, FAKE-uncertain, N/A-no-face) render correctly on `/analyses/[id]`.
- [ ] DPDP data export + delete endpoints tested end-to-end.
- [ ] Sentry (free) + Umami (self-hosted) instrumentation visible in respective dashboards.
- [ ] Rate limits enforced on `/api/auth/*`, `/api/demo/*`, and `POST /v1/jobs` (3/h/IP anonymous, 10/h authenticated).
- [ ] Uptime monitor (**UptimeRobot free, 50 monitors**) configured.
- [ ] `status.<domain>` page live (**Instatus free** or static Next.js page reading `/v1/healthz`).
- [ ] Audit log writes when admin views a user's analysis.
- [ ] CI grep-gate passes: `rg -n "stripe\|razorpay\|pricing\|upgrade\|premium" website/` returns 0 hits (V2L-08).

---

## 13. Open questions (to resolve before V2-beta)

- Single region (Bombay / Mumbai `ap-south-1`) or multi-region? Start single; Cloudflare free CDN sits in front of Vercel Hobby automatically.
- Host Postgres on **Neon free** or **Supabase free**? Start Neon (simpler, no realtime needed).
- Storage: **Cloudflare R2 free 10 GB** (no egress fees) vs **Backblaze B2 free 10 GB**? Start R2 for public previews, MinIO on the college L4 box for private uploads.
- Inference host: **college L4 box** is the primary path. Documented free fallback for anyone without L4 access: **Kaggle free notebooks** (P100/T4, ~30 h/week) for batch, **Google Colab T4** (session-limited) for demos. **No paid GPU hosts (Modal / RunPod / Fly GPU) are permitted.**
- Free-tier clip length cap: 15 s or 30 s? Start 15 s; expand based on observed p95 but never beyond 60 s (NFR-05).
