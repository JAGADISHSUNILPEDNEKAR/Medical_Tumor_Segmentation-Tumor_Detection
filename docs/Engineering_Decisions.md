# Architecture & Engineering Decision Audit
## Medical Image Segmentation & Tumor Detection — BraTS 3D U-Net Pipeline

*Grounded in the supplied `Complete_PRD.md` and the current implementation state: BraTS 2021 Task 1 (1251 validated cases, patient-level split 1001 train / 250 val), custom 3D U-Net at `base_channels=32` (~18.8M params), 128³ patches, trained on a single Colab Tesla T4 (14.56GB VRAM), 30 configured epochs with early-stopping patience 10 — run interrupted at epoch 5/30 (`proxy_val_mean_soft_dice = 0.2519`).*

### How to read the tags used below
- **[PRD]** — stated explicitly in the document you supplied
- **[IMPL]** — from your actual implementation state, not the PRD
- **[INFER]** — my engineering inference, not stated outright anywhere
- **[KNOWLEDGE]** — established ML/systems knowledge, project-independent
- **[CRITIQUE]** — my assessment or recommendation, not a PRD claim
- **[VERIFY]** — an external claim in the PRD's own §18 Research Addendum that needs independent checking before you repeat it to an interviewer or in a paper

---

## Executive Verdict

**Is the architecture coherent?** Yes, directionally — it's the right model family, the right metrics, and the right shape of system (async job service, decoupled ML pipeline, disclaimer-gated UI) for a 3-month, single-GPU, 1–2 person academic project. The document is unusually rigorous for a course deliverable: it has a real ERD, a real evaluation-edge-case policy, and an honest MVP/V1/Stretch/Out boundary. But it is a **specification**, not a **result** — and right now there's a real gap between what the docs imply (`headline_metrics.mean_dice: 0.81` in the API spec) and what you've actually measured (`proxy_val_mean_soft_dice = 0.2519` at epoch 5/30, run incomplete). That gap is the single most important thing to manage before any interview or demo.

**Strongest decisions**
- Custom 3D U-Net, nnU-Net-*inspired*, not nnU-Net-*dependent* — correctly optimizes for the actual learning objective and the 3-month/1-GPU/solo-team constraint (§ADR-001).
- Compound Dice + CE loss — the field-standard, low-risk choice for foreground/background imbalance in BraTS; nothing here needs defending hard.
- HD95 over raw Hausdorff distance — correct and well-justified (outlier robustness), and you can explain the mechanism, not just cite the convention.
- Async job queue + polling instead of synchronous inference — correctly avoids HTTP timeouts on a task that takes 1–2 minutes.
- Explicit edge-case conventions for Dice/HD95 (both-empty, one-sided-empty) — most student projects silently get this wrong; yours documents the convention, which is exactly what a reviewer checks first.
- Honest non-clinical-use framing carried through *every* layer (UX, security, README) rather than bolted on as a disclaimer — this is what turns "cool demo" into "credible prototype."

**Weakest decisions**
- Optimizer is **not actually decided** — the TRD lists "SGD with Nesterov momentum **or** AdamW as an easier-to-tune fallback" as if both were the plan. That's not a documented trade-off, it's an undecided hyperparameter dressed up as a sentence.
- No stated held-out **test** set distinct from the validation set used for checkpoint selection — you have train/val (1001/250), but if `best_model.pth` is selected *on* val and then val is also the number you report, that number is optimistically biased. This is the kind of thing a "senior engineer" will ask about first.
- Connected-component filtering threshold (`min_voxels=50`) is presented as a fixed constant with no sensitivity analysis, on a task where the smallest sub-region (enhancing tumor) is exactly the class most likely to be *legitimately* small.
- Several NFRs are qualitative where they should be numeric: "decimated to a bounded triangle budget," "polling `GET /results/{id}`" with no interval, "upload progress indicator" with no size/rate assumption.
- The API spec's example response (`mean_dice: 0.81`) reads as an aspirational target that could easily get quoted as an achieved result if you're not careful about tense in a demo or interview.

**What a top ML engineer would challenge immediately**
1. "You say nnU-Net-inspired — walk me through one place where your pipeline actually diverges from what nnU-Net would do automatically, and why."
2. "Your optimizer section names two options. Which one did you run, and what evidence do you have that it was the right call?"
3. "Is `0.81` mean Dice something you measured, or the number from BraTS leaderboards you're targeting?"
4. "Your CC-filtering threshold is 50 voxels — did you check what that does to enhancing-tumor Dice specifically?"
5. "What's your actual test-time protocol — is `best_model.pth` selected on the same data you're reporting Dice on?"

---

## Research Credibility Assessment

Your mentor's original pushback — "the 3D visualization isn't novel, radiologists already have 3D imaging" — was aimed at the *visualization layer*, not the segmentation model. That critique is fair as stated (§18 of your own PRD effectively concedes it: react-three-fiber + marching cubes is explicitly framed as "the already-solved part," not the contribution) but the mentor's proposed fixes (genomics, staging, symptom-based diagnosis) are the wrong direction — your own PRD's Out-of-Scope table correctly identifies *why*: BraTS contains no genomic sequencing data, no symptom data, and brain tumors aren't TNM-staged. Pursuing those would trade a defensible project for an indefensible one.

On the 4-tier scale:

| Tier | Where you are |
|---|---|
| 1. CRUD app with a model attached | **No** — the model is the actual bottleneck of the system, not a bolt-on. |
| 2. Legitimate ML engineering project | **Yes, today.** Correct architecture, correct metrics, correct edge-case handling, honest scope boundary. |
| 3. Serious medical-AI research prototype | **Not yet — but buildable.** You have the metrics and the pipeline; you're missing a completed training run and a genuinely non-crowded contribution claim. |
| 4. Production-oriented ML system | **Not the goal, and shouldn't be** — MVP explicitly and correctly excludes auth, PACS/EHR, regulatory artifacts. Don't let scope creep pull you here. |

**What moves you from Tier 2 to Tier 3 without uncontrolled scope expansion** — your own §18.9 already found this, and it's the right call: **case-retrieval-augmented, quantitatively-grounded report drafting**, radiologist-in-the-loop. It composes almost entirely from things already in your V1 tier (tumor volume/breakdown report, case history, checkpoint registry) plus one new component (embedding + similarity search + LLM drafting grounded in *your own* Dice/HD95/volume numbers). This directly answers the mentor's "not novel" critique — visualization isn't the contribution, *grounded retrieval-augmented reporting on top of your own model's outputs* is — without requiring genomics data you don't have or a staging system that doesn't apply to brain tumors.

**[VERIFY]** before repeating any of these in a paper or interview: the specific accuracy ranges cited for 2D-backbone comparisons (mid-80s to 99%), the RSNA-MICCAI MGMT external-validation "~80% no-better-than-chance" figure, and the claim that no published system yet combines RAG with BraTS-style quantitative grounding. These are plausible and well-sourced in your addendum, but "no one has done X" claims specifically have a short shelf life and are worth a fresh search before you commit to them in writing.

---

## Decision Matrix

| Decision | Chosen | Key alternatives | Primary reason | Key constraint | Major trade-off | Evidence required | Confidence |
|---|---|---|---|---|---|---|---|
| Architecture family | Custom 3D U-Net | 2D/2.5D U-Net, V-Net, SegResNet, UNETR, Swin UNETR, MedNeXt | Direct implementation satisfies the *learning* objective; 3D captures true volumetric context BraTS needs | Single T4 (14.56GB), 3-month solo timeline | Forgoes transformer long-range context and pretrained-backbone options | Ablation vs. a MONAI SegResNet baseline on your own split | **Strong** for the stated constraints; **Moderate** for pure accuracy |
| nnU-Net *ideas*, not framework | Custom pipeline, nnU-Net-inspired | Official nnU-Net, MONAI Auto3DSeg | Learning objective + reproducibility within your own codebase | 3-month timeline can't absorb nnU-Net's own config-search overhead | Loses nnU-Net's automatic configuration + 5-fold ensembling gains | Must name at least one concrete divergence point when asked | **Moderate** — defensible, but only if you can name the divergence |
| Loss | Dice + CE (compound) | Focal, Tversky, Focal-Tversky, Generalized Dice, plain CE/Dice | Field-standard; CE gives stable per-voxel gradient, Dice handles imbalance | BraTS foreground is a small fraction of the volume | Doesn't explicitly reweight the smallest class (enhancing tumor) beyond what Dice already does | Per-class Dice trend across epochs, esp. enhancing-tumor class | **Strong** |
| Optimizer | **Undecided** (SGD+Nesterov *or* AdamW) | Adam, AdamW, SGD, LAMB | Not resolved in the PRD | — | This is a real gap, not a documented trade-off | A short ablation on a data subset before committing | **Weak** — flagged as a hidden decision below |
| Normalization layer | InstanceNorm3d | BatchNorm3d, GroupNorm | Batch-independent; correct for batch sizes of 1–4 | 3D patches at 128³ force tiny batch sizes on a T4 | Loses BatchNorm's cross-sample regularization effect | None needed — this one is textbook-correct | **Strong** |
| Activation | LeakyReLU | ReLU, GELU, SiLU/Swish | Avoids dying-ReLU in a deep encoder/decoder | — | Marginal; modern segmentation nets increasingly use GELU/SiLU with little cost | A/B on a small subset if chasing the last 1–2% Dice | **Strong** for "safe default"; **Moderate** for "best possible" |
| Inference | Sliding-window + Gaussian blending | Whole-volume, uniform blending, TTA, ensemble | Bounds memory at 128³; Gaussian avoids seam artifacts at patch borders | T4 VRAM ceiling; 1–2 min/case target | Slower than whole-volume (if it fit); no ensembling by default | Overlap ablation (0.25/0.5/0.75) vs. Dice and latency | **Strong** |
| Post-processing | CC filtering, `min_voxels=50` | No filtering, per-class threshold, learned threshold | Removes spurious noise-voxel islands | — | Fixed threshold risks deleting small, legitimate enhancing-tumor regions | Threshold sweep vs. per-class Dice/HD95 | **Weak as stated** — untuned constant |
| Metrics | Dice + HD95, macro-averaged, per-case | IoU, ASSD, raw Hausdorff, sensitivity/precision, volume error | Field-standard on BraTS leaderboards; case-level avoids large-tumor cases dominating | — | Doesn't surface sensitivity/precision trade-offs (e.g. false negatives on small lesions) | None — this is standard practice | **Strong** |
| Visualization | react-three-fiber + marching cubes | vtk.js, raw Three.js, volumetric ray-casting | Idiomatic in required React stack; sufficient for rotate/zoom/pan/toggle UX reqs | Team already committed to React | No volumetric ray-casting of raw intensities (surface-only) | None — matches stated UX requirements exactly | **Strong** |
| Backend | FastAPI async jobs + polling | Flask, Django, Celery+Redis, WebSockets/SSE | Native async, OpenAPI docs, matches 1–2 min job duration | Single-instance academic deployment, FIFO queue is enough | No horizontal scaling story; polling is chattier than push | None needed at this scale; revisit if concurrent users grow | **Strong** for current scale |
| Database | SQLite (MVP) → Postgres (V1) | Postgres from day one, MongoDB | Zero-ops for single-instance demo; documented upgrade path | Academic/demo scale, no concurrent multi-writer load | SQLite's single-writer lock could bite if async jobs write concurrently | None needed now — revisit only if concurrent job writes appear | **Strong** |
| Deployment | Docker Compose, single GPU VM (stop outside demo windows) | Kubernetes, managed inference endpoint, serverless | Matches team size and budget (~$248/mo) | 1–2 person team, no DevOps headcount | No auto-scaling, no HA | None needed for academic scope | **Strong** |

---

## Detailed Decision Audit

### Decision: Problem formulation — multi-class 3D semantic segmentation
**Chosen:** Voxel-wise 4-class semantic segmentation (background + necrotic core + edema + enhancing tumor) from 4 co-registered MRI modalities.
**Why [PRD]:** Manual slice-by-slice delineation is slow and inter-observer variable; the BRD frames the product's value as *transparency and reproducibility of a research pipeline*, not diagnostic authority.
**Constraints:** BraTS-only scope (no arbitrary DICOM), single-GPU, 3-month window.
**Alternatives:** Binary tumor/no-tumor segmentation; instance segmentation; 2D slice classification (presence/absence); detection (bounding box) instead of dense segmentation.
**Why not alternatives:** Binary segmentation throws away the clinically relevant sub-region distinction that's the whole point of BraTS; detection/bounding-box loses the boundary precision HD95 is designed to measure; 2D classification answers a different, easier question ("is there a tumor") than the one your BRD sets out to solve.
**Trade-offs:** Multi-class dense segmentation is the hardest of these to get right (small-class imbalance, boundary ambiguity between necrotic/enhancing) but is the only one that actually matches the stated business problem.
**Failure modes:** A model that collapses to predicting only the dominant class (edema) still looks "reasonable" on raw pixel accuracy — this is exactly why Dice, not accuracy, is your primary metric.
**When I'd switch:** If the deliverable were reframed as triage ("does this scan need urgent review") rather than delineation, binary segmentation or even 2D classification would be the *better* engineering choice — don't over-build precision the use case doesn't need.
**Evidence needed:** Per-class Dice trend, not just mean Dice, to confirm the model isn't just getting the easy (large, high-contrast) classes right.
**Defensibility:** Strong.
**Interview answer:** "Sub-region delineation, not just detection, is what actually saves a radiologist time — they still have to draw boundaries by hand today. A bounding box or binary mask doesn't remove that step; a correctly-classed voxel mask does."

---

### Decision: Dataset & split — BraTS 2021 Task 1, patient-level 1001/250
**Chosen [IMPL]:** BraTS 2021 Task 1 (Kaggle `dschettler8845/brats-2021-task1`), 1251 validated cases, patient-level split into 1001 train / 250 val.
**Why:** BraTS is the de facto standard for this task — public, pre-registered, skull-stripped, with expert-consensus labels, and every metric convention in your evaluation section (HD95 in mm, per-class Dice) exists *because* BraTS made it the leaderboard standard.
**Constraints:** Public/de-identified data only (§11 Security Design); single-GPU compute budget.
**Alternatives:** A newer BraTS release (2023/2024 expands to Africa, pediatrics, metastases cohorts); combining multiple releases; a private/institutional dataset.
**Why not alternatives:** Newer releases fragment the task into sub-challenges (adult glioma, pediatric, metastases, Africa) with different label conventions — 2021 Task 1 is the cleanest single, well-documented target for a 3-month solo project and has the largest body of comparable published Dice numbers to benchmark against.
**Trade-offs:** A 2021-only model won't have seen the scanner/protocol diversity that 2023+'s multi-institution tracks were built to test — an honest limitation, not a flaw.
**Failure modes / leakage risk [CRITIQUE]:** You describe the split as "patient-level," which is the right unit — BraTS 2021 is one scan per patient (unlike some longitudinal releases), so the main leakage risk you actually need to rule out is simpler: confirm the split file itself was applied correctly (no case ID appearing in both train and val directories) rather than a subtler multi-scan-per-patient issue.
**When I'd switch:** If you later want a domain-shift claim (§18's stretch-tier idea), you'd need a *second*, disjoint-institution test set — BraTS 2021 alone can't support that claim on its own.
**Evidence needed:** A one-line automated assertion in your data-loading code that `train_ids ∩ val_ids = ∅` — cheap, and exactly what a reviewer will ask if you say "patient-level split" out loud.
**Defensibility:** Strong, once the leakage assertion exists; currently **unverified** rather than wrong.
**Interview answer:** "BraTS 2021 Task 1, official patient-level split — 1001 train, 250 val, no case appears in both. It's the version with the most comparable published numbers, which matters because my whole evaluation section is built around being cross-checkable against the leaderboard."

---

### Decision: No held-out test set distinct from the validation set
**Chosen [INFER from IMPL]:** Currently, `best_model.pth` selection and the headline Dice/HD95 you'd report both come from the same 250-case validation split.
**Why this happened:** Not a deliberate choice — it's what a 2-way split naturally gives you, and BraTS 2021's official structure doesn't hand you a labeled third split (the official *test* set is unlabeled, leaderboard-only).
**Constraints:** BraTS 2021's public test set has no released ground truth, so a true third split has to be carved out of your own training pool.
**Alternatives:** 3-way split (e.g., 800/150/100 or k-fold cross-validation with a held-out fold); submit to the official (closed) leaderboard for an unbiased number.
**Why not alternatives:** A 3-way split shrinks an already-modest training set further, which matters more on BraTS's relatively small case count than it would on a huge dataset; k-fold is more rigorous but multiplies training time by k on a single T4 you're already fighting for GPU-hours on.
**Trade-offs:** Reporting val-selected Dice as your headline number is standard practice in most course/demo projects and is not dishonest *if disclosed* — but it is a weaker claim than a truly held-out number, and you should say so proactively rather than have it surfaced as a gotcha.
**Failure modes:** If checkpoint selection (`best_model.pth`) and headline reporting use the *same* 250 cases, your reported Dice is optimistically biased by an unknown (probably small, but nonzero) amount.
**When I'd switch:** If the paper angle in §18.9 goes forward, a genuine held-out slice becomes worth the training-time cost — reviewers will ask this exact question.
**Evidence needed:** Either carve a small held-out test slice now (before more training compute is spent), or explicitly document "val-selected" as a caveat next to every reported number.
**Defensibility:** Moderate — common practice, but currently undocumented as a limitation.
**Interview answer:** "Right now checkpoint selection and reporting share the validation split, which I flag as a known limitation — the officially fair comparison would be the closed BraTS leaderboard, which I haven't submitted to. For the course deliverable I'm being explicit that the number is val-selected, not held-out."

---

### Decision: Preprocessing pipeline (orientation, resampling, foreground-only z-score, cropping)
**Chosen [PRD]:** Canonical reorientation → resample to a fingerprinted target spacing → per-modality foreground-only z-score normalization → crop to foreground bounding box.
**Why:** This is lifted conceptually from nnU-Net's dataset-fingerprinting methodology, which exists precisely because naive global normalization on MRI (unlike CT, which has a fixed Hounsfield-unit scale) is unreliable — MRI intensities are scanner- and protocol-dependent, so per-case, foreground-only normalization is the standard fix.
**Constraints:** Must be *identical* at train and inference time, or you get exactly the preprocessing/inference mismatch your own TRD advanced-viva question calls out.
**Alternatives:** Global (whole-volume, background-included) normalization; min-max scaling; native-resolution (no resampling) processing; adaptive per-case spacing instead of one fixed fingerprinted spacing.
**Why not alternatives:** Global normalization lets skull-stripped background zeros drag down the mean/std used to normalize actual tissue signal; min-max is far more sensitive to a single bright outlier voxel than z-score; native-resolution processing means every case has a different voxel grid, which breaks fixed-size patch-based training; per-case adaptive spacing reintroduces exactly the kind of runtime variability nnU-Net's fingerprinting step was designed to eliminate.
**Trade-offs:** Foreground-only normalization requires a reliable foreground mask (skull-stripping is assumed, not re-derived) — if that assumption breaks on a malformed case, normalization silently degrades rather than failing loud.
**Failure modes:** Nearest-neighbor interpolation for label resampling (correct, preserves discrete classes) vs. any smoothing interpolation for labels (wrong — would introduce fractional/interpolated classes, exactly TEST-02 in your own test plan checks for).
**When I'd switch:** If you ever ingest non-skull-stripped or non-BraTS data, foreground detection would need to be re-derived rather than assumed.
**Evidence needed:** TEST-02 already covers label-resampling correctness; add one more explicit check that the *same* fingerprint config file is loaded at both training and inference call sites (your own advanced viva answer already names this as the mechanism — make sure the code actually enforces it, e.g. a single shared config loader, not two copies).
**Defensibility:** Strong.
**Interview answer:** "Foreground-only z-score, computed per case, is standard for MRI because unlike CT there's no fixed intensity scale — a global stat would be dragged around by the zeroed-out skull-stripped background. Labels are resampled nearest-neighbor specifically to avoid inventing fractional label values."

---

### Decision: Architecture family — custom 3D U-Net over alternatives
**Chosen [PRD/IMPL]:** Custom 3D U-Net, `base_channels=32`, ~18.8M params, symmetric 4–5 stage encoder/decoder, InstanceNorm + LeakyReLU, trained on 128³ patches.
**Why:** Directly satisfies the stated learning objective (implement the architecture, not import it); 3D convolutions capture true volumetric spatial context that 2D/2.5D approaches reconstruct only approximately; fits within T4 VRAM at this channel width and patch size.
**Constraints:** Single T4 (14.56GB), 3-month solo timeline, need for full training-loop ownership/debuggability.
**Alternatives:** 2D U-Net (per-slice), 2.5D (multi-slice-as-channels), V-Net, SegResNet, UNETR, Swin UNETR, MedNeXt, an ensemble of several of these.
**Why not alternatives, under *these* constraints:**
- *2D/2.5D* — computationally cheaper and easier to batch, but discards true 3D context across slices, which matters for a boundary-sensitive metric like HD95; wrong trade for this project's stated goal.
- *V-Net* — architecturally close to a 3D U-Net; not a strong differentiator either way, mostly a historical/implementation-detail choice.
- *SegResNet* — MONAI's own reference BraTS architecture, pretrained checkpoints exist (§18.2) — a legitimate alternative to *starting from scratch*, but starting from scratch is what the learning objective actually asks for.
- *UNETR / Swin UNETR* — transformer-based, benefit from larger training sets and more compute than a single T4 and a 3-month window comfortably support; also meaningfully harder to implement and debug correctly from scratch in the available time.
- *MedNeXt* — a strong 2023+ ConvNeXt-style segmentation backbone, competitive with transformer approaches at lower compute — a genuinely reasonable "if I had more time" answer, worth naming as such in an interview.
- *Ensembling several of the above* — this is literally the current (2025) BraTS-challenge winning pattern (§18.3), but assumes multi-GPU, multi-day training budgets your own TRD doesn't have.
**Trade-offs accepted:** Gives up whatever accuracy edge a pretrained backbone (MONAI SegResNet) or a transformer hybrid might offer, in exchange for full ownership, debuggability, and a training budget that actually fits a T4.
**Failure modes:** At `base_channels=32`, model capacity is on the smaller end for BraTS (many published 3D U-Nets use 32→320 with a wider or deeper progression); if val Dice plateaus well below ~0.75–0.80 on the *full* class set even after full training, capacity — not just training completeness — becomes a live hypothesis.
**When I'd switch:** If you get access to a bigger GPU or more time, MedNeXt or a SegResNet-initialized fine-tune (§18.2) is the natural next step — not a full rebuild, a warm-start.
**Evidence needed:** A completed training run's per-class Dice curve, compared against published 3D U-Net baselines on the same split (roughly Dice 0.75–0.85 whole-tumor on BraTS is a typical published range for a competently-trained baseline 3D U-Net) to know whether you're capacity-limited or just undertrained.
**Defensibility:** Strong for the stated constraints; be ready to name MedNeXt/SegResNet by name as the "if constraints changed" answer, not just gesture at "other architectures."
**Interview answer:** "Given a single T4, a 3-month solo timeline, and a learning objective centered on implementing the architecture myself, a custom 3D U-Net is the rational choice — transformer variants like Swin UNETR need more data and compute than I have, and a pretrained backbone like MONAI's SegResNet would satisfy 'have a working segmenter' but not the actual assignment. If the constraint set changed — more GPU-hours, a team instead of solo — MedNeXt is the concrete next architecture I'd evaluate, not a vague 'try something fancier.'"

---

### Decision: "nnU-Net-inspired" without the official framework
**Chosen [PRD]:** Adopt nnU-Net's *ideas* — fingerprinting, resampling to fingerprinted spacing, foreground normalization, patch-based training with foreground oversampling, compound Dice+CE, sliding-window inference with Gaussian stitching, CC post-processing — without depending on the nnU-Net codebase, its automatic architecture search, or 5-fold ensembling.
**Is "nnU-Net-inspired" technically defensible?** **Yes, conditionally.** It's defensible *if* you can name at least one concrete place your pipeline diverges from what nnU-Net would actually configure automatically for BraTS (e.g., nnU-Net's automatic patch-size/batch-size heuristic based on GPU memory fingerprinting vs. your manually-chosen 128³/`base_channels=32`; nnU-Net's default 5-fold cross-validation ensembling vs. your single train/val split). Say those divergences out loud unprompted — that's what separates "inspired by" from "claims credit for work it didn't do." If you can't name a divergence when asked, the label starts to look like marketing rather than description.
**Learning value:** High — you own every line of the training loop, which is exactly the stated course objective.
**Research credibility:** Moderate on its own; the official-nnU-Net baseline is the thing reviewers actually compare against, so *not* running it means you have no controlled A/B showing your custom pipeline's ceiling relative to the reference implementation.
**Reproducibility:** Actually *stronger* than depending on the framework, for a small solo project — one versioned repo, no external framework-version pinning fragility.
**Implementation risk of divergence:** The most likely silent divergence points are (a) target spacing chosen by inspection/example rather than a genuine per-dataset fingerprinting pass, and (b) patch size/batch size chosen for T4 memory fit rather than nnU-Net's own memory-fingerprinting heuristic — both fine engineering choices, but they are divergences, not just "the same idea implemented differently."
**When I'd switch:** If a reviewer specifically wants an nnU-Net-baseline comparison number, running the actual framework once (even just for a benchmark, not as your production pipeline) closes that gap cheaply.
**Evidence needed:** One documented, named divergence, plus — ideally — one comparison metric against a MONAI SegResNet or MONAI-bundle nnU-Net baseline on your same split, even a partial/small-scale one.
**Defensibility:** Moderate → Strong once a concrete divergence is named on the record.
**Interview answer:** "I use nnU-Net's *methodology*, not its codebase — fingerprinted resampling, foreground normalization, compound loss, Gaussian sliding-window inference. Where I diverge: patch size and channel width are chosen to fit a single T4 rather than nnU-Net's own automatic memory-fingerprinting heuristic, and I'm running a single train/val split rather than nnU-Net's default 5-fold ensemble. That's a deliberate trade of some accuracy ceiling for something I can fully debug and reproduce solo in three months."

---

### Decision: Loss — compound Dice + Cross-Entropy
**Chosen [PRD]:** `L = L_CE + (1/|C_fg|) Σ L_Dice^c`.
**Why:** CE gives a stable per-voxel gradient signal everywhere (including on easy, correctly-classified voxels), while Dice directly optimizes the overlap metric you're actually being evaluated on and is comparatively robust to the severe foreground/background imbalance in BraTS (tumor voxels are a small fraction of a full brain volume).
**Alternatives:** Plain Dice, plain CE, Focal Loss, Tversky Loss, Focal-Tversky, Generalized Dice Loss (GDL).
**Why not alternatives:**
- *Plain CE alone* — dominated by the easy background class; would need heavy class weighting to even approach Dice's behavior for free.
- *Plain Dice alone* — can produce noisy, unstable gradients early in training when predictions are near-random and intersection is near zero; CE stabilizes this.
- *Focal Loss* — designed for extreme foreground/background imbalance with easy/hard example reweighting (originally for 2D detection); Dice+CE already handles BraTS's imbalance reasonably well, and Focal adds a tunable γ hyperparameter with no obvious value for this task.
- *Tversky / Focal-Tversky* — lets you explicitly trade precision vs. recall (useful if false negatives on small tumors are specifically costly); a legitimate upgrade path if enhancing-tumor recall turns out to be the weak point, but adds another hyperparameter (α/β) to tune without evidence yet that it's needed.
- *Generalized Dice Loss* — auto-weights classes inversely by volume, which can help small classes but sometimes over-corrects and destabilizes training on the dominant class; a reasonable thing to A/B once you have per-class Dice numbers showing which class is actually underperforming.
**Trade-offs:** Dice+CE is the safe, well-validated default — it doesn't specifically target your smallest, most clinically important class (enhancing tumor / necrotic core) beyond what Dice's per-class averaging already does.
**Failure modes:** If per-class Dice shows enhancing-tumor consistently far below edema/whole-tumor, that's the signal to move to Tversky or GDL — not a reason to switch loss *before* you have that evidence.
**When I'd switch:** After a completed training run, if per-class Dice reveals a specific imbalance problem Dice+CE isn't solving.
**Evidence needed:** Per-class (not just mean) Dice trend across training.
**Defensibility:** Strong.
**Interview answer:** "Dice+CE is the field-standard for BraTS-style imbalance — CE for stable early-training gradients, Dice because it directly optimizes what I'm evaluated on. If per-class results show enhancing tumor specifically underperforming, Tversky or Generalized Dice would be my next experiment, not my starting point."

---

### Decision: Optimizer & LR schedule — undecided (SGD+Nesterov *or* AdamW)
**Chosen [PRD]:** Not actually chosen — the TRD lists both "SGD with Nesterov momentum (≈0.99)" and "AdamW as an easier-to-tune fallback," with polynomial LR decay.
**[CRITIQUE] This is the PRD's clearest internal inconsistency.** A document can defensibly say "we chose X because of trade-off Y"; it cannot defensibly say "we chose X or Y" and call that a decision. Flag this yourself before an interviewer does.
**Why either might be reasonable:** SGD+Nesterov is nnU-Net's own historical default and, with a well-tuned polynomial schedule, often generalizes slightly better on segmentation benchmarks at convergence; AdamW converges faster and is far more forgiving of a mistuned learning rate, which matters when you have limited GPU-hours (T4, ~34 min/epoch) and can't afford a wasted LR sweep.
**Alternatives:** Adam (no weight decay), plain SGD, LAMB (for larger batch regimes — not relevant here given batch size is tiny).
**Why not alternatives:** Adam without decoupled weight decay is strictly dominated by AdamW for this kind of task; LAMB targets large-batch training you're nowhere near given InstanceNorm's small-batch requirement.
**Trade-offs:** AdamW = faster to get *something* working on a compute-constrained single GPU; SGD+Nesterov = potentially better final Dice *if* you have the epoch budget to let a polynomial decay schedule actually converge — which, given your run was interrupted at epoch 5 of 30, you currently don't.
**Failure modes:** Given your actual training situation (interrupted runs, ~34 min/epoch, free-tier GPU quota pressure), **AdamW is very likely the pragmatically correct choice right now** — it needs fewer well-tuned epochs to reach a reasonable result, which matters when you can't guarantee finishing 30 epochs in one sitting.
**When I'd switch:** Once you have reliable, uninterrupted GPU access and can afford a full 30-epoch (or longer) run, SGD+Nesterov + polynomial decay is worth a controlled comparison run.
**Evidence needed:** A short (5–10 epoch) same-data ablation, AdamW vs. SGD+Nesterov, comparing loss-curve stability and val Dice trend — cheap, and it resolves the single biggest "hidden decision" in the whole document.
**Defensibility:** **Weak as currently written** — this is the one place I'd fix before showing the PRD to anyone technical.
**Interview answer:** "The PRD listed both as options — I resolved it by running AdamW first, because my actual compute situation is interrupted Colab sessions on a free-tier T4, and AdamW needs fewer well-tuned epochs to get a usable signal. I'd revisit SGD+Nesterov with a full polynomial-decay schedule once I have uninterrupted GPU time for a full run." *(Say this only once you've actually made and logged the choice — right now it's still open.)*

---

### Decision: Normalization layer — InstanceNorm3d
**Chosen [PRD]:** InstanceNorm3d throughout encoder/decoder.
**Why [KNOWLEDGE]:** BatchNorm's running statistics become unreliable at the batch sizes (1–4, often literally 1 on a T4 at 128³) that 3D volumetric training forces; InstanceNorm normalizes per-sample, independent of batch size, which is exactly the standard fix and precisely why nnU-Net and most competitive 3D medical segmentation networks default to it.
**Alternatives:** BatchNorm3d, GroupNorm.
**Why not alternatives:** BatchNorm needs a batch size large enough for its running mean/variance to be meaningful — not available here; GroupNorm is a legitimate middle ground (batch-independent like InstanceNorm, but normalizes across channel groups rather than per-instance-per-channel) and is a fair "why not this instead" answer to have ready, but InstanceNorm remains the more battle-tested default specifically for this exact task (nnU-Net's own default).
**Trade-offs:** InstanceNorm discards BatchNorm's implicit cross-sample regularization effect (it can't "see" other samples in the batch at all) — a mild disadvantage that matters more for tasks with abundant data and large batches, i.e., not this one.
**Failure modes:** None specific to worry about here — this is close to the textbook-correct choice given the batch-size constraint.
**When I'd switch:** If GPU memory ever allowed genuinely large batches (unlikely at 128³ on consumer/free-tier hardware), BatchNorm's regularization benefit becomes worth revisiting.
**Evidence needed:** None required to defend the choice itself.
**Defensibility:** Strong.
**Interview answer:** "3D volumetric training at 128³ patches forces batch sizes of 1–4 — BatchNorm's running statistics are unreliable at that scale. InstanceNorm normalizes per-sample, independent of batch size, which is why it's the standard for this exact regime, including in nnU-Net itself."

---

### Decision: Activation — LeakyReLU
**Chosen [PRD]:** LeakyReLU, negative slope ≈0.01.
**Why:** Avoids the dying-ReLU problem (neurons permanently zeroed for negative inputs) in a reasonably deep encoder/decoder, at negligible extra compute cost over plain ReLU.
**Alternatives:** ReLU, GELU, SiLU/Swish.
**Why not alternatives:** Plain ReLU risks dead units, especially early in training with imbalanced gradients; GELU/SiLU are smoother and increasingly standard in newer segmentation architectures (including MedNeXt), often worth a small accuracy edge, but add compute cost and aren't the default nnU-Net historically used — LeakyReLU is the "safe, proven, cheap" choice, not necessarily the "best possible" one.
**Trade-offs:** Marginal — this is one of the lowest-stakes decisions in the whole architecture; not worth spending ablation budget on until everything else is validated.
**When I'd switch:** Only after the model is otherwise well-tuned and you're chasing the last 1–2% of Dice.
**Defensibility:** Strong (as a safe default).
**Interview answer:** "LeakyReLU avoids dead units without adding meaningful compute cost — it's the proven default for this architecture family. GELU/SiLU are the modern upgrade I'd consider if I were chasing the last couple points of Dice, not before."

---

### Decision: Data augmentation — flips, small rotation, scale/intensity/gamma/noise; no elastic deformation
**Chosen [PRD]:** Sagittal-axis flips only (no anterior-posterior or superior-inferior flips), small-angle rotation, light scaling, intensity shift, gamma, Gaussian noise. Elastic deformation explicitly excluded.
**Why:** Respects brain anatomy — flipping along the wrong axis would produce anatomically implausible volumes (a brain isn't symmetric front-to-back or top-to-bottom the way it roughly is left-to-right); elastic deformation is explicitly avoided because it can distort tumor boundary *shape*, which directly corrupts the thing HD95 measures.
**Which are label-safe / must apply identically to image and mask:** All the geometric transforms (flips, rotation, scaling) must be applied identically to image and label; intensity-only transforms (gamma, noise, intensity shift) apply to the image only and never touch the label.
**Alternatives:** Full elastic deformation (nnU-Net's own default augmentation), stronger rotation ranges, MixUp/CutMix-style volume mixing, no augmentation at all.
**Why not alternatives:** Elastic deformation is nnU-Net's actual default and does typically help generalization — its exclusion here is the one place your augmentation strategy is *conservative relative to nnU-Net's own recipe*, worth naming as a deliberate, HD95-motivated deviation rather than an oversight; MixUp-style volume mixing doesn't have an obvious label-mixing semantics for dense segmentation and is rarely used in this domain.
**Trade-offs:** Skipping elastic deformation likely costs a small amount of generalization (this is a real, named trade-off, not free) in exchange for HD95 stability and lower implementation risk (elastic deformation is genuinely fiddly to implement correctly in 3D without distorting the label boundary).
**Failure modes:** None of the current augmentations should corrupt boundary quality; if HD95 turns out worse than Dice would predict, augmentation strength is one hypothesis to check, but a lower-risk one than architecture or training completeness.
**When I'd switch:** Once base training is stable and complete, a controlled elastic-deformation ablation (with vs. without) directly tests whether the conservative choice cost you anything measurable.
**Evidence needed:** With/without augmentation-component ablation on a validation subset, tracking both Dice and HD95 separately (not just Dice).
**Defensibility:** Strong, and notably more defensible than most student projects' hand-wavy augmentation sections because you can articulate *why* elastic deformation was excluded rather than just not mentioning it.
**Interview answer:** "I excluded elastic deformation specifically because it can distort tumor boundary shape, and HD95 is a boundary metric — that's a real trade-off against nnU-Net's own default recipe, made deliberately, not by oversight. Flips are restricted to the sagittal axis only, because the brain isn't anatomically symmetric front-to-back or top-to-bottom."

---

### Decision: Inference — sliding-window with Gaussian-weighted blending
**Chosen [PRD]:** Tile the volume into overlapping 128³ patches, run the model on each, blend overlapping predictions with a Gaussian weight (heavier at patch centers).
**Why:** Bounds GPU memory (can't fit a full BraTS volume through the network at once on a T4); overlapping + Gaussian blending specifically avoids the seam artifacts that plain non-overlapping tiling produces at patch borders, where the network has the least context.
**Alternatives:** Whole-volume inference (if memory allowed), non-overlapping tiling, uniform-weight blending, test-time augmentation (TTA — flip/rotate the input, average predictions), full model ensembling.
**Why not alternatives:** Whole-volume inference simply doesn't fit T4 memory at any reasonable channel width; non-overlapping tiling is cheaper but reintroduces the seam artifacts Gaussian blending exists to fix; uniform blending is a weaker version of the same fix — Gaussian's center-weighting is what actually addresses the "patch edges have the least context" problem, not just "some kind of blending"; TTA and ensembling both improve accuracy at a roughly linear multiple of inference cost, which conflicts with your own ~1–2 min/case NFR unless you have latency budget to spare.
**Trade-offs:** Overlap fraction (0.5 stated) directly trades inference latency against boundary quality — more overlap = smoother stitching but more redundant compute.
**Failure modes:** If overlap is too low, seam artifacts reappear at patch borders even with Gaussian weighting; if too high, you may blow past the 1–2 min/case NFR.
**When I'd switch:** If a future GPU had enough memory for whole-volume inference, sliding-window becomes unnecessary complexity — but on any GPU class realistically available to a student, this stays the right approach.
**Evidence needed:** An overlap ablation (0.25/0.5/0.75) measured against both Dice/HD95 *and* wall-clock latency — this is a two-axis trade-off, and you currently only have the accuracy side justified in the PRD, not the latency side.
**Defensibility:** Strong.
**Interview answer:** "Sliding-window is a memory necessity on a single T4, not a stylistic choice — a full volume doesn't fit. Gaussian blending specifically fixes the seam-artifact problem that plain overlapping-average blending doesn't fully solve, because it downweights exactly the low-context patch edges where errors concentrate."

---

### Decision: Post-processing — connected-component filtering, `min_voxels=50`
**Chosen [PRD]:** Per-class connected-component analysis; components under 50 voxels are discarded as noise.
**Why:** Sliding-window inference can produce small, spurious, disconnected false-positive blobs, especially near patch seams; CC filtering is a cheap, standard cleanup step.
**Alternatives:** No post-processing; a per-class (rather than global) threshold; a data-driven threshold (e.g., chosen by sweeping against val Dice per class); morphological opening/closing instead of pure component-size filtering.
**Why not alternatives:** No post-processing leaves noise voxels in the output, hurting both Dice (false positives) and HD95 (spurious boundary points can dominate a percentile-based distance metric); morphological ops are a reasonable complement but don't replace the connected-component logic needed to actually *remove* isolated blobs rather than just erode their edges.
**Trade-offs — this is the sharpest one in the whole pipeline:** BraTS's enhancing-tumor class is *legitimately* often small. A single global 50-voxel threshold, applied uniformly across necrotic core / edema / enhancing tumor, risks deleting real small enhancing-tumor regions exactly because they're small — the same property that makes noise blobs removable is shared by the clinically smallest, arguably most important class.
**Failure modes [CRITIQUE]:** This is your clearest candidate for a silent, unvalidated design choice that could be actively hurting your enhancing-tumor Dice/HD95 without anyone noticing, because the aggregate mean Dice number wouldn't obviously reveal it.
**When I'd switch:** Immediately worth revisiting once you have any trained checkpoint — this is a near-zero-cost experiment (post-processing, no retraining needed) with real potential upside.
**Evidence needed:** Sweep `min_voxels` ∈ {0, 10, 25, 50, 100, 200}, measured **per class**, specifically watching what happens to enhancing-tumor Dice/HD95 — not just the mean.
**Defensibility:** Weak as a fixed, untuned constant; easily strengthened with one cheap sweep.
**Interview answer:** "The 50-voxel threshold is currently a placeholder, not a validated value — I haven't yet swept it per class, and enhancing tumor is exactly the class most at risk of being over-filtered by a single global threshold, since it's often legitimately small. That's next on my list, not something I'm claiming is tuned."

---

### Decision: Evaluation metrics — Dice + HD95, per-class, macro-averaged, case-level
**Chosen [PRD]:** Dice per class → mean Dice (macro-average across cases, never a single pooled voxel count); HD95 in physical mm (via voxel spacing); documented conventions for both-empty (Dice=1.0, HD95=0) and one-sided-empty (HD95="undefined," never a numeric placeholder).
**Why:** These are exactly the metrics and conventions BraTS itself uses — using anything else makes your numbers incomparable to the literature you're citing.
**Alternatives:** IoU/Jaccard, raw Hausdorff distance, ASSD (average symmetric surface distance), sensitivity/recall, precision, volume error.
**Why not alternatives:**
- *IoU* — monotonically related to Dice (Dice = 2·IoU/(1+IoU)), so reporting both adds no new information; Dice is simply the field convention here.
- *Raw Hausdorff* — dominated by a single worst-case outlier voxel, unstable for model comparison; HD95 discards the top 5% most extreme distances while still penalizing genuine boundary disagreement, which is exactly why it, not raw HD, is the BraTS-leaderboard standard.
- *ASSD* — a legitimate complementary boundary metric (averages over *all* surface points rather than the 95th percentile), less sensitive to any single outlier than even HD95; worth reporting *alongside* HD95 as a robustness check, not a replacement.
- *Sensitivity/precision* — genuinely useful for understanding the false-negative/false-positive balance behind a Dice number, and arguably underused in your current spec; cheap to add since you already have predicted and ground-truth masks.
- *Volume error* — directly relevant to your own V1 "tumor volume report" feature; also cheap to add and would connect your evaluation section to your product feature.
**Trade-offs:** Macro-averaging by case (not pooling all voxels across cases) correctly prevents a few large-tumor cases from dominating the aggregate number — the right choice, but means a handful of very-hard, very-small-tumor cases can swing your mean Dice more than their voxel count alone would suggest.
**Failure modes:** None in the metric definitions themselves — they're correctly specified; the risk is entirely in *implementation* (get the empty-mask edge cases wrong and every downstream number is silently miscalibrated), which is exactly why TEST-01 exists in your own test plan.
**When I'd switch/extend:** Add sensitivity/precision and ASSD as secondary metrics once Dice/HD95 are validated — cheap, and directly strengthens the "research credibility" case.
**Evidence needed:** TEST-01 (unit test against hand-computed synthetic masks) is the right validation and should be run *before* trusting any Dice/HD95 number from real data.
**Defensibility:** Strong.
**Interview answer:** "Dice and HD95, computed per case then macro-averaged, matching BraTS leaderboard convention exactly — including the edge-case handling, which is where most re-implementations quietly diverge. HD95 over raw Hausdorff specifically because raw HD is dominated by one worst-case outlier voxel and is unstable for comparing models."

---

### Decision: Visualization — react-three-fiber + marching-cubes surface mesh
**Chosen [PRD]:** 2D axial/coronal/sagittal slice viewer with overlay toggle, plus a decimated marching-cubes surface mesh rendered via react-three-fiber (rotate/zoom/pan, per-sub-region toggle).
**Why:** Satisfies every stated UX requirement (US-02: rotate/zoom/pan/toggle sub-regions) at much lower implementation cost than a purpose-built medical volume renderer, and integrates natively with the required React frontend stack.
**Alternatives:** vtk.js (purpose-built medical volume rendering, supports true volumetric ray-casting of raw intensities, not just extracted surfaces), raw Three.js without React bindings, WebGPU-based custom rendering.
**Why not alternatives:** vtk.js supports genuine volumetric ray-casting (seeing raw intensity gradients inside the volume, not just an extracted surface) but has a materially heavier learning curve and setup cost, and your stated UX requirements (rotate a *tumor shape*, toggle *sub-regions*) are fully satisfied by a surface mesh — you don't need volumetric ray-casting to meet the spec you wrote; raw Three.js without react-three-fiber's bindings would mean manually managing the React/WebGL lifecycle, adding integration risk for no UX benefit given you're already committed to React.
**Trade-offs:** You give up the ability to inspect raw intensity gradients *inside* the volume (useful for, e.g., seeing partial-volume effects at a boundary) in exchange for meaningfully faster implementation and a simpler, lighter-weight rendering pipeline.
**Failure modes:** Marching-cubes mesh extraction can produce a very high triangle count on noisy masks — decimation is required (correctly noted in the PRD) but the actual triangle-budget number is unspecified (see Hidden Decisions).
**When I'd switch:** If a future requirement needed radiologists to inspect raw intensity texture *inside* the tumor volume (not just its predicted boundary shape), vtk.js's volumetric ray-casting would become necessary — your own ADR-002 already flags this correctly as a "V1/stretch upgrade if needed."
**Evidence needed:** A concrete triangle-count cap, chosen empirically against target frame rate on a "typical laptop GPU" (as your own accessibility requirement states, but doesn't quantify).
**Defensibility:** Strong.
**Interview answer:** "react-three-fiber renders a marching-cubes-extracted surface mesh — sufficient for every UX requirement I actually wrote (rotate, zoom, toggle sub-regions), at much lower implementation cost than vtk.js's volumetric ray-casting, which I don't need unless a future requirement asks radiologists to inspect raw intensity texture inside the tumor, not just its predicted shape."

---

### Decision: Backend — FastAPI, async job queue, `202 Accepted` + polling
**Chosen [PRD]:** FastAPI with async background task execution; every inference-triggering endpoint returns a job ID immediately; results fetched via polling `GET /results/{id}`.
**Why:** Sliding-window inference takes real time (targeted 1–2 min/case) — running it inside a single synchronous HTTP request risks client/gateway timeouts; async + polling is the simplest pattern that avoids that without adding new infrastructure.
**Alternatives:** Flask/Django (synchronous by default), Celery + Redis (dedicated distributed task queue), WebSockets or Server-Sent Events (push instead of poll).
**Why not alternatives:** Flask/Django would need bolted-on async support to match FastAPI's native `async def` + background-task model; Celery+Redis is the *more scalable* answer but is real infrastructure overhead (a message broker, worker processes, deployment complexity) that a single-instance, FIFO-queue, academic-scale deployment doesn't need yet — your own NFR explicitly says "queued FIFO rather than parallel-executed," which a simple in-process background-task queue already satisfies; WebSockets/SSE would remove client-side polling overhead and give instant status updates, which is a genuine UX improvement, but adds connection-management complexity for a marginal benefit at this scale and user count.
**Trade-offs:** Polling is chattier (repeated requests) than a push-based approach, and introduces a "how often" tuning question you haven't answered yet (see Hidden Decisions); in exchange, it needs zero additional infrastructure.
**Failure modes:** At *this* scale, none — the risk is purely: if concurrent-user count grows, FIFO in-process queuing with no worker pool becomes a bottleneck and starts looking like the wrong choice fast.
**When I'd switch:** The moment you have more than a handful of concurrent users, or jobs that can fail and need automatic retry/backoff — that's exactly when Celery+Redis (or an equivalent managed task queue) becomes the right call, not before.
**Evidence needed:** None needed to defend current scale; if you want to preempt the "won't this scale?" question, name Celery+Redis explicitly as the upgrade path, the way your DB section already does for SQLite→Postgres.
**Defensibility:** Strong for stated scale.
**Interview answer:** "FastAPI's native async plus an in-process background task queue is enough for a single-instance, FIFO academic deployment — my own NFR explicitly doesn't require parallel job execution. Celery+Redis is the obvious next step the moment I need multiple concurrent users or automatic job retries, but building that now would be infrastructure the current scale doesn't justify."

---

### Decision: Database — SQLite (MVP) → PostgreSQL (V1 upgrade path)
**Chosen [PRD]:** SQLite for MVP, documented PostgreSQL migration path for V1.
**Why:** Zero-ops, file-based, matches a single-instance academic/demo deployment with no concurrent multi-user write load.
**Alternatives:** PostgreSQL from day one; MongoDB or another document store.
**Why not alternatives:** Postgres from day one adds real operational overhead (a running server process, connection management, migration tooling) that buys you nothing at single-user demo scale; MongoDB's document-flexibility isn't a good fit for the genuinely relational structure you already have (cases → jobs → results → metrics, with real foreign keys) — your own ERD is relational by design, so a relational DB is the right family regardless of which one.
**Trade-offs:** SQLite's single-writer lock could become a real issue *specifically* if multiple async jobs try to write results concurrently — worth a quick check (WAL mode helps; still worth confirming under your actual job-queue concurrency model) even at MVP scale.
**When to actually switch:** **[CRITIQUE]** — challenge the "PostgreSQL later" framing a bit: for a single-instance academic project that never grows past a handful of concurrent users, SQLite may be sufficient *indefinitely*, not just at MVP. The honest trigger for switching isn't a project-phase label (V1) so much as an actual condition: concurrent multi-writer load, or a need for real database-level access control once auth ships.
**Evidence needed:** None needed to defend MVP choice; if pressed on "why not Postgres now," the WAL-mode / single-writer-concurrency point is your strongest concrete answer.
**Defensibility:** Strong.
**Interview answer:** "SQLite is zero-ops and sufficient for a single-instance demo with no concurrent multi-user writes — Postgres buys nothing at this scale except operational overhead. The actual trigger to migrate isn't 'we reached V1,' it's concurrent multi-writer load or DB-level access control once auth ships — either of those, and the migration path is already documented."

---

### Decision: Security — no MVP auth, opaque server-generated IDs, server-side upload validation
**Chosen [PRD]:** MVP ships without gated authentication; session-based auth and RBAC deferred to V1; case/job/result IDs are opaque UUIDs, never sequential integers; every upload is validated server-side (NIfTI header parse, 4-modality shape/affine consistency) regardless of client-side validation.
**Why:** Matches the actual threat model of a single-user/small-class academic demo using only public, de-identified BraTS data — there's no real patient data to protect, so full auth/RBAC would be security theater relative to the actual risk.
**What's a real control vs. theater here:**
- *Real control, appropriate now:* opaque UUIDs (prevents one session from browsing another's cases by guessing sequential IDs, even without full auth) and server-side-only validation (client-side checks are UX convenience, not a trust boundary) — these matter *regardless* of scale.
- *Appropriately deferred, not theater-by-omission:* session-based auth/RBAC — genuinely unnecessary for a no-real-PHI, single-user demo, and correctly scoped to V1 rather than pretended to be already solved.
- *Missing even at demo scale [CRITIQUE]:* upload rate limiting. Even without user accounts, an unauthenticated `/predict` endpoint accepting large file uploads with no rate limit is a resource-exhaustion / cost-blowup risk (your own GPU-hour cost model assumes controlled, demo-window usage) — this is cheap to add (a simple per-IP rate limit at the reverse-proxy layer) and isn't in the current security design.
**Alternatives:** Full auth from day one; API-key gating instead of full sessions; no validation at all (rejected outright — this would be a real gap, not a scope trade-off).
**Why not alternatives:** Full auth from day one is real engineering effort spent on a threat that doesn't exist yet for public research data; API-key gating is a lighter middle ground worth considering specifically to address the rate-limiting gap above without building full session auth.
**Trade-offs:** Current design correctly prioritizes "protect against ID-guessing and malformed/oversized uploads" over "protect against unauthorized access," which matches a no-real-PHI deployment — but leaves a cost-control gap open.
**Failure modes:** An open, unauthenticated, unrate-limited `/predict` endpoint during a public demo window is the concrete failure scenario — someone (accidentally or not) submits enough jobs to exhaust your GPU-hour budget or queue.
**When I'd switch:** Before any deployment window where the URL might be shared beyond your immediate reviewers, add basic rate limiting — this is a same-week fix, not a V1-scale project.
**Evidence needed:** None to defend the current auth deferral; the rate-limiting gap is worth fixing regardless of what an interviewer asks.
**Defensibility:** Strong on auth deferral, Moderate overall once the rate-limiting gap is weighed in.
**Interview answer:** "No auth in MVP is a deliberate, correctly-scoped decision — there's no real patient data, so RBAC would be security theater relative to the actual risk. What *does* matter regardless of auth is opaque IDs and server-side-only validation, both of which are in place. The one gap I'd flag myself: no rate limiting yet on the upload endpoint, which matters for cost control even without a real security threat."

---

### Decision: Deployment — Docker Compose, single GPU VM, stopped outside demo windows
**Chosen [PRD]:** `docker-compose` for local dev (CPU-acceptable); a single cloud GPU VM, lab GPU box, or managed endpoint for the deployed version, explicitly stopped outside active demo/grading windows to control the ~$248/month estimate.
**Why:** Matches team size (1–2), no dedicated DevOps role, and a cost model that assumes intermittent rather than always-on usage.
**Alternatives:** Kubernetes, a managed inference endpoint (e.g., a hosted GPU-serving platform), serverless GPU functions.
**Why not alternatives:** Kubernetes' orchestration value (auto-scaling, self-healing, rolling deploys) is entirely wasted on a single-instance academic deployment and adds real operational learning curve for no benefit at this scale; managed inference endpoints could simplify ops but usually cost more per GPU-hour than a raw VM you fully control the stop/start of — a real trade-off worth naming, not a strictly worse option; serverless GPU functions have cold-start latency that could conflict with your own 1–2 min/case NFR depending on the provider.
**Trade-offs:** Manually stopping the VM outside demo windows is the actual cost-control mechanism — it's a manual/procedural control, not an automated one, which is a fine trade for a project this size but worth naming as a limitation if asked "how would this handle production traffic."
**When I'd switch:** The moment usage becomes continuous/unpredictable rather than scheduled around known demo windows, a managed endpoint's pay-per-inference model likely beats a VM you're manually starting and stopping.
**Evidence needed:** None to defend at current scale.
**Defensibility:** Strong.
**Interview answer:** "Docker Compose plus a single GPU VM, manually stopped outside demo windows, matches a 1–2 person team with no dedicated ops role and a cost model built around scheduled usage, not continuous traffic. Kubernetes' value proposition — auto-scaling, self-healing — is entirely unused at single-instance scale."

---

### Decision: CI/CD & observability — GitHub Actions, PyTest, structured JSON logging
**Chosen [PRD]:** Lint + PyTest + frontend tests on push/PR, CPU-mode PyTorch in CI for speed; structured JSON logs (request ID, job/inference ID, model/checkpoint version, timestamp — explicitly never voxel data or patient-identifying metadata); API/compute/ML-stage latency metrics.
**Why:** CPU-mode CI is the correct call — GPU runners are expensive and unnecessary just to validate that preprocessing/metrics/API logic is *correct*, as opposed to *fast*.
**What's essential vs. over-engineering here:** Essential — the unit tests on preprocessing/metrics correctness (TEST-01, TEST-02) and the explicit "never log voxel data or PHI-adjacent fields" rule, since that's a real, cheap-to-violate-by-accident privacy control. Arguably over-engineered relative to project scale — full p95/p99 latency percentile tracking and GPU VRAM peak-utilization monitoring read like production-service observability; reasonable to *keep* since they're cheap to log and directly useful for debugging your own OOM/timeout issues during development, but not something to over-invest further in (e.g., don't build a full metrics dashboard/alerting stack for a single-instance academic deployment).
**Alternatives:** No CI at all; GPU-backed CI runners; a full observability stack (Prometheus/Grafana).
**Why not alternatives:** No CI risks exactly the kind of metric-implementation regression your own risk register calls out as high-impact; GPU CI runners cost real money for a benefit (testing actual inference speed) that isn't what CI is for — CI validates correctness, load/perf testing is a separate, occasional activity; a full Prometheus/Grafana stack is real infrastructure for a single-instance deployment that doesn't need 24/7 alerting.
**Trade-offs:** None significant — this section is close to right-sized already.
**Defensibility:** Strong.
**Interview answer:** "CI runs PyTorch in CPU mode — I'm validating correctness, not benchmarking inference speed, so GPU runners would cost money for no benefit. Structured logs explicitly exclude voxel data and anything patient-identifying, which matters even for public research data as a habit worth having before it matters for real."

---

### Decision: Scope prioritization — MVP / V1 / Stretch / Out
**Chosen [PRD]:** MVP locks ingestion, 3D U-Net training + sliding-window inference, Dice/HD95 evaluation, async API, 2D+3D viewers, disclaimer, and core docs/tests. V1 adds volume/location reporting, slice-level accuracy maps, radiomics, case history, polished dashboard, ensembling, uncertainty quantification. Stretch includes survival prediction, domain-shift testing, self-supervised pretraining, saliency overlays, WHO-grade/MGMT classification heads. Out explicitly excludes genomic sequencing, symptom-based diagnosis, literal TNM staging, DICOM/PACS/EHR/regulatory artifacts, and full nnU-Net-framework dependency.
**Is the tiering logically derived from user value × technical risk × implementation cost × constraints?** **Yes, and unusually well for a student project** — each tier entry in your own §17 table names *why* it's placed there in terms of exactly those axes (e.g., ensembling is V1 specifically because ADR-001 already documented skipping it for time, not capability; survival prediction is Stretch specifically because it needs BraTS's actual provided metadata, which must be confirmed before promising it — a real, named risk, not glossed over).
**Scope-creep risks worth naming explicitly:** The mentor's original ask (genomics, staging, symptom diagnosis) is exactly the scope-creep vector your Out-of-Scope table was built to resist — and it resists it with *reasons* (no genomic data in BraTS, brain tumors aren't TNM-staged, no ethical path to symptom data) rather than just "too hard." That's the right shape of pushback to give a mentor: not "we don't have time," but "the dataset doesn't contain what that feature needs."
**Trade-offs:** The one placement worth double-checking is uncertainty quantification (Monte Carlo dropout, V1 tier) — it's real, valuable, and relatively cheap to add *if* your base model is already dropout-enabled, but if the current architecture wasn't built with dropout layers in mind, it's a bigger lift than its V1 placement implies; worth confirming against your actual `ml/model/` code before promising it in a demo.
**Defensibility:** Strong.
**Interview answer:** "Every tier placement is tied to a specific constraint, not vibes — survival prediction is Stretch because it depends on metadata that may or may not exist in my specific BraTS release, and I said so rather than promising it. The mentor's genomics/staging/symptom suggestions land in Out of Scope for a concrete reason each: BraTS doesn't contain that data, and brain tumors aren't staged the way the mentor's framing assumed."

---

## Hidden Decisions You Need to Make

| Parameter | Why it matters | What should determine it | Candidate values / ranges | How to validate |
|---|---|---|---|---|
| **Optimizer choice** | Directly affects convergence speed and final Dice; currently listed as "either" in the PRD | Actual GPU-time budget (interrupted Colab sessions favor faster convergence) | AdamW (lr ≈1e-4–3e-4) now; SGD+Nesterov (lr ≈1e-2, poly decay) once you have uninterrupted full-run access | Short 5–10 epoch same-data ablation, compare loss-curve stability + val Dice |
| **Learning rate** | Wrong LR is the single most common cause of a stalled 3D-segmentation training run | Optimizer choice above, plus a short LR-range test | AdamW: 1e-4–3e-4; SGD: 1e-2 with poly decay (power ≈0.9, nnU-Net's own default) | LR range test (ramp LR up over a few hundred steps, watch loss) before committing to a full run |
| **Batch size** | Directly interacts with InstanceNorm behavior and VRAM headroom on a 14.56GB T4 at 128³ patches | Actual observed VRAM usage per patch at `base_channels=32` | Very likely 1–2 on a T4 at this patch size/width; confirm empirically, don't assume | Increase batch size until OOM, back off one step; log actual peak VRAM |
| **Mixed precision (AMP)** | Could roughly halve memory/time on a T4, directly easing the "34 min/epoch, run got interrupted" problem | Whether numerical stability holds for InstanceNorm + Dice loss under fp16 | On (torch.cuda.amp) — near-free win if stable | Compare a few epochs with/without AMP: loss curve shape, wall-clock/epoch |
| **Channel width / depth** | `base_channels=32`, "4–5 stages... capped e.g. 320" is a range, not a fixed spec | Available VRAM headroom after batch size/AMP are fixed | 32→64→128→256(→320 capped), 4–5 stages as stated | Only worth adjusting *after* a completed baseline run shows a capacity ceiling, not before |
| **Gradient clipping** | Not mentioned; 3D segmentation losses can spike, especially early with Dice near-zero gradients | Whether loss curve shows spikes/instability in early epochs | Clip at global norm ≈12 (a common nnU-Net-adjacent default) if spikes appear | Log gradient norm per step for the first epoch; only add clipping if norms spike |
| **Gradient accumulation** | Relevant only if true batch size 1–2 is too small for stable InstanceNorm behavior | Whether training loss is visibly noisy step-to-step at batch size 1 | Accumulate 2–4 steps if batch=1 proves noisy | Compare loss-curve smoothness with/without accumulation at fixed effective batch |
| **CC-filtering threshold** | Currently a fixed, unvalidated `50` — risks deleting real small enhancing-tumor regions | Per-class Dice/HD95 sensitivity, not a single global guess | Sweep {0, 10, 25, 50, 100, 200}, likely per-class rather than global | Cheap post-hoc sweep on any existing checkpoint's val predictions — no retraining needed |
| **Target spacing** | Determines resampled voxel grid for every case; wrong choice mismatches train/inference | A genuine dataset-fingerprinting pass (median spacing across your training cases), not an assumed constant | Compute directly from your 1001-case training split's median voxel spacing | Log the actual fingerprinted value into `configs/dataset_fingerprint.yaml`; assert it's loaded identically at train and inference |
| **Sliding-window overlap** | Trades inference latency against boundary smoothness; currently stated as 0.5 with no ablation shown | Your own 1–2 min/case NFR vs. observed Dice/HD95 at each overlap level | 0.25 / 0.5 / 0.75 | Measure both metrics and wall-clock latency at each overlap on a validation subset |
| **Validation/test split strategy** | Currently 2-way (1001/250); checkpoint selection and reporting share the same split | Whether you need a defensible, truly held-out number (e.g. for a paper) | Carve a small (e.g. 30–50 case) held-out slice from val now, before more compute is spent | Confirm `train ∩ val ∩ held_out = ∅`; report held-out number separately from val-selected number |
| **Checkpoint-selection criterion** | "Best" checkpoint needs a defined metric — mean Dice? Worst-class Dice? Combined Dice+HD95? | Which failure mode you care most about avoiding (missing small tumors vs. overall overlap) | Mean Dice is the simplest defensible default; worst-class Dice is stricter and arguably more clinically meaningful | Log both per checkpoint; pick the criterion *before* looking at results, not after |
| **Early-stopping patience** | Already set to 10 — reasonable, but untested since no run has gone that long | Whether val Dice plateaus meaningfully before patience triggers, once training actually completes | Current value (10) is a fine starting point | Just needs a completed run to actually observe |
| **Random seed** | Affects reproducibility claims (your own success metric: "a second person can reproduce this") | Fixed, logged seed for every run you report numbers from | Any fixed int (e.g. 42) — the value doesn't matter, *logging and fixing it* does | Confirm seeding covers PyTorch, NumPy, and any data-shuffling order, not just the model init |
| **Checkpoint save frequency** | Affects both disk usage and how much progress is lost on an interrupted Colab session (directly relevant given your epoch-5 interruption) | Your actual interruption risk (free-tier Colab disconnects) | Save every epoch (not just "best"), given your demonstrated interruption risk | Confirm `last_model.pth` is actually being overwritten every epoch, not just at some coarser interval |
| **Mesh triangle budget** | "Decimated to a bounded triangle budget" has no stated number | Target frame rate on "a typical laptop G, GPU" (your own accessibility requirement) | Start around 50k–150k triangles post-decimation as a first target, then tune to observed frame rate | Measure actual frame rate on a mid-range laptop GPU at a few candidate triangle counts |
| **API polling interval** | Unspecified frontend detail; too aggressive wastes requests, too slow feels unresponsive | Expected job duration (1–2 min) vs. perceived responsiveness | ~2–3 second interval with light backoff is a reasonable default | Basic UX check — does the status update feel responsive without hammering the endpoint |

---

## Architecture Red Flags

🔴 **Critical**
- **Undecided optimizer presented as a decision** (SGD+Nesterov *or* AdamW). This is the PRD's clearest internal inconsistency — fix it by actually running the short ablation described above and updating the doc.
- **Reported/example headline metric (`mean_dice: 0.81`) vs. actual current training state (`proxy_val_mean_soft_dice = 0.2519` at epoch 5/30, interrupted).** If this API example is quoted in a demo or interview as an achieved number, that's a credibility risk you can avoid entirely just by labeling it clearly as illustrative/target until you have a real completed-run number.
- **No held-out test set distinct from the validation set used for checkpoint selection.** Currently a real, if common, source of optimistic bias in any reported Dice/HD95 — cheap to fix (carve a small held-out slice) before it's asked about.

🟠 **Important**
- **CC-filtering threshold (`min_voxels=50`) is an untuned constant** on a task where the smallest class (enhancing tumor) is exactly the one most at risk of being over-filtered.
- **Batch size is never stated anywhere**, despite being the most consequential hyperparameter for InstanceNorm-based 3D training stability at your patch size/VRAM combination.
- **No documented mixed-precision decision**, despite a clear, low-risk opportunity to ease the "34 min/epoch, run interrupted before completion" problem you're actually facing.
- **Checkpoint-save frequency isn't specified**, despite direct relevance to your demonstrated interruption risk (free-tier Colab disconnects) — losing progress on an interrupted run is a real, already-observed cost, not a hypothetical.

🟡 **Minor**
- Mesh triangle budget stated qualitatively ("bounded") with no number.
- API polling interval unspecified.
- No upload/API rate limiting even at demo scale (cheap fix, low current risk).
- Gradient clipping and accumulation not addressed either way.

🟢 **Fine as-is**
- SQLite → PostgreSQL upgrade path and its stated trigger condition.
- Opaque, server-generated UUIDs for cases/jobs/results.
- Docker Compose + manually-stopped GPU VM deployment model.
- Compound Dice+CE loss.
- HD95 as the boundary metric, with documented edge-case conventions.
- react-three-fiber + marching-cubes visualization choice given the stated UX requirements.
- InstanceNorm3d + LeakyReLU as the normalization/activation pairing.

---

## Recommended Experiments

Ordered by **impact × uncertainty × cost** — cheapest, highest-value first:

1. **Finish (or meaningfully extend) the interrupted training run** — resolves the single biggest current gap between documented and actual results; everything else is secondary until you have a real, non-epoch-5 number to reason from. *(High impact, high uncertainty resolved, moderate cost — this is the one that isn't optional.)*
2. **CC-filtering threshold sweep** ({0,10,25,50,100,200}, per class) on any existing checkpoint's predictions — no retraining required, directly addresses a 🟠 flag. *(High impact, near-zero cost.)*
3. **Mixed-precision (AMP) on/off comparison** for a few epochs — directly attacks your actual GPU-time constraint. *(High impact given your stated interruption problem, low cost.)*
4. **Optimizer ablation** (AdamW vs. SGD+Nesterov) on a short, fixed-epoch, same-data run — resolves the 🔴 undecided-optimizer flag with evidence instead of a coin flip. *(High impact, moderate cost.)*
5. **Sliding-window overlap ablation** (0.25/0.5/0.75) measured against both Dice/HD95 *and* wall-clock latency — validates your own stated 1–2 min/case NFR isn't being silently violated. *(Moderate impact, low cost.)*
6. **Patient-level split leakage assertion** — a one-line automated check that `train_ids ∩ val_ids = ∅`. *(Low cost, closes a credibility gap cheaply — do this even though it's probably already fine.)*
7. **Held-out slice carve-out** from the current val set, reported separately from the val-selected number going forward. *(Moderate cost now, meaningfully strengthens any future paper or rigorous demo claim.)*
8. **Gaussian vs. uniform blending ablation** — cheap way to actually validate the seam-artifact claim rather than just asserting it. *(Low cost, moderate credibility value.)*
9. **Batch-size / VRAM ceiling check** at `base_channels=32`, 128³ patches — turns an unstated assumption into a logged, defensible number. *(Low cost, closes a 🟠 flag.)*
10. **Augmentation component ablation** (with/without geometric augmentation) tracking Dice *and* HD95 separately — validates the "elastic deformation excluded to protect HD95" reasoning rather than leaving it as an assertion. *(Moderate cost, do this after the run actually completes.)*

---

## Senior-Level Viva / Interview Questions

**Beginner**
- *What is image segmentation, and how is it different from classification?* — Classification assigns one label to a whole image; segmentation assigns a label to every voxel, producing a spatial mask, not a single category.
- *What does Dice coefficient measure?* — The overlap between predicted and ground-truth masks: 2×intersection / (sum of both areas), ranging 0 (no overlap) to 1 (perfect match).
- *Why 3D instead of 2D for this task?* — Tumors are 3D structures; a 2D slice-by-slice model reconstructs 3D context only approximately and can miss volumetric continuity a true 3D convolution captures directly.
- *What framework is the API built with?* — FastAPI, chosen for native async support and automatic OpenAPI docs.

**Intermediate**
- *Why InstanceNorm instead of BatchNorm here?* — 3D volumetric training at this patch size forces batch sizes of 1–4; BatchNorm's running statistics are unreliable that small, while InstanceNorm normalizes per-sample, independent of batch size.
- *Why does the loss combine Dice and Cross-Entropy instead of using one alone?* — CE gives a stable per-voxel gradient even early in training; Dice directly optimizes the overlap metric and handles the foreground/background imbalance that dominates BraTS volumes.
- *What's the purpose of sliding-window inference?* — The full volume doesn't fit in GPU memory at once; the volume is tiled into overlapping patches, each run through the model, and predictions are blended back together.
- *Why is Gaussian weighting used in the blending step?* — Patch edges have the least surrounding context and are where errors concentrate; Gaussian weighting downweights edges and trusts patch centers more, avoiding visible seam artifacts.

**Advanced**
- *How do you avoid a train/inference preprocessing mismatch when using nnU-Net's ideas without its framework?* — A single, versioned fingerprint config (target spacing, normalization parameters) is computed once and loaded identically by both the training and inference code paths — one source of truth, not two independently-maintained copies.
- *Why HD95 instead of raw Hausdorff distance?* — Raw Hausdorff is set by a single worst-case outlier voxel, making it unstable for model comparison; HD95 discards the most extreme 5% of distances while still penalizing genuine boundary disagreement.
- *What's the risk in using a fixed connected-component filtering threshold?* — It's applied uniformly across classes of very different typical size; a threshold tuned to remove noise in the largest class can just as easily delete real instances of the smallest, clinically important class.
- *Why is macro-averaging by case, not pooled voxels, the correct aggregation for Dice?* — Pooling voxels across cases lets a few large-tumor cases dominate the aggregate number; case-level macro-averaging treats every case equally, which better reflects per-patient performance.

**Expert**
- *Where does your custom pipeline actually diverge from what official nnU-Net would configure automatically?* — Patch size and channel width are hand-set to fit a single T4 rather than derived from nnU-Net's own memory-fingerprinting heuristic; the model is trained on a single train/val split rather than nnU-Net's default 5-fold ensemble.
- *How would you know if your model is undertrained versus capacity-limited?* — Compare the per-class Dice trend against published 3D U-Net baselines on the same split at a comparable epoch count; a model still improving when training stops is undertrained, one that's plateaued well below typical baselines despite full training suggests a capacity or preprocessing issue instead.
- *What would change your architecture choice if constraints changed?* — More GPU/time would make MedNeXt or a MONAI-SegResNet warm-start the natural next evaluation, not a full architecture rebuild; a multi-GPU budget would make the current 2025-pattern ensembling approach (nnU-Net+MedNeXt+Swin UNETR) worth pursuing directly.
- *Is your reported Dice number a fair, unbiased estimate of model quality?* — Not yet as currently structured — checkpoint selection and headline reporting share the same validation split, which optimistically biases the reported number by an unknown amount; a genuinely held-out slice or an official leaderboard submission would be needed for an unbiased claim.

---

## "Why Not X?" Interview Prep

**Q: Why didn't you use the official nnU-Net framework directly?**
- *30-sec:* "The course objective is implementing 3D segmentation myself, not wrapping a library — nnU-Net's own config-search overhead also doesn't fit a 3-month solo timeline."
- *2-min:* "nnU-Net would satisfy 'have a working segmenter' but not the actual learning goal of implementing the architecture, loss, and training loop directly. I adopted its methodology — fingerprinting, foreground normalization, compound loss, Gaussian sliding-window inference — without its codebase, which also keeps the whole pipeline in one repo I fully own and can debug, rather than depending on a large external framework's internals under a hard deadline."
- *Deep follow-up:* "Where does your pipeline actually diverge from what nnU-Net's automatic configuration would choose?" → Patch size/channel width set by hand for T4 memory fit, not derived from nnU-Net's own fingerprinting heuristic; single train/val split instead of 5-fold ensembling.
- *Counterargument:* "So you have no controlled comparison against the actual reference implementation's ceiling."
- *Best response:* "Correct, and that's a named limitation, not a hidden one — a MONAI-bundle nnU-Net run on the same split is the cheapest way to close that gap, and it's on my list rather than something I'm claiming already exists."

**Q: Why U-Net instead of a transformer-based architecture like Swin UNETR?**
- *30-sec:* "Transformer segmentation models generally need more training data and compute than a single T4 and a 3-month solo timeline comfortably support."
- *2-min:* "Swin UNETR and similar hybrids can outperform CNNs given enough data and compute, but they're also meaningfully harder to implement and debug correctly from scratch in the available time — and 'implement it myself' is the actual assignment. A 3D U-Net is the right complexity level for what I can fully own and validate in three months solo."
- *Deep follow-up:* "What would make you switch?" → More GPU-hours and either more time or a team — MedNeXt first (competitive with transformers at lower compute), transformer hybrids only if data/compute genuinely scaled up.
- *Counterargument:* "Isn't that just picking the easier option?"
- *Best response:* "It's picking the option whose failure modes I can actually diagnose and fix within the timeline — a transformer I can't fully debug in three months is a worse outcome than a CNN I can."

**Q: Why 3D instead of 2D or 2.5D?**
- *30-sec:* "Tumors are volumetric; 2D loses true cross-slice spatial context that matters for a boundary metric like HD95."
- *2-min:* "2D/2.5D is cheaper to train and batch, and it's a completely reasonable choice for some tasks — but BraTS's ground truth and evaluation convention (Dice/HD95 computed in 3D, in physical mm) are built around volumetric masks. A 2D model reconstructs 3D structure only approximately by stacking independent slice predictions, which tends to produce more boundary discontinuity between slices — exactly the failure mode HD95 is sensitive to."
- *Deep follow-up:* "Under what conditions would 2D actually be the better engineering call?" → If the task were triage/screening (presence/absence) rather than delineation, or if compute were far more constrained than a single T4.
- *Counterargument:* "2D would have trained faster and let you iterate more within three months."
- *Best response:* "True, and that's a real trade-off I accepted — but faster iteration on the wrong problem formulation isn't actually faster progress toward the stated deliverable."

**Q: Why InstanceNorm instead of GroupNorm?**
- *30-sec:* "Both are batch-independent; InstanceNorm is the more battle-tested default for this exact task, including in nnU-Net itself."
- *2-min:* "GroupNorm normalizes across channel groups rather than per-instance-per-channel, and is a legitimate middle ground between InstanceNorm and BatchNorm. I didn't run a head-to-head — InstanceNorm is what nnU-Net defaults to for 3D medical segmentation at small batch sizes, and matching that established convention was a reasonable default rather than something I needed to independently re-derive."
- *Deep follow-up:* "Have you actually ablated this?" → No — flag honestly: "The PRD does not specify this as validated; it's a convention-matched default, not an experimentally confirmed choice."
- *Counterargument:* "So it's an assumption, not a decision."
- *Best response:* "It's a well-evidenced convention from the closest reference methodology I'm already grounding the rest of the pipeline in — cheap and low-risk to leave as-is unless per-class results suggest otherwise."

**Q: Why Dice + CE instead of Focal Tversky or Generalized Dice?**
- *30-sec:* "Dice+CE is the field-standard, low-risk default for BraTS's imbalance; Tversky/GDL are the upgrade path if per-class results show a specific class is underperforming."
- *2-min:* "CE gives stable per-voxel gradients everywhere including easy background voxels; Dice directly optimizes overlap and handles imbalance reasonably well on its own. Tversky lets you explicitly trade precision vs. recall with an extra hyperparameter, and GDL auto-weights by inverse class volume — both are real options, but I don't yet have per-class evidence telling me Dice+CE is failing on a specific class, so reaching for a more complex loss now would be premature."
- *Deep follow-up:* "What per-class result would make you switch?" → Enhancing-tumor Dice meaningfully lower than edema/whole-tumor despite full training.
- *Counterargument:* "Isn't enhancing tumor known to be the hardest class in BraTS generally? Why not preempt it?"
- *Best response:* "Fair — it is generally the hardest class, and I'd consider adding Tversky as a documented next experiment rather than switching blind; I just don't want to change a well-validated default on an assumption I haven't measured yet on my own data."

**Q: Why HD95 instead of raw Hausdorff or ASSD?**
- *30-sec:* "Raw Hausdorff is dominated by a single outlier voxel and unstable for model comparison; HD95 is the BraTS-leaderboard standard for exactly that reason."
- *2-min:* "HD95 discards the top 5% most extreme surface distances while still penalizing genuine boundary disagreement, which makes it far more stable for comparing models than raw Hausdorff. ASSD (averaging over all surface points) is a legitimate complementary metric — less sensitive to any single outlier than even HD95 — and I'd consider reporting it alongside HD95 as a robustness check, not as a replacement, since HD95 is what makes my numbers directly comparable to published BraTS results."
- *Deep follow-up:* "What does HD95 not tell you that ASSD would?" → ASSD reflects average boundary agreement across the whole surface; HD95 specifically captures how bad the worst *non-outlier* disagreement is — different failure modes.
- *Counterargument:* "Doesn't discarding the top 5% just hide your worst failures?"
- *Best response:* "It discards true outliers, not systematic ones — if 5%+ of a mask's boundary is off, HD95 still reflects that; it's specifically the single-voxel-artifact case it protects against, which raw HD can't distinguish from a real large-scale failure."

**Q: Why FastAPI with polling instead of Celery+Redis or WebSockets?**
- *30-sec:* "Polling and an in-process job queue match a single-instance, FIFO-scale deployment; Celery+Redis is real infrastructure this scale doesn't need yet."
- *2-min:* "My own NFR explicitly states jobs are queued FIFO, not parallel-executed — an in-process background task queue already satisfies that. Celery+Redis would add a message broker and worker-process management for a scaling need I don't have yet at single-user demo scale. WebSockets/SSE would give instant push updates instead of polling, which is a genuine UX improvement, but adds connection-management complexity for marginal benefit given jobs already take 1–2 minutes — a few seconds of polling latency is imperceptible relative to that."
- *Deep follow-up:* "What's your actual trigger to add Celery?" → More than a handful of concurrent users, or a need for automatic job retry/backoff on failure.
- *Counterargument:* "Polling wastes requests compared to push."
- *Best response:* "At a 1–2 minute job duration and single-digit concurrent users, that waste is negligible — I'd rather not build push-notification infrastructure to solve a load problem I don't have."

**Q: Why SQLite instead of PostgreSQL from day one?**
- *30-sec:* "Zero-ops and sufficient for single-instance, no-concurrent-writer academic scale; Postgres buys nothing here except operational overhead."
- *2-min:* "The real risk with SQLite is its single-writer lock under concurrent write load — worth checking with WAL mode given the async job queue, but at single-user demo scale that's a non-issue in practice. Postgres becomes the right call the moment there's genuine concurrent multi-writer load or a need for DB-level access control once auth ships — that's a documented, named trigger, not a vague 'eventually.'"
- *Deep follow-up:* "Have you actually load-tested SQLite under your job queue's write pattern?" → Honest answer: "The PRD does not specify this as tested — it's a reasonable assumption given FIFO job execution, not a validated benchmark."
- *Counterargument:* "Isn't 'we'll migrate later' often a decision that never actually happens?"
- *Best response:* "That's a fair general concern, but the schema is already fully relational and UUID-keyed specifically so the migration is close to mechanical when the trigger condition actually arrives — it's not a 'someday' with no path, it's a documented, low-friction upgrade."

**Q: Why react-three-fiber instead of vtk.js?**
- *30-sec:* "vtk.js supports true volumetric ray-casting, which I don't need — every stated UX requirement (rotate, zoom, toggle sub-regions) is satisfied by a surface mesh."
- *2-min:* "vtk.js can render raw intensity gradients inside the volume, not just an extracted surface — genuinely more powerful, and purpose-built for medical imaging, but with a materially heavier learning curve and setup cost. react-three-fiber integrates natively with the React stack I'm already committed to, and marching-cubes surface extraction fully satisfies the actual UX spec I wrote: rotate/zoom/pan a tumor shape and toggle sub-regions on and off. I don't need to see inside the volume's raw intensity texture for that."
- *Deep follow-up:* "What would make volumetric ray-casting actually necessary?" → A requirement to inspect raw intensity texture inside the predicted boundary — e.g., partial-volume effects at an edge — which is a real vtk.js use case, correctly deferred as a V1/stretch upgrade in my own ADR-002.
- *Counterargument:* "Isn't vtk.js the more 'correct' medical-imaging tool regardless?"
- *Best response:* "It's the more capable tool for volumetric intensity inspection specifically — but using the heavier tool without a requirement that needs its extra capability would be over-engineering, not correctness."

**Q: Why sliding-window inference with Gaussian blending instead of test-time augmentation or ensembling?**
- *30-sec:* "Sliding-window is a memory necessity, not optional; TTA/ensembling both multiply inference cost roughly linearly, which conflicts with my own 1–2 min/case latency target."
- *2-min:* "The full volume doesn't fit T4 memory at once, so sliding-window is required regardless of what else I do. TTA (averaging predictions across flipped/rotated versions of the input) and full ensembling both improve accuracy at a real latency cost — multiple full forward passes instead of one — which conflicts with my stated NFR unless I have latency budget to spare. Gaussian blending specifically (versus uniform blending) is the cheap fix for patch-seam artifacts, since it downweights exactly the low-context patch edges where errors concentrate, at effectively zero extra inference cost."
- *Deep follow-up:* "Is 1–2 minutes a hard requirement or an aspiration?" → PRD states it as "targeted... a measured, reported number, not a guaranteed SLA" — so there's real room to trade some of that budget for TTA if it's shown to matter.
- *Counterargument:* "Couldn't you afford one round of TTA and still be well within 1–2 minutes?"
- *Best response:* "Possibly — that's actually a fair, cheap experiment I haven't run: measure a single TTA pass's Dice/HD95 gain against its latency cost and see if it fits inside the stated budget with room to spare."

---

## Final Architecture Defense

*A coherent explanation you could give an expert interviewer, problem → constraints → architecture → decisions → trade-offs → evaluation → deployment → limitations.*

"The problem is that tumor sub-region delineation in multi-modal brain MRI is slow, manual, and inter-observer variable — and the goal of this project isn't to replace that process, it's to build a transparent, reproducible research pipeline that shows a radiologist or researcher a candidate segmentation they can inspect, not just trust.

My constraints were tight and specific: a single Tesla T4 with 14.56GB of VRAM, three months, effectively solo, and a course objective centered on implementing the architecture myself rather than importing a solved one. Those constraints ruled out several strong-looking options before I even got to comparing accuracy — official nnU-Net's own config-search overhead doesn't fit a solo 3-month window; transformer-based segmenters like Swin UNETR generally need more data and compute than I have and are meaningfully harder to debug from scratch under deadline; full multi-model ensembling — which is genuinely the current winning pattern on the 2025 BraTS challenges — assumes multi-GPU, multi-day training budgets I don't have.

So I built a custom 3D U-Net, conceptually grounded in nnU-Net's methodology — dataset fingerprinting, foreground-only z-score normalization, patch-based training with foreground oversampling, compound Dice+CE loss, sliding-window inference with Gaussian-weighted stitching, connected-component post-processing — without depending on the nnU-Net codebase itself. That's a deliberate trade: I give up nnU-Net's automatic architecture search and default 5-fold ensembling, in exchange for a pipeline I fully own, can debug end-to-end, and can actually finish inside my timeline. Concretely, at base_channels=32 with 128³ patches, that's roughly an 18.8-million-parameter model, using InstanceNorm because true 3D volumetric training at this patch size forces batch sizes of one to four, where BatchNorm's running statistics stop being reliable.

On evaluation, I use Dice and HD95, computed per case and macro-averaged, matching the BraTS leaderboard convention — including the documented edge-case handling for empty masks and absent classes, which is where a lot of re-implementations quietly get it wrong. HD95 specifically over raw Hausdorff distance because raw HD is set by a single worst-case outlier voxel and is unstable for comparing models; HD95 discards the top five percent most extreme distances while still penalizing genuine boundary disagreement.

On deployment, the whole thing is wrapped in a FastAPI service with an async job queue — inference takes one to two minutes, so it returns a job ID immediately and the frontend polls for results, rather than risking a synchronous request timing out. Full volumetric review happens through a React frontend: a 2D slice viewer for detail, and a react-three-fiber 3D mesh view, extracted via marching cubes, for tumor shape — sufficient for every UX requirement I actually specified, at much lower implementation cost than a purpose-built medical volume renderer like vtk.js, which I don't need unless a future requirement asks someone to inspect raw intensity texture inside the tumor rather than just its predicted shape.

I'll be direct about where this stands right now, not where the spec says it should stand. My most recent training run was interrupted at epoch five of a planned thirty, on free-tier Colab GPU time, with a proxy validation Dice around 0.25 — that is not a finished result, and I don't present it as one. The documentation's example API response shows a headline Dice of 0.81, which is an illustrative target grounded in typical published BraTS baseline performance, not something I've measured yet. I also don't currently have a validation split that's held out from checkpoint selection, so even once training completes, my first reported number will be honestly labeled as val-selected rather than an unbiased held-out result. And my optimizer choice — SGD with Nesterov momentum versus AdamW — is still open; given that my actual constraint right now is interrupted training sessions on a free GPU tier, I'd lean AdamW first, because it needs fewer well-tuned epochs to produce a usable signal, and revisit SGD with a full polynomial decay schedule once I have uninterrupted compute.

None of that undermines the architecture — it's exactly the honest gap between a specification and a result that any real engineering project has at this stage. What I'd want an interviewer to take away is that every major decision here traces back to a named constraint, not a preference, and that I can point to the specific place my pipeline diverges from the reference methodology I'm citing, rather than claiming an equivalence I haven't earned."
