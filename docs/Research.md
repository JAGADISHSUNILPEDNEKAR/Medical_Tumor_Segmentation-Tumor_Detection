# Brain Tumor AI: Systematic Literature Review & Research Positioning
### Prepared against the "Medical Image Segmentation & Tumor Detection" PRD (BraTS 3D U-Net project)

---

**A note on scope before you read this.** The brief this was built against asks for a multi-database systematic review spanning 28 output sections — the kind of undertaking a review-paper team spends weeks on across PubMed, IEEE Xplore, ScienceDirect, SpringerLink, and Google Scholar with manual snowballing. What follows is a single-pass, evidence-grounded synthesis built from ~30 targeted searches across arXiv, PubMed/PMC, Nature, ScienceDirect, Springer, and MICCAI/BraTS challenge proceedings, cross-checked against your existing PRD's own §18 addendum. Every number below traces to a real, named source (listed in §29). Where I couldn't verify a specific figure, I've said so rather than filled the gap — per your own instructions not to invent numbers. Treat this as a strong first draft of the "related work" and "gap analysis" sections of a paper, not a finished submission.

---

## 1. Executive Summary

Your PRD's core engine — a custom 3D U-Net, conceptually grounded in nnU-Net, trained on BraTS, evaluated with Dice/HD95 — sits in one of the most mature, most heavily benchmarked corners of medical AI. nnU-Net has been the segmentation baseline to beat since it won BraTS 2020 and the Medical Segmentation Decathlon [1], and every top BraTS submission since 2021 is either nnU-Net, a transformer hybrid of it (Swin UNETR, MedNeXt), or an ensemble of several such models [2,3,4,5,6]. That means the segmentation core, as scoped, is **solid engineering and a legitimate learning exercise, but not itself a research contribution** — the field has already answered "can a 3D U-Net segment BraTS gliomas well" with a resounding yes (pooled Dice around 84% across published detection/segmentation studies with external validation [33]).

Two adjacent areas your PRD's MVP correctly avoids turned out, on investigation, to be genuinely unreliable rather than just "hard": **MGMT methylation prediction from MRI alone is close to chance level once tested outside the training institution** (roughly 80% of 420 systematically-trained models showed no significant difference from 50% accuracy on external validation, and the BraTS 2021 radiogenomics challenge's own first-place model dropped to 54.8% external accuracy) [27]. **IDH mutation prediction fares better but still degrades substantially under external testing** — pooled meta-analytic sensitivity/specificity of 80%/85% for IDH, but individual studies with AUROCs above 0.90 internally often fall to 0.73–0.86 externally [28,54]. Your Stretch-tier framing of these as "genuine open research attempts, not promised features" is the empirically correct posture.

The most defensible novel angle, and the one your PRD's own §18 addendum already converges on, is **not** a new segmentation architecture or another CNN-backbone comparison — both are saturated (2D classification backbone comparisons alone number in the dozens for 2024–2026 [13,14,15,16,17,20,21,22]). It is a **narrow systems contribution**: grounding an AI-drafted radiology-style report directly in your own pipeline's quantitative Dice/HD95/volume numbers and a retrieved similar prior case, with a radiologist sign-off gate — a pattern well-established for chest X-ray (retrieval-augmented report generation, or "RAG for radiology") [38,39,40] but, as far as this search could establish, not yet applied to BraTS-style volumetric tumor sub-region segmentation with numeric grounding. That gap is real, buildable from pieces already in your V1 tier, and honestly scoped for a 1–2 person, 3-month project.

---

## 2. My PRD — Academic Problem Definition

**Restated in academic terms:** the PRD specifies a supervised, fully-convolutional, volumetric semantic segmentation problem — multi-class voxel-wise classification of four co-registered, skull-stripped structural MRI sequences (T1, T1ce, T2, FLAIR) into background plus three glioma sub-region classes (necrotic/non-enhancing core, peritumoral edema, enhancing tumor), trained and evaluated on the BraTS corpus, with a compound Dice + cross-entropy objective, sliding-window inference with Gaussian-weighted patch blending, and Dice/HD95 as the primary evaluation metrics, computed per class and macro-averaged per case. This is wrapped in a decision-support (not diagnostic) application with 2D/3D visualization and an async job API.

**Ambiguities and technically unrealistic assumptions worth flagging:**

- **BraTS release/version is never pinned.** This matters more than it looks. The three-class label taxonomy in your DB schema (`necrotic`, `edema`, `enhancing`) matches the classic BraTS ≤2020/2021/2023 pre-treatment adult glioma scheme. BraTS 2024's post-treatment glioma task uses a *different*, four-region taxonomy — enhancing tissue (ET), surrounding non-enhancing FLAIR hyperintensity (SNFH), non-enhancing tumor core (NETC), and resection cavity (RC) [7,8]. If your research addendum's citations to BraTS 2024/2025 ensembling literature are meant to inform your own training data choice, you'll need to decide explicitly which release you're training/evaluating on, because the label schema is not interchangeable.
- **The "~1–2 minutes per case" latency target has no hardware pinned to it.** Sliding-window inference over a full 240×240×155 volume with 128³ patches and 50% overlap is a real, measurable compute cost that varies by an order of magnitude between, say, an RTX 4090 and a shared cloud T4. As written, the NFR is a target to measure and report, not a guarantee — which the TRD itself says, but it's worth being explicit that "under budget" claims in any eventual paper need the GPU model stated every time the number appears.
- **The case-retrieval-augmented report-drafting idea (your own §18.9's proposed positioning) has no corresponding LLD yet.** The current API spec (`/predict`, `/evaluate`, `/results/{id}`) has no embedding step, no similarity index, and no drafting endpoint. If this is the direction you take, it needs its own low-level design before it's a system, not just a research idea.
- **The checkpoint registry doesn't track split provenance at the patient level.** Given how central patient-level (vs. slice-level) splitting is to whether *any* accuracy number in this space is trustworthy (§19 below), your `checkpoints` table should probably record which patient IDs were in train vs. validation vs. test, not just a split name string, so that a reviewer (or future-you) can audit for leakage directly from the DB.

---

## 3. Evolution of Brain Tumor AI Research

The field's trajectory, as reconstructed from the literature searched:

**Pre-2012 — traditional image processing.** Thresholding, region growing, watershed, and atlas-based methods on single-sequence MRI; high false-positive rates, no learned representation.

**2012–2017 — classical ML and the birth of BraTS.** The BraTS challenge began in 2012 with 50 cases [173] and grew to 253 cases by 2015 [173], run under Menze et al.'s original benchmark paper [164]. Classical ML (SVM, Random Forest) on handcrafted radiomic/texture features (GLCM, wavelets, Gabor) dominated. Cheng et al.'s 2017 Figshare dataset (3,064 T1-weighted contrast-enhanced slices from 233 patients, three tumor types) became a second major benchmark, still in heavy use in 2025–2026 papers [85–94].

**2015–2019 — 2D/3D CNNs and ImageNet transfer learning.** BraTS grew to 477–660 cases (2017–2020 editions) [173]. 2D CNNs (AlexNet, VGG16/19, ResNet, GoogLeNet) fine-tuned from ImageNet weights became the default for slice-level tumor-type classification, alongside early 3D CNN segmentation work.

**2020–2021 — self-configuring 3D U-Nets take over.** Isensee et al.'s nnU-Net (Nature Methods, 2021) won the Medical Segmentation Decathlon and the 2020 BraTS challenge by automating preprocessing, architecture, and post-processing decisions rather than hand-designing them [1,17]. BraTS 2021 scaled to 2,040 patients (1,251 training cases) with an added MGMT radiogenomic classification task [165,168].

**2021–2022 — attention and the first transformers.** Attention U-Net, SE/CBAM blocks, and U-Net++ layered attention onto the U-Net backbone with modest, inconsistent gains (§ "attention mechanisms" below). Hatamizadeh et al.'s Swin UNETR (2022) was the first transformer-based architecture to place competitively in a BraTS challenge, edging out nnU-Net by roughly 0.5% average Dice on BraTS 2021 [25,26,31].

**2023–2025 — hybrids and ensembles win, transformers alone don't clearly beat CNNs.** Winning BraTS-family submissions increasingly ensemble nnU-Net, Swin UNETR, and the newer ConvNeXt-inspired MedNeXt [3] rather than picking one architecture — e.g., BraTS-PEDs 2023's top methods were nnU-Net/Swin UNETR ensembles and self-supervised nnU-Net extensions [4]; BraTS-SSA/Africa 2025 was won with segmentation-aware augmentation plus model ensembling [6]; BraTS-PED 2025 used a frequency-aware ensemble of nnU-Net, fine-tuned Swin UNETR, and HFF-Net [5].

**2023–2026 — multimodal, genomic, and language-model integration.** Radiogenomics (MRI → IDH/MGMT status) matured into a well-studied but externally fragile subfield [27,28,53]. Vision-language models for radiology report generation moved from chest X-ray (MIMIC-CXR-scale datasets) into brain MRI specifically with AutoRG-Brain (2024) and a LLaVA-Med 1.5 glioblastoma fine-tune, alongside retrieval-augmented ("RAG") drafting pipelines that ground generated text in retrieved similar cases — again, so far, concentrated on chest X-ray rather than volumetric tumor segmentation [38,39,60–69].

---

## 4. Complete Taxonomy of Detection & Classification Techniques

| Category | What it covers | Where it stands in 2026 |
|---|---|---|
| **A. Traditional image processing** | Thresholding, region growing, watershed, morphological ops, GLCM/wavelet/Gabor texture, handcrafted radiomics | Superseded for primary segmentation; radiomic *features* (via PyRadiomics) remain useful as a post-hoc, interpretable layer on top of a DL-predicted mask [PRD §18.5]. |
| **B. Classical ML** | SVM, RF, KNN, Naive Bayes, logistic regression, XGBoost on handcrafted or CNN-embedded features | Still common as the final classifier in "DL feature extraction + classical ML head" hybrids, and as an honest, low-variance baseline for small-data tasks like IDH prediction where deep nets overfit [54]. |
| **C. Deep CNNs (2D)** | AlexNet → VGG16/19 → ResNet50/101 → DenseNet → Inception/Xception → MobileNet/EfficientNet → ConvNeXt | The dominant family for **slice-level tumor-type classification** (glioma/meningioma/pituitary/no-tumor), not for volumetric segmentation. See §5/§9 for benchmark numbers — accuracies cluster in the mid-90s to high-90s%, with no architecture winning consistently across independent studies. |
| **D. Transfer learning** | ImageNet-pretrained backbones fine-tuned on brain MRI | Near-universal for 2D classification; MedicalNet/Med3D offers an analogous pretrained-3D-encoder route for volumetric tasks [PRD §18.2]. |
| **E. 3D deep learning** | 3D CNN, 3D ResNet, 3D U-Net, V-Net, 3D DenseNet | Necessary for segmentation tasks that need inter-slice context; more memory- and compute-hungry, and the family your PRD correctly commits to. |
| **F. Segmentation** | U-Net, 3D U-Net, Attention U-Net, U-Net++, nnU-Net, V-Net, DeepLab, transformer/hybrid segmentation | nnU-Net-family and its ConvNeXt/transformer descendants (MedNeXt, Swin UNETR) are current SOTA (§6, §10). |
| **G. Attention mechanisms** | Spatial/channel attention, SE, CBAM, self-attention, Attention U-Net | Genuinely help, but modestly and inconsistently — see the dedicated discussion below. |
| **H. Transformer-based** | ViT, Swin, TransUNet, UNETR, Swin-UNETR, ViT-CNN hybrids | Competitive with, not decisively superior to, well-tuned CNNs on BraTS-scale data; need more data/compute and are prone to overfitting on small cohorts [148–155]. |
| **I. Hybrid/ensemble** | CNN+Transformer, multi-model ensembles, feature/decision-level fusion | The actual winning pattern on recent BraTS-family leaderboards (§13) — but mostly at multi-GPU training budgets your PRD's single-GPU NFR doesn't have. |
| **J. Lightweight/edge** | MobileNet, EfficientNet, distillation, pruning, quantization | An active, separate research thread (parameter counts down to ~0.5M with 96–99% reported accuracy on public classification sets) [103–111], relevant to your project only if a screening/triage head is ever added on top of the segmentation core. |

---

## 5. Tumor Classification Methods, by Level

| Level | Task | Data typically used | Representative algorithms | Reported accuracy range | What remains genuinely unsolved |
|---|---|---|---|---|---|
| **1 — Detection** | Tumor vs. normal | Single or multi-sequence MRI, 2D slices | CNN classifiers, often binary VGG16/ResNet | High 90s% on public sets [37] | Mostly solved on curated public data; real-world screening performance under distribution shift is under-studied. |
| **2 — Type classification** | Glioma / meningioma / pituitary / no-tumor | 2D T1-CE slices (Figshare, Kaggle combined set) | VGG16, ResNet50, EfficientNet, Xception, ensembles | Roughly 90–99.9% depending on split rigor [13–22] | Reported numbers are not comparable across papers because dataset composition, split method, and even test-set overlap vary wildly (§19). |
| **3 — Glioma grading (LGG/HGG)** | Low- vs. high-grade glioma | Structural MRI ± DTI | CNN/DCNN, transfer learning, ensembles | Meta-analytic pooled sensitivity 94%, specificity 93%, AUC 0.98 across 33 studies (Sun et al., 2023) [32,158,159]; individual studies report up to ~99% on BraTS-derived sets [162] | Meta-analysis itself flags "great heterogeneity" — pooled numbers mask wide study-to-study variance and near-total absence of prospective validation. |
| **4 — Molecular/subtype (IDH, 1p/19q, MGMT)** | Non-invasive molecular status prediction | mpMRI ± radiomics ± clinical | Radiomics + ML, 3D CNN, hybrid DLRN | IDH: meta-analytic 80%/85% sens/spec [53]; individual AUCs 0.80–0.99 internally [50] but degrading to 0.73–0.86 externally [51,54]. MGMT: ~80% of systematically-trained models statistically indistinguishable from chance on external validation [27] | This is the field's clearest reliability gap — see §16, §19. |
| **5 — Prognosis/outcome** | Overall survival, recurrence, treatment response | Imaging + age + resection status | Radiomics + regression, 3D CNN + radiomics | For BraTS's official GTR-only survival task, a simple linear-regression-on-age baseline has repeatedly outperformed or matched imaging-based approaches [80,82] | Imaging features have not been shown to add reliable prognostic value beyond simple clinical variables in the GTR subgroup; this is a well-documented negative result, not an oversight. |

---

## 6. Segmentation Methods

**Architecture family and what it buys you:**

- **U-Net / 3D U-Net** — the baseline encoder-decoder-with-skip-connections design; still competitive when properly tuned, as nnU-Net demonstrates.
- **nnU-Net** — not a new architecture but a self-configuring *pipeline* (fingerprinting → resampling/normalization rules → 2D/3D/cascade config search → 5-fold training → ensemble/config selection) built on top of standard U-Net blocks [1,19,21]. It "surpasses most existing approaches, including highly specialized solutions" across 23 public biomedical segmentation benchmarks per its own validation [1], and a 2024 update (nnU-Net ResEnc XL) reports a further ~0.93% average Dice gain over the original nnU-Net across multiple datasets by swapping in ResNet/ConvNeXt-style encoders [13].
- **Attention U-Net / U-Net++** — add gated or nested skip connections; gains are real but small in most ablations (§ attention discussion).
- **MedNeXt** — a ConvNeXt-block-based, compound-scalable 3D ConvNet designed specifically for data-scarce medical segmentation; its authors report it retains the trainability advantages of convolutional inductive bias while approaching transformer-level representational capacity, with state-of-the-art results across four segmentation tasks spanning CT and MRI [3,122–125]. It is increasingly the "modern CNN" ingredient in 2024–2025 BraTS-family winning ensembles [5,121,126].
- **UNETR / Swin UNETR** — transformer (ViT/Swin) encoders paired with a CNN decoder via skip connections. Swin UNETR was reported as one of the first transformer-based models to place competitively in a BraTS challenge, outperforming the closest competing approaches by roughly 0.4–0.7% average Dice across tumor sub-regions on BraTS 2021 [25,26,29,31].
- **DeepLab / hybrid CNN-Transformer** (TransUNet, TransBTS, Swin-UNet3D) — generally reported in the 78–91% Dice range on earlier BraTS editions (2019/2020), below nnU-Net and Swin UNETR on the same benchmarks [23,28].

**Detection → localization → segmentation → classification, as a pipeline:** detection answers "is there a tumor," localization gives a bounding region, segmentation delineates voxel-exact sub-region boundaries, and classification assigns a categorical label (tumor type, grade, or molecular status) either from the segmented region's features or end-to-end. Your PRD's pipeline sits entirely in the segmentation stage; the "classification" stretch-goal ideas (WHO grade, MGMT) would sit downstream of it, consuming the segmentation output as an ROI.

---

## 7. MRI Modalities

Your PRD correctly restricts to the four BraTS-standard structural sequences. What the literature says each contributes, and what's known about combining them:

| Sequence | What it shows | Evidence on standalone vs. combined use |
|---|---|---|
| **T1** | Baseline anatomy, poor tumor contrast | Rarely sufficient alone for tumor delineation. |
| **T1c (post-contrast)** | Enhancing tumor / blood-brain-barrier breakdown | Best single sequence for enhancing-tumor sub-region; central to MGMT/IDH radiomics pipelines [55,57]. |
| **T2** | Edema and tumor bulk | Frequently the strongest single sequence for IDH prediction in ablation studies — e.g., "T2w in the coronal plane consistently produced the most competitive models" in a large systematic MGMT evaluation [100]. |
| **FLAIR** | Edema/infiltration, suppresses CSF signal | Standard for whole-tumor boundary; combined with T2 for edema delineation. |
| **DWI/ADC, SWI, perfusion MRI** | Cellularity, hemorrhage, vascularity | Not part of BraTS; used in radiogenomics/grading studies outside the segmentation-challenge literature, generally as an additive signal rather than a replacement. |

Multi-sequence fusion measurably helps: a large multicenter IDH/grading study found the T1+T1c+FLAIR combination gave the best external-validation AUCs (grading 0.80, IDH 0.77) versus weaker single-sequence performance, while also showing that models degrade only moderately — not catastrophically — when one sequence is missing at inference time, provided the training pipeline explicitly models for incomplete input [51]. This is directly relevant to your BR-001 "reject cases missing any modality" design decision: the literature suggests a *more* clinically useful (if harder to build) version of that requirement would degrade gracefully rather than hard-reject, though hard-rejecting is the right, honest MVP choice given your 3-month scope.

---

## 8. Datasets

| Dataset | Modality | Tumor types | Size (approx.) | Task(s) | Access | Key limitation |
|---|---|---|---|---|---|---|
| **BraTS 2012–2015** | T1/T1c/T2/FLAIR | Glioma (HGG/LGG) | 50 → 253 cases [173] | Segmentation | Public, license varies by year | Small by later standards; early annotation protocols less standardized. |
| **BraTS 2017–2020** | Same | Glioma | 477–660 cases [173] | Segmentation, OS prediction (from 2017) | Public | OS task's GTR subset (~211 patients, 77 test) is small enough that an age-only linear baseline is a genuinely hard-to-beat comparator [80,82]. |
| **BraTS 2021** | Same | Glioma | 2,040 patients, 1,251 training cases [165,168] | Segmentation + MGMT radiogenomic classification | Public (Synapse), RSNA-ASNR-MICCAI benchmark | MGMT task proved largely unreliable under external validation (§16). |
| **BraTS 2023–2024** | Same | Adult glioma (pre- and post-treatment), meningioma, pediatric, metastases, Africa, sub-tracks | ~2,040 (GLI) + task-specific pediatric/meningioma/metastasis cohorts | Segmentation across many sub-tracks; post-treatment adds a 4-region label scheme (ET/SNFH/NETC/RC) [7,8] | Public | Label taxonomy differs by track/year — a real integration hazard noted in §2. |
| **BraTS-Africa** | Same | Adult glioma, Sub-Saharan Africa | Built from 660 BraTS 2020 cases + 114 ISLES 2015 cases as an initial base, expanded with African-institution data [164,167] | Segmentation, generalizability | Public | Explicitly designed to test whether Western-institution-trained models generalize to under-represented scanner/population distributions — most published models have never been tested this way. |
| **Figshare / Cheng et al. 2017** | T1-weighted contrast-enhanced | Meningioma, glioma, pituitary | 3,064 slices, 233 patients [85–94] | 2D classification | Public (CC BY 4.0) | Slice-level, not volumetric; heavy reuse across a decade of classification papers means "novel" results on this set are hard to distinguish from re-derivations. |
| **Kaggle "Brain Tumor MRI Dataset"** | T1-CE, mixed sources | Glioma, meningioma, pituitary, no-tumor | 7,023 images [91,92] | 2D 4-class classification | Public | A composite of Figshare + SARTAJ + Br35H with documented label-quality fixes in the glioma class — provenance and de-duplication across the merged sources is not independently audited in most papers that use it. |
| **TCGA-GBM / TCGA-LGG (via TCIA)** | Multi-sequence MRI + genomics | Glioblastoma, low-grade glioma | Institution-dependent | Radiogenomics, molecular subtype | Public, requires TCIA access | Genomic and imaging data linkage quality varies; used as the external-validation set in several IDH studies [58]. |
| **REMBRANDT** | Multi-sequence MRI + genomics | Glioma | ~130 patients | Radiogenomics | Public (TCIA) | Older, smaller, less consistently used in recent (2024–2026) literature than TCGA/TCIA-GBM. |
| **UCSD-PTGBM** | Includes ADC, ASL, DSC, RSI, SWAN | Glioblastoma | Recent (2025) release | External validation for MGMT | Public | Used as an external cohort precisely because it is *not* BraTS-derived, addressing domain-shift concerns [102]. |

**On data leakage — this is the single most important caveat for any benchmark table in this space.** Multiple independent studies quantify what happens when slices from the same patient's volume end up split across train and test sets instead of being kept patient-exclusive: one study found slice-level splitting inflated test accuracy by 29–55 percentage points across four independent neuroimaging cohorts, with an experiment on randomly-labeled data producing ~96% "accuracy" under slice-level splitting versus the correct ~50% under subject-level splitting [42,43,44,46]. A related Parkinson's-MRI study found slice-level and longitudinal leakage inflated accuracy by over 67% and 30% respectively [48]. This is not specific to Parkinson's/Alzheimer's work — it is a generic property of any pipeline that treats 2D slices as i.i.d. samples, and it applies directly to the Figshare/Kaggle-style 2D brain-tumor classification literature reviewed in §5/§9, where patient-level splitting is inconsistently reported or enforced [45,49].

---

## 9. Major Algorithms and Architectures — Selected Benchmark Detail

**2D classification backbones (glioma/meningioma/pituitary/no-tumor, mostly Figshare- or Kaggle-derived):**

- VGG16, ResNet50, ResNet101, MobileNetV2, DenseNet201, EfficientNetB3, InceptionV3, and Xception have all been reported in the 90–99.9% accuracy range depending on study, with **no architecture winning consistently** — VGG16 topped one 2025 comparison at 99.87–99.98% average cross-validated accuracy with EfficientNetB3 [33], Xception topped another with an F1 of 0.9817 [34], and a third found a custom lightweight CNN (0.57M parameters) beating all six tested pretrained backbones at 99.54% [40]. Ensembling VGG16+ResNet50+a custom CNN has been reported at 99.69% [35].
- A recurring, honest theme across several of these papers is that reported gains from ensembling or architecture choice are often within noise of the underlying dataset's ceiling, and that a well-regularized custom CNN with far fewer parameters matches or beats deep pretrained backbones on these specific benchmark sets [37,40].

**Volumetric segmentation (BraTS):**

| Challenge / dataset | Model | DSC (%) | HD95 (mm) | Source |
|---|---|---|---|---|
| BraTS 2019 | TransUnet | 78.17 | 6.92 | [23] |
| BraTS 2019 | Swin-Unet | 78.49 | 4.83 | [23] |
| BraTS 2019 | ResUnet+ | 88.30 | — | [23] |
| BraTS 2020 | nnU-Net | 91.18 | 8.49 | [23] |
| BraTS 2020 | ResUnet+ | 92.80 | — | [23] |
| BraTS 2021 | UNETR | 91.11 | — | [23] |
| BraTS 2021 | Swin UNETR | 92.61 | 5.30 | [23,25,26] |
| BraTS 2021 | SegResNet | 92.65 | 3.60 | [23] |
| BraTS 2021 | Coupling nnU-Net | 92.83 | 3.76 | [23] |
| BraTS 2023 (5-model comparison) | Swin UNETR | 89.65 avg | — | [10] |
| BraTS 2023 | UNETR | 85.40 avg | — | [10] |
| BraTS 2023 | Residual U-Net | 73.41 avg | — | [10] |
| BraTS 2023 | 3D U-Net (plain) | 68.90 avg | — | [10] |
| BraTS-PED 2025 (winning ensemble) | nnU-Net + fine-tuned Swin UNETR + HFF-Net | 72.3 (ET) / 95.6 (NET) / 89.5 (ED) / 92.3 (TC) / 92.3 (WT) | — | [5] |
| FeTS 2024 (federated) | PID-controller aggregation | 73.3 (ET) / 76.1 (TC) / 75.1 (WT) | 33.9 / 33.6 / 32.3 | [7] |

Two things stand out from this table. First, the spread within a *single* challenge year (e.g., 68.9% to 92.6% average Dice on architectures all evaluated on BraTS 2021/2023-adjacent data) is far larger than the spread *between* the best CNN and the best transformer — architecture choice within the modern-U-Net family matters less than whether the model is well-tuned at all. Second, federated/cross-institution settings (FeTS 2024) produce meaningfully lower Dice than single-institution BraTS training runs, which is a direct, quantified illustration of the generalization gap discussed in §19.

---

## 10. 2D vs. 3D

3D approaches preserve inter-slice anatomical context that 2D slice-wise methods discard by construction, at the cost of substantially higher GPU memory and training time (your own TRD's InstanceNorm-over-BatchNorm choice is a direct consequence of the small batch sizes — 1–4 patches — that 3D volumetric training forces). Empirically, a 2024 meta-analysis of 19 brain-tumor detection/segmentation studies with external validation found **3D models outperformed both 2D and ensemble counterparts on detection**, with the caveat that this held specifically for studies meeting their inclusion bar (external validation, clear metrics) [116–118]. This is a meaningful data point in favor of your PRD's 3D-first architecture choice, though it should be cited carefully — it is one meta-analysis's finding on a specific inclusion-criteria subset, not a field-wide consensus.

---

## 11. CNN vs. Transformer vs. Hybrid

The honest summary, synthesized across §6/§9's numbers: **pure transformer segmentation architectures (UNETR) generally underperform well-tuned CNNs (nnU-Net, SegResNet) on BraTS-scale data**, and where a transformer hybrid (Swin UNETR) does edge out nnU-Net, the margin is small — roughly half a percentage point of average Dice on BraTS 2021 [25,26]. This matches the broader medical-imaging literature's consensus that vision transformers lack the convolutional inductive bias needed to train well on the comparatively small datasets typical of medical imaging, and are prone to overfitting or unstable cross-domain transfer without large-scale pretraining or hybrid CNN components [148–155]. MedNeXt was explicitly designed to capture transformer-style large-receptive-field benefits while keeping a convolutional (more trainable-on-small-data) backbone, and its authors report it beating both plain CNNs and transformer baselines on data-limited medical segmentation tasks [3,122]. The practical takeaway for a single-GPU, 3-month project: **a well-tuned modern CNN (nnU-Net or MedNeXt-S) is a more defensible primary architecture than a transformer**, with transformer/hybrid variants best treated as an ablation rather than the headline model.

---

## 12. Ensemble Methods

Ensembling nnU-Net, MedNeXt, and Swin UNETR variants (often with GAN-based augmentation or topology-refinement steps) is, as your own PRD addendum already notes, the literal winning pattern across the 2025 BraTS-family sub-challenges [PRD §18.3]. The evidence gathered here reinforces that: BraTS-PEDs 2023's top three approaches were all ensembles or ensemble-adjacent (nnU-Net+Swin UNETR, Auto3DSeg/SegResNet, self-supervised nnU-Net) [4]; BraTS-PED 2025's winner combined three architectures with frequency-domain decomposition [5]; BraTS-SSA 2025 was won with segmentation-aware augmentation plus ensembling [6]. The FeTS 2024 federated-learning challenge is a useful counterpoint — even the best cross-institution aggregation method there produced meaningfully lower Dice (73–76%) than single-institution ensembles achieve on BraTS proper, underscoring that **ensembling improves benchmark scores more reliably than it improves genuine cross-institution generalization** [7]. Whether combining models improves *generalization* specifically (as opposed to leaderboard rank) is not well-isolated in the papers reviewed — most ensemble papers report only same-distribution validation/test splits, not held-out-institution results.

---

## 13. Explainable AI

Grad-CAM, SHAP, and LIME are the dominant XAI tools in this literature, near-universally used to produce heatmaps confirming a classifier attended to the tumor region rather than background [70–77,85,105]. Formal validation against radiologists is rare but not absent: one systematic review found a study where CAM-generated heatmaps were compared against manual tumor-region coloring by 10 doctors across 120 MRI images, with overlap never dropping below 98% (100% in most cases), followed by a structured trust survey [72]. That is a genuinely useful data point — but it is also close to the *only* such formally-validated example surfaced across this search, out of dozens of papers that generate Grad-CAM heatmaps and simply assert they "enhance clinical trust" without a reader study. The gap between "we produced a heatmap" and "we validated that heatmap against expert judgment" is large and mostly unfilled in this literature.

---

## 14. Multimodal MRI + Clinical Data

The clearest, best-evidenced finding in this entire review is a **negative result**: for BraTS's official overall-survival task (glioblastoma patients with gross total resection), a simple linear regression on patient age alone has repeatedly matched or beaten radiomics- and imaging-based approaches, placing 3rd out of 26 participants in the BraTS 2018 survival challenge using nothing but age [80,82]. A systematic evaluation explicitly testing whether adding radiomic features to age improves accuracy for the GTR subgroup found **no consistent improvement** [80]. More recent end-to-end deep pipelines combining segmentation + contrastive-learning-based survival classification do exist and report improved ranking (2nd/5th place out of 60+ teams in one BraTS survival+segmentation submission [84]), but the age-only baseline's persistence across nearly a decade of subsequent challenge cycles is the honest headline finding here, not the incremental wins.

---

## 15. MRI + Genomics / Radiogenomics

This is the area where the gap between internal and external validation performance is most dramatic and best-documented:

- **IDH mutation status:** a systematic review/meta-analysis of deep-learning radiomics pooled sensitivity/specificity at 80%/85% for IDH and 75%/82% for 1p/19q co-deletion, explicitly flagging segmentation method and degree of DL integration as significant sources of inter-study variability, and calling for multi-center harmonization and prospective validation before clinical translation [53]. Individual studies show internal AUCs of 0.80–0.99 [50] dropping to 0.73–0.86 on true external test sets [51,54,58], with one externally-validated model reporting a respectable 0.835 AUC / 85.1% accuracy on an independent cohort [59] — so IDH prediction is degraded but not destroyed by external validation, unlike MGMT.
- **MGMT promoter methylation:** the field's most sobering finding. A dedicated external-validation study trained 420 models across combinations of datasets, CNN architectures, MRI sequences, and random seeds, finding that **80.2% of models showed no statistically significant difference from 50% chance-level accuracy**, and that the BraTS 2021 radiogenomics challenge's own first-place solution — which presumably won on internal validation — achieved only 56.2% AUROC and 54.8% accuracy on a genuinely external test set [27,95–101]. The paper's own conclusion is unambiguous: MGMT methylation status may not be reliably predictable from preoperative MRI alone, even with deep learning, at current data scales. This directly validates your PRD's own framing of MGMT prediction as a Stretch-tier "genuine open research attempt, not a promised feature."

The field-level implication for radiogenomics research generally: any paper claiming a novel MGMT or IDH predictor **must** report external, ideally multi-institution, validation to be credible — internal cross-validation numbers in this specific sub-area have a documented history of not replicating.

---

## 16. AI-Based Report Generation

Vision-language and multimodal-LLM systems for radiology report generation have matured fastest in chest X-ray (MIMIC-CXR, ~377,110 images / 227,835 reports [62]) and are now expanding into brain MRI specifically:

- **AutoRG-Brain** (2024) pairs automatic anomaly/structure segmentation with a visual-prompting language model to produce region-grounded findings, released alongside a supporting benchmark (RadGenome-Brain MRI, ~3,400 scans with paired masks and reports) and reportedly trialled with junior doctors using its output as a starting draft [PRD §18.4]. It targets general brain-MRI findings across disease types, not BraTS-style quantitative tumor sub-region reporting specifically.
- **Hallucination remains the field's central unsolved problem.** A 2026 narrative review of VLMs in diagnostic imaging found factual errors in roughly 22% of AI-generated reports across the reviewed literature, alongside a documented lack of external validation on diverse multi-institutional datasets and unresolved liability/workflow-integration barriers [65,68]. Multiple recent papers propose mitigations — direct preference optimization to suppress hallucinated references to prior exams [67], semantic-consistency-based uncertainty quantification with sentence-level confidence [61], fine-grained hallucination detection models like ReXTrust [63] — which collectively indicate this is treated as a first-class open problem, not a solved one.
- **Retrieval-augmented generation (RAG) for radiology reports** is an active 2025–2026 direction, again concentrated on chest X-ray: encode the query image, retrieve the most similar prior case(s) from a vector index (commonly FAISS), and condition the LLM's draft on the retrieved case's real report text, explicitly to reduce hallucination and keep the pipeline auditable rather than to remove the radiologist from the loop [PRD §18.8, citing LaB-RAG, RA-RRG, RAD-SRAC, and a 2026 grounded multimodal retrieval system]. **No system surfaced in this search applies this retrieval-augmented pattern to BraTS-style 3D tumor sub-region segmentation with quantitative (Dice/HD95/volume) grounding** — which is the gap your PRD addendum already identifies and that this independent search corroborates rather than contradicts.

---

## 17. Major Research Gaps

| Gap | Evidence | Why existing approaches fail | Opportunity | Difficulty | Novelty | Publication potential |
|---|---|---|---|---|---|---|
| **Quantitatively-grounded report drafting for volumetric segmentation** | AutoRG-Brain and RAG-for-CXR literature both exist; neither combines retrieval-augmented drafting with numeric Dice/HD95/volume grounding for BraTS-style output [38,39,PRD§18.4/18.8] | Existing systems generate free text from images, not text provably tied to the same numbers a clinician would check independently | Case-embedding + similarity retrieval + LLM drafting grounded in your own pipeline's metrics, gated by radiologist sign-off | Moderate (mostly composes existing V1-tier pieces) | Moderate–high | Workshop paper / applied-systems track realistic |
| **Latency-constrained ensembling** | Winning BraTS ensembles assume multi-GPU, multi-day budgets [4,5,6]; your own TRD sets a single-GPU, ~1–2 min/case target | No published study isolates how much ensembling's Dice/HD95 gain survives under a hard single-GPU inference budget | Systematic sweep: single model vs. 2–3 model ensemble at matched latency | Low–moderate | Moderate | Solid workshop/short-paper result |
| **Patient/site-level evaluation rigor in classification literature** | 29–55 percentage-point accuracy inflation documented from slice-level leakage in analogous neuroimaging tasks [42,43,44,46,48]; brain-tumor classification papers inconsistently report split methodology [45,49] | Slice pooling before splitting lets the model learn patient-specific anatomy rather than pathology | An audited, patient-level-split reproduction of the popular Figshare/Kaggle 2D classification benchmarks, reporting the *true* achievable accuracy | Low (mostly rigor, not novelty) | Low as a paper on its own, but strengthens any classification-adjacent claim you make | Good as a methods/negative-result note, not a standalone paper |
| **MGMT/IDH external-validation reporting norm** | 80.2% of 420 systematically trained MGMT models statistically indistinguishable from chance externally [27]; IDH degrades similarly but less catastrophically [51,54,58] | Small, single-institution training sets produce site-specific shortcuts that don't transfer | If you attempt radiogenomics at all, report external validation by default and treat a null result as a valid, publishable finding | Low to attempt, but requires access to a genuinely external cohort (e.g., UCSD-PTGBM [102]) | Low novelty in the prediction itself, moderate value in rigorous negative reporting | Best as a component of a larger paper, not standalone |
| **Cross-scanner/cross-site robustness on BraTS itself** | BraTS pools multiple institutions/scanners by design; FeTS 2024's federated results (Dice 73–76%) are meaningfully below single-institution ensembles [7]; almost no individual BraTS paper site-stratifies its own test set | Standard random splits within BraTS still mix scanners/sites across train/test in ways that may not reflect true out-of-distribution generalization | Site-stratified (leave-one-institution-out) evaluation of your trained model, reporting the Dice/HD95 gap vs. the standard random split | Moderate (requires site metadata, which BraTS provides) | Moderate–high | Genuinely useful, publishable finding even as a secondary contribution |
| **Radiologist-validated explainability** | Formal reader-study validation of Grad-CAM/SHAP output is rare — one clearly documented instance found [72] against dozens of papers that assert trust without testing it | Heatmap generation is treated as sufficient; validation against expert judgment is the exception, not the norm | A small, honestly-scoped reader study (even 2–3 radiologists, a modest case count) comparing your model's attention/uncertainty maps against expert-marked regions | Moderate (needs clinician time, which is the real bottleneck) | Moderate | Adds real weight to a paper's clinical-relevance claims if feasible within your constraints |
| **Uncertainty quantification mapped to report language** | Monte Carlo dropout / conformal prediction for glioma segmentation and IDH prediction both have precedent [54,PRD V1-tier] but are rarely connected to natural-language confidence statements in a generated report | Confidence scores and generated text are typically produced by separate, disconnected pipeline stages | Route your segmentation/IDH uncertainty estimate into the hedging language of the drafted report (e.g., "enhancing tumor volume estimate: moderate confidence") | Moderate–high | Moderate–high | Strong fit for the report-generation paper as an ablation/extension |

---

## 18. Problems With Existing Research

- **Benchmark traps are common and under-disclosed.** Slice-level data leakage, small single-source datasets, and absent external validation collectively explain much of the variance in reported accuracy across the 2D classification literature reviewed in §5/§9 — the same underlying task produces accuracy claims ranging from the mid-60s to 100% depending almost entirely on evaluation rigor rather than modeling novelty [42–49].
- **Radiogenomics has a documented reproducibility crisis for MGMT specifically**, and a less severe but real one for IDH — internal validation numbers in this sub-area should be treated with active skepticism until externally replicated [27,51,54,58].
- **Transformer gains over well-tuned CNNs are real but small** (roughly half a percentage point of Dice on BraTS 2021 for Swin UNETR vs. nnU-Net [25,26]), while transformer training/data requirements are meaningfully higher — a cost/benefit trade that favors CNNs and CNN-hybrids (MedNeXt) for constrained-compute projects like yours.
- **Explainability claims routinely outrun explainability validation.** The overwhelming majority of XAI papers in this space generate a heatmap and assert improved trust without testing that claim against expert judgment [70,71,73–77].
- **Multi-center external validation is the exception, not the rule**, across nearly every sub-area surveyed here — segmentation, classification, and radiogenomics alike. The 2024 meta-analysis of brain tumor detection/segmentation studies that specifically *required* external validation for inclusion only found 19 qualifying studies out of a much larger initial pool [116–118], which is itself informative about how rare rigorous external validation is in this literature.

---

## 19. What Has NOT Been Solved

- MGMT methylation prediction from preoperative MRI alone, at anything better than roughly chance level, externally [27].
- A standardized, enforced patient-level (and ideally site-level) evaluation protocol across the popular 2D classification benchmarks — the community keeps re-deriving the same Figshare/Kaggle numbers without a shared, leakage-audited evaluation harness.
- Report generation that is provably grounded in the same quantitative segmentation metrics a clinician would independently check, for volumetric tumor segmentation specifically (chest X-ray has this via RAG; brain-tumor segmentation does not yet, per this search) [PRD §18.4/18.8].
- Prospective (as opposed to retrospective, held-out-set) clinical validation of essentially any system reviewed here — every source found in this search evaluates on retrospective public or single-institution data.
- Reliable growth/progression prediction across timepoints for the same patient, which is structurally blocked by BraTS not providing longitudinal same-patient scans at scale — your own PRD's "Out of scope" framing of this is correct, not overly cautious.

---

## 20. Comparison With My PRD

| PRD Component | Existing research? | Closest papers | Current best approach | What my PRD adds | Novelty level |
|---|---|---|---|---|---|
| Custom 3D U-Net (nnU-Net-inspired) segmentation core | Extensively solved | nnU-Net [1]; MedNeXt [3]; Swin UNETR [2] | nnU-Net/MedNeXt ensembles | A from-scratch, understood, config-driven reproduction — pedagogically real, not a research contribution | **Already solved** |
| Dice + HD95 evaluation module with documented edge cases | Solved, field-standard | BraTS challenge protocol itself | Per-case macro-averaged Dice/HD95 with explicit empty/absent-class handling | Correct implementation of an existing standard | **Already solved** |
| 2D slice + 3D mesh visualization app | Engineering, not a research question | — | — | Genuinely useful demoability; not a novelty claim | **Already solved** (by design, per your own ADR-002) |
| Async FastAPI job service, checkpoint registry, case history | Engineering | — | — | Reproducibility and demoability | **Already solved** |
| Tumor volume / radiomics report (V1) | Solved via PyRadiomics; precedent in I3CR-WANO's end-to-end pipeline [PRD §18.5] | I3CR-WANO (Chakrabarty et al.) | Segment → extract PyRadiomics features → report | Straightforward integration of an established library | **Incremental** |
| Ensembling for accuracy (V1) | Extensively solved, current winning pattern on BraTS-family leaderboards [4,5,6] | Multiple 2025 BraTS sub-challenge winners | Multi-model ensembles at multi-GPU training budgets | A latency-constrained (single-GPU) version of the same idea, honestly scoped | **Incremental**, unless reframed around the latency constraint specifically |
| MGMT/radiogenomics classification head (Stretch) | Solved as a task, unreliable as a result | Kim et al. external validation study [27] | None reliably externally validated | An honest attempt with mandatory external validation reporting | **Low novelty in the prediction; moderate value in rigorous negative reporting** |
| Survival prediction (Stretch) | Solved as a task, dominated by a trivial baseline | Age-only linear regression [80,82] | Age-only baseline, rarely beaten for GTR patients | Framing your own attempt honestly against that baseline | **Low novelty** |
| Case-retrieval-augmented, quantitatively-grounded report drafting | **No direct precedent found for BraTS-style volumetric segmentation.** Closest analogues: AutoRG-Brain (general brain-MRI findings, not quantitative sub-region grounding) [PRD §18.4]; RAG-for-CXR literature (different modality) [PRD §18.8] | AutoRG-Brain; LaB-RAG/RA-RRG (chest X-ray) | Free-text generation from images, not numerically grounded in a segmentation pipeline's own metrics | Grounding generated findings directly in your Dice/HD95/volume numbers plus a retrieved similar case, gated by radiologist sign-off | **Moderately-to-strongly novel** — the most defensible original contribution available to this project |

---

## 21. Potential Novel Contributions (Ranked Research Directions)

**1. Case-retrieval-augmented, quantitatively-grounded report drafting** *(flagship direction)*
- **Research question:** Can a drafted radiology-style finding for a new BraTS-format case be grounded in (a) the pipeline's own Dice/HD95/volume numbers and (b) a retrieved similar prior case's report, in a way that measurably reduces hallucination versus ungrounded generation?
- **Hypothesis:** Numeric + retrieved-case grounding produces higher factual-consistency scores and shorter radiologist edit-distance than an ungrounded LLM baseline drafting from the image/mask alone.
- **Architecture:** 3D U-Net/MedNeXt segmentation → PyRadiomics + encoder-bottleneck case embedding → FAISS similarity index over your case history → LLM drafting step conditioned on (new case's own metrics, retrieved case's metrics + report) → radiologist sign-off gate.
- **Dataset:** BraTS training/validation split, self-generated report drafts (no existing brain-tumor-segmentation report corpus to train against, so evaluation must be structured around factual-consistency and edit-distance rather than reference-text overlap metrics).
- **Baselines:** ungrounded LLM captioning from the image; template-only report (no LLM); AutoRG-Brain-style ungrounded findings generation, if reproducible.
- **Metrics:** factual-consistency score against ground-truth Dice/HD95/volume; radiologist (or your own careful) edit-distance; retrieval-quality (top-k relevance of retrieved case).
- **Ablations:** with/without retrieval, with/without numeric grounding, retrieval-k sweep.
- **Expected improvement:** directional (fewer hallucinated findings), not necessarily a single headline accuracy number.
- **Difficulty:** moderate. **Novelty:** moderate–high. **Clinical significance:** genuine (auditability). **Publication potential:** workshop/applied-systems paper.

**2. Latency-constrained ensembling study**
- **Question:** how much of the Dice/HD95 gain from multi-model ensembling (§12) survives under a single-GPU, ~1–2 min/case budget?
- Baselines: nnU-Net alone; MedNeXt-S alone; Swin UNETR alone; 2-model and 3-model ensembles, all measured at matched wall-clock latency, not matched model count.
- **Difficulty:** low–moderate. **Novelty:** moderate. **Publication potential:** solid short paper/workshop result.

**3. Site-stratified (leave-one-institution-out) generalization audit**
- Use BraTS's institution metadata to hold out entire sites rather than random cases; report the Dice/HD95 gap versus a standard random split.
- **Difficulty:** moderate. **Novelty:** moderate–high (rarely done in individual BraTS papers). **Publication potential:** strong secondary contribution to any paper from this project.

**4. Uncertainty-calibrated segmentation routed into report confidence language**
- Monte Carlo dropout or conformal prediction on the segmentation output, mapped to explicit confidence phrasing in the drafted report from Direction 1.
- **Difficulty:** moderate–high. **Novelty:** moderate–high. **Publication potential:** strong extension/ablation of Direction 1, not necessarily standalone.

**5. Radiologist-validated explainability mini-study**
- A small, honestly-scoped comparison of Grad-CAM/attention or uncertainty maps against expert-marked regions, following the overlap-and-survey pattern of the one well-validated precedent found [72] — scaled to whatever clinician access you can realistically secure.
- **Difficulty:** moderate (bottlenecked by clinician time). **Novelty:** moderate. **Publication potential:** meaningful if feasible, skip if not.

**6. Honest, externally-validated radiogenomics attempt**
- Train an MGMT and/or IDH classifier on BraTS 2021, evaluate on a genuinely external cohort (e.g., UCSD-PTGBM [102]), and report the result — including a null result — as a first-class finding rather than something to hide.
- **Difficulty:** low to attempt, contingent on external-cohort access. **Novelty:** low in the prediction itself, real value in rigorous reporting. **Publication potential:** best folded into a larger paper as a "we checked, and here's what we honestly found" section.

**7. Self-supervised pretraining vs. transfer learning under fixed labeled-data budget**
- Pretrain a 3D encoder on unlabeled/lightly-labeled volumes (masked-patch reconstruction) and compare against MedicalNet/Med3D transfer learning and training-from-scratch, all at matched labeled-data budgets.
- **Difficulty:** moderate–high (separate training stage). **Novelty:** moderate. **Publication potential:** solid if time allows; likely first to be cut under a 3-month deadline.

---

## 22. Recommended Research Architecture

```
4-Modality NIfTI Case (T1, T1ce, T2, FLAIR)
        │
        ▼
Preprocessing (nnU-Net-style fingerprinting/resample/z-score — already in TRD)
        │
        ▼
Segmentation backbone: nnU-Net-style 3D U-Net (baseline) ──ablation──> MedNeXt-S / Swin UNETR
        │
        ├──> PyRadiomics feature extraction (shape/texture on predicted mask)
        ├──> Encoder-bottleneck or radiomics case embedding
        │
        ▼
Case-embedding similarity index (FAISS) over case-history corpus
        │
        ▼
LLM drafting step, conditioned on:
   - this case's own Dice/HD95/volume/radiomics numbers
   - the retrieved similar case's numbers + report text
        │
        ▼
Radiologist review / sign-off gate  ──>  Results Dashboard (existing V1 tier)
```

This composes almost entirely from pieces already scoped in your V1 tier (tumor volume/breakdown report, plain-English findings summary, case history, checkpoint registry) plus the two genuinely new components: the case-embedding similarity index and the LLM drafting step, as your own addendum already concluded [PRD §18.9].

---

## 23. Recommended Experiments and Ablations

- **Patient-level split audit** of any classification component you add, before trusting any accuracy number (§8, §19).
- **Site-stratified test split** on BraTS, alongside the standard random split, to report the generalization gap explicitly.
- **Modality ablation**: drop each of T1/T1c/T2/FLAIR individually and re-evaluate Dice/HD95, following the precedent set by multi-sequence IDH/grading studies [51].
- **Architecture ablation**: nnU-Net vs. MedNeXt-S vs. Swin UNETR, all under your stated latency budget, not just raw Dice.
- **Ensemble-vs-single-model at matched latency** (Direction 2 above).
- **Retrieval-k ablation** for the RAG drafting step (Direction 1) — how many retrieved cases, and does adding more help or just add noise.
- **Factual-consistency and edit-distance evaluation** of drafted reports against ground-truth metrics and (if feasible) your own or a colleague's manual review.
- **Explainability evaluation**: Grad-CAM/uncertainty-map overlap against manually marked regions if any clinician access is available (§21.5).
- **Computational efficiency evaluation**: report inference latency, VRAM, and parameter count for every architecture/ensemble configuration tested — this is a first-class metric for this project's specific constraints, not an afterthought.

---

## 24. Recommended Baselines

| Baseline | Appropriate for | Why / why not |
|---|---|---|
| nnU-Net | Segmentation | **Mandatory.** It is the field's reference baseline; any segmentation paper without it invites an immediate reviewer objection. |
| MedNeXt-S | Segmentation | Strong "modern CNN" comparator, appropriate for your single-GPU budget [3,126]. |
| Swin UNETR / UNETR | Segmentation | Transformer comparator — include to show you considered it, expect it to roughly match or trail nnU-Net/MedNeXt at your scale [25,26]. |
| 2–3 model ensemble (nnU-Net + MedNeXt, or + Swin UNETR) | Segmentation | Matches current field practice; pair with the latency-matched ablation from Direction 2. |
| VGG16 / ResNet50 / DenseNet / MobileNet / EfficientNet | **Not segmentation baselines.** Only relevant if you add a 2D screening/triage classification head. | These are 2D ImageNet-style classification backbones from a different sub-literature (§4, §9); including them as segmentation baselines would be a category error your PRD's own §18.1 already correctly avoids. |

---

## 25. Publication Strategy

Given a 1–2 person team and a 3-month timeline, a realistic target is a **MICCAI satellite/workshop paper** (the BrainLes workshop specifically runs alongside the BraTS challenge and is the natural home for BraTS-adjacent systems work) or an **arXiv technical report paired with an open-source release**, rather than a top-tier journal submission on the first pass. Suggested structure:

1. **Title (draft):** *"Grounding AI-Drafted Findings in Quantitative Segmentation Metrics: A Retrieval-Augmented Report-Drafting Pipeline for Volumetric Brain Tumor MRI"*
2. **Research problem:** AI-generated radiology findings for brain-tumor MRI are not provably tied to the same quantitative metrics (Dice, HD95, volume) a clinician would independently check, unlike the retrieval-augmented pattern now established for chest X-ray.
3. **Research gap:** no system in the reviewed literature applies numeric-and-case grounding to BraTS-style volumetric tumor sub-region reporting specifically.
4. **Research question / hypothesis:** as in §21, Direction 1.
5. **Novel contribution:** the grounding mechanism and its evaluation, not the segmentation backbone.
6. **Methodology / experimental design:** as in §22–24.
7. **Expected results:** directional improvement in factual consistency and reduced hallucination versus an ungrounded baseline; honestly report any null results (especially if you also attempt radiogenomics, per Direction 6).
8. **Limitations:** small case-history corpus for retrieval at project scale; no prospective clinical evaluation; single-institution (BraTS-composited) data.
9. **Clinical implications:** framed strictly as decision support with mandatory sign-off, consistent with your PRD's existing non-clinical-use disclaimer.
10. **Target venues:** BrainLes/MICCAI workshop track; if journal-bound, mid-tier venues like *Computers in Biology and Medicine*, *Journal of Medical Imaging*, or a *Frontiers in Oncology/Radiology* AI special issue are realistic given project scope — top-tier venues (*Medical Image Analysis*, *Nature Methods*) would need a stronger empirical validation (multi-institution, prospective) than a 3-month solo project can deliver.

---

## 26. Top 5 Most Promising Research Directions

1. **Case-retrieval-augmented, quantitatively-grounded report drafting** — the only direction with no direct precedent in the reviewed literature for this exact modality/task combination, and it composes from pieces you've already scoped.
2. **Site-stratified generalization audit** — cheap to run (BraTS provides institution metadata), rarely done, and strengthens any other claim in the paper.
3. **Latency-constrained ensembling** — directly answers a question your own TRD's constraints raise, that the field's multi-GPU-budget ensembling papers don't answer.
4. **Honest, externally-validated radiogenomics attempt** — low cost to attempt, genuine value in rigorous negative reporting given the field's documented reproducibility problems.
5. **Uncertainty-calibrated report confidence language** — a natural, moderate-effort extension of Direction 1 rather than a separate project.

---

## 27. Final Recommendation

Build the segmentation core solidly — it is necessary infrastructure and a legitimate implementation exercise, but do not present it, an ensemble of it, or a 2D-classification-backbone comparison as the paper's novel contribution; all three are saturated. Do not oversell radiogenomics: if you attempt MGMT or IDH prediction, budget for external validation and be prepared to report a null result as a real finding, because that is what the strongest evidence in this space (§16, §19) suggests is the honest expectation. The most defensible, buildable, and genuinely under-explored angle available to this project is the **case-retrieval-augmented, quantitatively-grounded report-drafting system**, evaluated with the rigor (patient/site-level splitting, external-validation instincts, honest negative-result reporting) that this review found conspicuously absent across much of the surrounding literature.

---

## 28. "If I wanted to publish this paper, this is the exact gap I would target"

**The gap:** AI systems that draft radiology-style findings from brain MRI are not, in the published literature reviewed here, provably grounded in the same quantitative segmentation metrics (Dice, HD95, tumor sub-region volume) that a clinician reviewing the case would independently compute and check — for volumetric tumor sub-region segmentation specifically, as distinct from general findings generation.

**Why it matters:** Hallucination in AI-generated radiology reports is a documented, unresolved problem — roughly 22% factual-error rates are reported across the current VLM-for-diagnostic-imaging literature [65,68] — and the field's own proposed mitigation, retrieval-augmented grounding, exists and works for chest X-ray but has not, as far as this search could establish, been extended to the volumetric-segmentation case where a rich set of quantitative outputs (Dice, HD95, per-class volume) already exists to ground the text in, rather than relying on retrieved text alone.

**Evidence that it exists:** AutoRG-Brain (2024) targets general brain-MRI findings, not quantitative BraTS-style sub-region grounding [PRD §18.4]. The RAG-for-radiology-reports literature (LaB-RAG, RA-RRG, RAD-SRAC, and related 2025–2026 work) is concentrated on chest X-ray / MIMIC-CXR-style data and grounds drafts in retrieved *text*, not a segmentation pipeline's own *numbers* [PRD §18.8]. No paper surfaced in ~30 targeted searches combines both for brain tumor segmentation.

**What existing papers fail to do:** they generate free text from images (with or without retrieval), but do not tie that text to a verifiable, independently-recomputable set of numbers produced by the same pipeline generating the text.

**My proposed solution:** the pipeline in §22 — segment, extract case-level metrics and an embedding, retrieve the most similar prior case, draft a finding conditioned on both the new case's own metrics and the retrieved case's metrics and report text, and gate everything behind radiologist sign-off before it is stored as final.

**Why it is novel:** it is not a new segmentation architecture (that space is saturated) and not a new report-generation architecture (VLMs and RAG both exist) — it is a specific, previously-unexamined combination: numeric self-grounding plus case retrieval, applied to a task (volumetric tumor sub-region segmentation) where the numeric grounding signal is unusually rich and currently unused for this purpose.

**How I would experimentally prove it:** compare factual-consistency scores and (if any reviewer access is available) edit-distance/acceptance rates between (a) an ungrounded LLM caption of the segmentation output, (b) a retrieval-only draft, and (c) the full numeric-plus-retrieval-grounded draft, across an ablation sweep on retrieval-k and with/without numeric conditioning, evaluated on a held-out BraTS split.

**What would make the contribution publishable:** honest, patient/site-aware evaluation; explicit reporting of failure cases and any null results; a clearly-scoped, non-clinical-use framing throughout, matching the disclaimer already built into this project; and release of the pipeline/code so the specific grounding mechanism — not the segmentation backbone — is the reproducible artifact under review.

---

## 29. Sources

*Segmentation architectures & challenges*
[1] Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods* 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
[2] Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images," MICCAI BrainLes 2021. https://arxiv.org/abs/2201.01266
[3] Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation," MICCAI 2023. https://arxiv.org/abs/2303.09975
[4] "BraTS-PEDs: Results of the Multi-Consortium International Pediatric Brain Tumor Segmentation Challenge 2023." https://arxiv.org/pdf/2407.08855
[5] "Frequency-Aware Ensemble Learning for BraTS 2025 Pediatric Brain Tumor Segmentation." https://arxiv.org/pdf/2509.19353
[6] "How We Won BraTS-SSA 2025." https://arxiv.org/pdf/2510.03568
[7] "The MICCAI Federated Tumor Segmentation (FeTS) Challenge 2024." https://arxiv.org/pdf/2512.06206
[8] "The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI." https://arxiv.org/abs/2405.18368

*Transformer/hybrid segmentation benchmarks*
[10] "From Convolution to Transformer: A Comparative Study of U-Net Variants for Brain Tumor and Retinal Vessel Segmentation." https://arxiv.org/pdf/2606.22168
[23] "A4-Unet: Deformable Multi-Scale Attention Network for Brain Tumor Segmentation" (comparison table across BraTS 2019/2020/2021). https://arxiv.org/pdf/2412.06088
[25],[26],[29],[31] Hatamizadeh et al., Swin UNETR (as above) and related review. https://arxiv.org/pdf/2201.01266

*Attention mechanisms*
[130] "LATUP-Net: A Lightweight 3D Attention U-Net with Parallel Convolutions." https://arxiv.org/pdf/2404.05911
[131] "RDAU-Net... CBAM for Brain Tumor Segmentation," *Frontiers in Oncology* 2022. https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.805263/full
[132],[134] "A combined attention mechanism for brain tumor segmentation of lower-grade glioma" (ECASE-Unet). https://www.sciencedirect.com/science/article/abs/pii/S0010482525007310
[138] "Comparative Study of UNet-based Architectures... liver tumor" (SE/CBAM ablation table). https://arxiv.org/pdf/2510.25522

*2D classification benchmarks*
[13] Springer UCAmI 2025, ResNet101/ResNet50/VGG16 Gabor comparison. https://link.springer.com/chapter/10.1007/978-3-032-16992-1_25
[14] "Brain tumor classification using fine-tuned transfer learning models," PMC11459499.
[15] "Performance Comparison of ResNet50, VGG16, and MobileNetV2." ResearchGate 389752044.
[16] "MRI-Based Brain Tumor Classification Using Ensemble CNN, VGG16, and ResNet50." Springer 2025.
[17] "Brain Tumor Classification using VGG16, ResNet50, and Inception V3." ResearchGate 370150476.
[18] "Comparative Analysis of Resource-Efficient CNN Architectures for Brain Tumor Classification." https://arxiv.org/pdf/2411.15596
[19] "Transfer learning for accurate brain tumor classification in MRI," PMC12149374.
[20] "Brain Tumor Classification in MRI Images: A Computationally Efficient CNN." https://arxiv.org/pdf/2605.12560
[21] "Brain Tumor Classification... Combined Transfer Learning and CNNs," PubMed 42346896.
[22] "An interpretable vision transformer framework for automated brain tumor classification." https://arxiv.org/pdf/2604.21311

*Data leakage*
[42],[43],[44],[46] "Effect of data leakage in brain MRI classification using 2D CNNs," *Scientific Reports* (2021). https://www.nature.com/articles/s41598-021-01681-w
[45] "Hybrid deep NN with PCA... brain tumor," PMC13021980.
[48] "An analysis of data leakage and generalizability in MRI-based classification of Parkinson's Disease." https://www.sciencedirect.com/science/article/abs/pii/S1051200424000320
[49] "A multi-scale attention and efficient convolution framework for brain tumor classification" (leakage-check protocol). https://www.sciencedirect.com/science/article/pii/S2772442526000353

*Datasets*
[85]–[94] Cheng et al. Figshare dataset and derivatives; Kaggle "Brain Tumor MRI Dataset." Representative: https://www.nature.com/articles/s41598-024-74731-8 ; https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
[164],[167] "The BraTS-Africa Dataset," *Radiology: AI* (2025). https://doi.org/10.1148/ryai.240528
[165],[168] Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark." https://arxiv.org/abs/2107.02314
[173] "Unsupervised Deep Generative Models for Anomaly Detection in Neuroimaging" (BraTS dataset-size history table). https://arxiv.org/pdf/2510.14462
[102] "Explainable Deep Radiogenomic Molecular Imaging for MGMT Methylation Prediction" (UCSD-PTGBM external cohort). https://arxiv.org/html/2601.07035

*Radiogenomics*
[27],[95]–[101] Kim et al., "Validation of MRI-Based Models to Predict MGMT Promoter Methylation in Gliomas: BraTS 2021 Radiogenomics Challenge," *Cancers* 14(19):4827 (2022). https://doi.org/10.3390/cancers14194827
[28],[53] "Diagnostic performance of deep learning for predicting glioma IDH and 1p/19q co-deletion: a systematic review and meta-analysis." https://arxiv.org/pdf/2411.02426
[50] "MRI-Based Radiomics for Non-Invasive Prediction of Molecular Biomarkers in Gliomas," PMC12897266.
[51] "Robust deep learning for incomplete MRI sequences in glioma grading and IDH prediction," PubMed 42142114.
[54] "Predicting IDH Mutation... with Conformal Prediction," PMC13103005.
[58] "External Validation of a CNN for IDH Mutation Prediction," PMC9025144.
[59] "An externally validated ML model for predicting IDH mutation," PMC11630777.

*Grading*
[32],[158],[159] Sun et al., "Performance of deep learning algorithms to distinguish high-grade glioma from low-grade glioma: a systematic review and meta-analysis," *iScience* (2023). https://doi.org/10.1016/j.isci.2023.106815
[162] "Glioma classification in MRI using a hybrid deep learning framework with majority vote ensemble." ScienceDirect.

*Survival prediction*
[80],[82] "Robustness of Radiomics for Survival Prediction of Brain Tumor Patients Depending on Resection Status," *Frontiers in Computational Neuroscience* (2019). PMC6857096.
[84] "Brain Tumor Segmentation and Survival Prediction Using Multimodal MRI Scans With Deep Learning." ResearchGate 335213676.

*Systematic reviews / meta-analyses of segmentation & detection*
[33],[116]–[118] "Artificial Intelligence Detection and Segmentation Models: A Systematic Review and Meta-Analysis of Brain Tumors in MRI." PMC11976016.
[112]–[114] "Brain metastasis tumor segmentation and detection using deep learning algorithms: A systematic review and meta-analysis." PubMed 37967585.
[119] "Deep Learning for Brain Tumor Segmentation and Classification: A Systematic Review of Methods and Trends." ScienceDirect S154622182501015X.
[120] "Advancing Brain Tumor Diagnosis Using Deep Learning: A Systematic and Critical Review." PMC13204197.

*Explainability*
[70] "Exploring the potential of explainable AI in brain tumor detection and classification: a systematic review," *Artificial Intelligence Review* (2025). Springer.
[72] "Explainable AI in Diagnostic Radiology for Neurological Disorders: A Systematic Review, and What Doctors Think About It." PMC11764244.

*Report generation / VLMs / RAG*
[38],[39] AutoRG-Brain, https://arxiv.org/abs/2407.16684 ; RadGenome-Brain MRI (per PRD §18.4).
[60] "Hallucination Mitigating for Medical Report Generation." https://arxiv.org/pdf/2601.15745
[61] "Semantic Consistency-Based Uncertainty Quantification for Factuality in Radiology Report Generation." PubMed 41488130.
[62] "Vision-language models for medical report generation and visual question answering: a review," *Frontiers in AI* (2024).
[63] "ReXTrust: A Model for Fine-Grained Hallucination Detection in AI-Generated Radiology Reports." https://arxiv.org/pdf/2412.15264
[65],[68] "Vision-language models in diagnostic imaging: review of technical advances, clinical validation, and practical deployment." PubMed 41483727.
[67] "Direct Preference Optimization for Suppressing Hallucinated Prior Exams in Radiology Report Generation." https://arxiv.org/pdf/2406.06496

*GAN augmentation*
[140]–[147] Representative: "Multi-modal brain tumor segmentation via conditional synthesis with Fourier domain adaptation," PubMed 38245925 (4–5% Dice gain); "Synthesis of Glioblastoma Segmentation Data Using GAN," Springer 2024 (0.90→0.94 accuracy); "Assessment of Using Synthetic Data in Brain Tumor Segmentation," https://arxiv.org/pdf/2508.11922.

*Lightweight/edge*
[103]–[111] Representative: "A novel lightweight CNN design for MRI brain tumor image classification." *Discover Computing* (2025); "A minimal-net CNN model for an IoT-based brain tumor detection and monitoring system." PMC13358162.

*MedNeXt / modern architectures*
[121]–[129] "MedNeXt" (as [3] above) and follow-on applications, including CoMNet (https://arxiv.org/pdf/2606.15305) and BraTS-Africa fine-tuning work (https://arxiv.org/pdf/2412.14100).

*Transformer data requirements*
[148]–[155] Representative: "Application of transformer models in medical image segmentation: a narrative review." PMC13178340; "MoViT: Memorizing Vision Transformers for Medical Image Analysis." https://arxiv.org/pdf/2303.15553

**Note on citation completeness:** bracket numbers above correspond to the search results reviewed while building this document; several ranges (e.g., [85]–[94], [140]–[147]) group multiple closely-related sources on the same point rather than listing each individually, in the interest of a readable reference list. If you need a fully expanded, deduplicated bibliography in a specific citation format (BibTeX, APA, IEEE) for an actual paper draft, that's a quick follow-up — say the word and I'll produce it.
