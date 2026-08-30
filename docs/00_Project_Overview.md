# Project Overview

The **Medical Image Segmentation & Tumor Detection System** bridges multi-modal 3D brain MRI volumes (BraTS-format T1, T1ce, T2, FLAIR scans) with a full-stack decision-support application for visualizing and quantifying brain tumor sub-regions.

Manual delineation of tumor sub-regions (necrotic core, edema, enhancing tumor) in multi-modal MRI is slow, inter-observer variable, and a bottleneck in both clinical workflow and research. This project trains a custom 3D U-Net — conceptually grounded in nnU-Net (Isensee et al., *Nature Methods*, 2021) — on the BraTS dataset, wraps sliding-window inference in a FastAPI service, and exposes 2D slice + 3D mesh visualization plus Dice/Hausdorff95 evaluation through a React dashboard. It is delivered as a demonstrable, reproducible, end-to-end system: upload a scan → run inference → view the 3D segmentation → read Dice/HD95 metrics → view a results dashboard.

> **Positioning:** Research / decision-support prototype. **Not a diagnostic device. Not clinically validated.** Every clinical-facing screen carries a persistent non-clinical-use disclaimer.

```
       ┌────────────────────────────────────────────────────────┐
       │       RADIOLOGIST / ONCOLOGIST / ML RESEARCHER          │
       └───────────────────────────┬────────────────────────────┘
                                   │
               ┌───────────────────▼───────────────────┐
               │  REACT + TAILWIND 3D VIEWER DASHBOARD  │
               │  (2D Slice Viewer + react-three-fiber) │
               └───────────────────┬───────────────────┘
                                   │ REST APIs
               ┌───────────────────▼───────────────────┐
               │       FASTAPI INFERENCE SERVICE        │
               │   (Async Job Queue + Structured Logs)  │
               └───────────┬───────────────┬───────────┘
                           │               │
        ┌──────────────────┴──┐         ┌──┴──────────────────┐
        │  PYTORCH 3D U-NET   │         │ SQLITE / POSTGRES DB│
        │ - nnU-Net inspired  │         │ - Case/Job Records  │
        │ - Sliding Window    │         │ - Dice/HD95 Metrics │
        │ - Dice + CE Loss    │         │ - Checkpoint Registry│
        └─────────────────────┘         └─────────────────────┘
```

**Track:** Multimodal AI (Computer Vision) · **Dataset:** BraTS (Brain Tumor Segmentation Challenge) · **Duration:** 3 Months
