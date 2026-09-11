import type { ReactNode } from "react";

import { DisclaimerBanner } from "./DisclaimerBanner";

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:bg-white focus:px-3 focus:py-2"
      >
        Skip to main content
      </a>
      <DisclaimerBanner />
      <header className="border-b border-ink-900/10 bg-white">
        <div className="mx-auto flex max-w-6xl items-baseline justify-between gap-4 px-4 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-accent-700">
              Multimodal MRI research
            </p>
            <p className="font-display text-xl text-ink-950">
              Tumor Segmentation Workbench
            </p>
          </div>
          <p className="hidden text-sm text-ink-500 sm:block">Phase 1 foundation</p>
        </div>
      </header>
      <main id="main-content" className="flex-1">
        {children}
      </main>
      <footer className="border-t border-ink-900/10 bg-white px-4 py-4 text-sm text-ink-500">
        <p className="mx-auto max-w-6xl">
          BraTS-format T1, T1ce, T2, and FLAIR research workflow. Public
          de-identified research data only.
        </p>
      </footer>
    </div>
  );
}
