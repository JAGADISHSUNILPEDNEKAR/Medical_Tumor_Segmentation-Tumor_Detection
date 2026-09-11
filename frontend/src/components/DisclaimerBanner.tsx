export function DisclaimerBanner() {
  return (
    <div
      role="note"
      aria-label="Non-clinical use disclaimer"
      className="border-b border-caution-800/20 bg-caution-50 px-4 py-3 text-caution-800"
    >
      <p className="mx-auto max-w-6xl text-sm leading-relaxed">
        <strong className="font-semibold">Research prototype.</strong> This
        application is not a medical device, is not clinically validated, and
        must not be used for diagnosis, treatment, or other clinical decisions.
        AI-generated analysis is not a clinical diagnosis.
      </p>
    </div>
  );
}
