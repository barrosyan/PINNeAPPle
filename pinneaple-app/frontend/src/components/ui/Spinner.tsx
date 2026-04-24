import clsx from "clsx";

interface SpinnerProps {
  size?: "sm" | "md" | "lg";
  className?: string;
}

const SIZES = { sm: "w-4 h-4", md: "w-6 h-6", lg: "w-10 h-10" };

export function Spinner({ size = "md", className }: SpinnerProps) {
  return (
    <div
      className={clsx(
        "border-2 border-border border-t-primary rounded-full animate-spin",
        SIZES[size],
        className
      )}
    />
  );
}

export function FullPageSpinner() {
  return (
    <div className="flex flex-col items-center justify-center flex-1 gap-3">
      <Spinner size="lg" />
      <span className="text-muted text-sm">Loading…</span>
    </div>
  );
}

interface ProgressBarProps {
  value: number;   // 0..100
  label?: string;
  animated?: boolean;
}

export function ProgressBar({ value, label, animated = true }: ProgressBarProps) {
  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-xs text-muted mb-1">
          <span>{label}</span>
          <span>{Math.round(value)}%</span>
        </div>
      )}
      <div className="h-2 bg-surface3 rounded-full overflow-hidden">
        <div
          className={clsx(
            "h-full bg-primary rounded-full transition-all duration-300",
            animated && "relative overflow-hidden"
          )}
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        >
          {animated && (
            <div className="absolute inset-0 bg-white/20 animate-pulse" />
          )}
        </div>
      </div>
    </div>
  );
}
