import clsx from "clsx";
import type { ReactNode } from "react";

type Variant = "default" | "primary" | "success" | "warning" | "error" | "secondary";

const STYLES: Record<Variant, string> = {
  default:   "bg-surface3 text-muted border-border",
  primary:   "bg-primary/15 text-primary border-primary/30",
  success:   "bg-success/15 text-success border-success/30",
  warning:   "bg-warning/15 text-warning border-warning/30",
  error:     "bg-error/15 text-error border-error/30",
  secondary: "bg-secondary/15 text-secondary border-secondary/30",
};

interface BadgeProps {
  children: ReactNode;
  variant?: Variant;
  className?: string;
}

export function Badge({ children, variant = "default", className }: BadgeProps) {
  return (
    <span className={clsx("badge border", STYLES[variant], className)}>
      {children}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, Variant> = {
    done:    "success",
    running: "warning",
    error:   "error",
    stopped: "default",
    pending: "secondary",
    created: "default",
  };
  return <Badge variant={map[status] ?? "default"}>{status}</Badge>;
}
