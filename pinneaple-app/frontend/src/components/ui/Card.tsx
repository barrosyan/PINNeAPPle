import clsx from "clsx";
import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  action?: ReactNode;
}

export function Card({ children, className, title, subtitle, action }: CardProps) {
  return (
    <div className={clsx("card p-5", className)}>
      {(title || action) && (
        <div className="flex items-start justify-between mb-4">
          <div>
            {title    && <h3 className="text-base font-semibold text-text">{title}</h3>}
            {subtitle && <p className="text-xs text-muted mt-0.5">{subtitle}</p>}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  accent?: boolean;
  className?: string;
}

export function MetricCard({ label, value, sub, accent, className }: MetricCardProps) {
  return (
    <div
      className={clsx(
        "card p-4",
        accent && "border-primary/30 shadow-glow",
        className
      )}
    >
      <div className="text-xs text-muted uppercase tracking-wide mb-1">{label}</div>
      <div className={clsx("text-2xl font-bold", accent ? "text-primary" : "text-text")}>
        {value}
      </div>
      {sub && <div className="text-xs text-muted mt-1">{sub}</div>}
    </div>
  );
}
