import clsx from "clsx";
import type { ReactNode } from "react";

interface Tab {
  id:    string;
  label: string;
  icon?: ReactNode;
}

interface TabsProps {
  tabs:     Tab[];
  active:   string;
  onChange: (id: string) => void;
  className?: string;
}

export function Tabs({ tabs, active, onChange, className }: TabsProps) {
  return (
    <div className={clsx("flex gap-1 p-1 bg-surface2 rounded-lg border border-border", className)}>
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={clsx(
            "flex items-center gap-1.5 px-4 py-2 rounded-md text-sm font-medium transition-all",
            active === t.id
              ? "bg-surface text-text shadow-sm border border-border"
              : "text-muted hover:text-text"
          )}
        >
          {t.icon}
          {t.label}
        </button>
      ))}
    </div>
  );
}

interface TabsUnderlineProps {
  tabs:     Tab[];
  active:   string;
  onChange: (id: string) => void;
}

export function TabsUnderline({ tabs, active, onChange }: TabsUnderlineProps) {
  return (
    <div className="flex gap-6 border-b border-border">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={clsx(
            "flex items-center gap-1.5 pb-3 text-sm font-medium transition-all",
            "border-b-2 -mb-px",
            active === t.id
              ? "text-primary border-primary"
              : "text-muted border-transparent hover:text-text"
          )}
        >
          {t.icon}
          {t.label}
        </button>
      ))}
    </div>
  );
}
