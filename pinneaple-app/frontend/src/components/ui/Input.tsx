import clsx from "clsx";
import type { InputHTMLAttributes, TextareaHTMLAttributes } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export function Input({ label, error, className, ...props }: InputProps) {
  return (
    <div className="w-full">
      {label && <label className="label">{label}</label>}
      <input
        className={clsx("input-base", error && "border-error focus:ring-error", className)}
        {...props}
      />
      {error && <p className="mt-1 text-xs text-error">{error}</p>}
    </div>
  );
}

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
}

export function Textarea({ label, error, className, ...props }: TextareaProps) {
  return (
    <div className="w-full">
      {label && <label className="label">{label}</label>}
      <textarea
        className={clsx(
          "input-base resize-none",
          error && "border-error focus:ring-error",
          className
        )}
        {...props}
      />
      {error && <p className="mt-1 text-xs text-error">{error}</p>}
    </div>
  );
}

interface NumberInputProps {
  label?: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: "int" | "float";
  className?: string;
}

export function NumberInput({
  label, value, onChange, min, max, step = 1, format = "int", className,
}: NumberInputProps) {
  return (
    <div className="w-full">
      {label && <label className="label">{label}</label>}
      <input
        type="number"
        className={clsx("input-base", className)}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = format === "int" ? parseInt(e.target.value) : parseFloat(e.target.value);
          if (!isNaN(v)) onChange(v);
        }}
      />
    </div>
  );
}
