interface SliderProps {
  label?: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  showValue?: boolean;
  unit?: string;
}

export function Slider({
  label, value, onChange, min, max, step = 1, showValue = true, unit = "",
}: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className="w-full">
      {(label || showValue) && (
        <div className="flex justify-between items-center mb-1">
          {label && <label className="label mb-0">{label}</label>}
          {showValue && (
            <span className="text-xs font-mono text-primary">
              {value}{unit}
            </span>
          )}
        </div>
      )}
      <div className="relative h-2 bg-surface3 rounded-full">
        <div
          className="absolute left-0 top-0 h-full bg-primary rounded-full"
          style={{ width: `${pct}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full opacity-0 cursor-pointer h-full"
        />
      </div>
      <div className="flex justify-between text-xs text-muted mt-1">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}
