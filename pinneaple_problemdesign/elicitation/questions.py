from __future__ import annotations

from typing import List
from ..schema import Gap


def build_stage_gaps(stage: str, spec_dict: dict) -> List[Gap]:
    gaps: List[Gap] = []

    def add(gid: str, q: str, sev: str, why: str, how: str):
        gaps.append(Gap(id=gid, question=q, severity=sev, rationale=why, how_to_obtain=how))

    if stage == "intake":
        if not spec_dict.get("title"):
            add("intake.title", "What is the project/problem title?", "important",
                "Helps consolidate scope and communication.", "Provide a short, descriptive title.")
        if not spec_dict.get("goal"):
            add("intake.goal", "What is the end goal (what do you want to predict/estimate/optimize)?", "blocker",
                "We cannot define targets or validation without a goal.", "Describe the desired output and use case.")
        if spec_dict.get("task_type", "other") == "other":
            add("intake.task_type", "Which task type fits best (forecasting/inverse/PDE/operator/control/optimization)?", "important",
                "Task type selects the modeling strategy.", "Pick the closest option and explain briefly.")
        if not spec_dict.get("domain_context"):
            add("intake.context", "Describe the system in 3–6 sentences (process, variables, operating conditions).", "important",
                "Context drives assumptions and risks.", "Explain as you would to another engineer.")

    if stage == "io_and_time":
        if not spec_dict.get("inputs"):
            add("io.inputs", "What are the input features (names + units)?", "blocker",
                "We need inputs to define the model interface.", "List sensor/feature names and units.")
        if not spec_dict.get("outputs"):
            add("io.outputs", "What are the target outputs to forecast/estimate (names + units)?", "blocker",
                "Targets define training labels and metrics.", "List target variables and units.")
        if not spec_dict.get("frequency"):
            add("time.frequency", "What is the sampling frequency (Δt)?", "important",
                "Defines windowing, horizon, and temporal splits.", "Example: 1Hz, 10min, irregular.")
        if not spec_dict.get("input_window"):
            add("time.input_window", "What initial history window length (input_window) should we use?", "important",
                "Affects memory and leakage risk.", "Example: 64 steps, 24h.")
        if not spec_dict.get("horizon"):
            add("time.horizon", "What forecast horizon (horizon) is required?", "important",
                "Horizon defines the prediction target.", "Example: 16 steps, 6h, 24h.")

    if stage == "data":
        data = spec_dict.get("data", {}) or {}
        if not data.get("sources"):
            add("data.sources", "What are the data sources (files/DB/API/sensors)?", "blocker",
                "No source => no pipeline.", "State where data lives and who owns it.")
        if not data.get("format"):
            add("data.format", "What is the data format (csv/parquet/zarr/hdf5/etc)?", "important",
                "Affects ingestion and validation.", "Provide format and approximate volume.")
        if not data.get("sampling"):
            add("data.sampling", "Is sampling regular? Any missing timestamps or irregular sampling?", "important",
                "Impacts windowing and imputation.", "Describe missingness patterns.")
        if not data.get("variables_observed"):
            add("data.variables_observed", "Which variables are observed (columns/signals)?", "important",
                "Defines eligible inputs.", "List the main observed variables.")
        if not data.get("target_variables"):
            add("data.target_variables", "Which variables are the targets?", "blocker",
                "Defines y labels.", "List target variables and whether derived.")
        if not data.get("val_split_policy"):
            add("data.val_split_policy", "How should validation be done (temporal split policy)?", "important",
                "Prevents leakage and measures real generalization.", "Suggested: last 20% time as validation.")

    if stage == "physics":
        phys = spec_dict.get("physics", {}) or {}
        if not phys.get("governing_equations") and not phys.get("constraints"):
            add("physics.constraints", "Do you have equations or soft constraints (bounds, monotonicity, conservation)?", "nice_to_have",
                "Can improve generalization and stability.", "If no equations, state known constraints and limits.")
        if phys.get("governing_equations") and not phys.get("units"):
            add("physics.units", "What are the variable units used in the equations?", "important",
                "Units/scaling are critical for physics-consistent training.", "Provide unit per variable symbol.")

    if stage == "constraints":
        cons = spec_dict.get("constraints", {}) or {}
        dep = spec_dict.get("deployment", {}) or {}
        if not cons.get("hardware"):
            add("constraints.hardware", "What hardware is available (CPU/GPU/VRAM)?", "important",
                "Sets feasible architectures and batch sizes.", "Example: CPU-only, 1xGPU 16GB.")
        if not cons.get("max_training_time"):
            add("constraints.max_training_time", "What maximum training time is acceptable?", "nice_to_have",
                "Helps choose baseline and tuning strategy.", "Example: 30min, 2h, overnight.")
        if dep.get("latency_budget_ms") is None:
            add("deployment.latency", "Is there an inference latency budget (ms)?", "nice_to_have",
                "Impacts model size/deployment design.", "Provide target latency or rough bound.")

    if stage == "validation":
        val = spec_dict.get("validation", {}) or {}
        if not val.get("primary_metrics"):
            add("validation.metrics", "Which primary metrics should we optimize/report (MAE/RMSE/MAPE/CRPS)?", "important",
                "Metrics define success and tuning.", "Pick 1–3 metrics.")
        if not val.get("acceptance_criteria"):
            add("validation.acceptance", "What acceptance criteria defines success (e.g., MAE < X at horizon H)?", "important",
                "Defines go/no-go decision.", "Provide tolerance per horizon/unit.")
        if not val.get("robustness_tests"):
            add("validation.robustness", "Which robustness tests matter (missing, drift, noise, extremes)?", "nice_to_have",
                "Avoids deployment failure modes.", "List 2–5 critical scenarios.")

    return gaps
