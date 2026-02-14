# scripts/cosmos — Cosmos (LLM / World Foundation Model) integration

Esses scripts assumem que você tem um endpoint HTTP para Cosmos (ou gateway interno)
que aceita JSON e retorna JSON.

## Interface esperada do endpoint

### 1) Scene-to-spec
POST {COSMOS_BASE_URL}/scene_to_spec
Request JSON:
{
  "prompt_yaml": "<conteúdo do configs/problemdesign/flow_obstacle_2d_prompt.yaml>",
  "schema_yaml": "<conteúdo do schema>",
  "context": {...}  # opcional
}

Response JSON:
{
  "spec_yaml": "<yaml do spec>",
  "meta": {...}
}

### 2) Generate variations
POST {COSMOS_BASE_URL}/generate_variations
Request JSON:
{
  "spec_yaml": "<yaml>",
  "n": 25,
  "strategy": "grid" | "random"
}

Response JSON:
{
  "variations": [
    {"nu": 0.01, "obstacle_circle": {"cx":..., "cy":..., "r":...}, "id":"..."},
    ...
  ]
}

### 3) Evaluate-and-refine
POST {COSMOS_BASE_URL}/evaluate_and_refine
Request JSON:
{
  "spec_yaml": "<yaml>",
  "metrics": {...},
  "history": [...]
}

Response JSON:
{
  "recommendations": {...},
  "updated_spec_yaml": "<optional>"
}

## Se você NÃO tiver endpoint agora
Você pode pular a pasta cosmos e usar:
- configs/problemdesign/flow_obstacle_2d_spec_example.yaml
