# scripts/mvp — scripts locais “cola-tudo” do MVP

Aqui ficam scripts para:
- gerar/copiar JSONs do bundle (conditions/manifest)
- validar bundle contra schema + presença de arquivos
- rodar uma matriz de runs da Arena (native + physicsnemo)

Eles rodam fora do Omniverse.

## Uso recomendado
1) Se você ainda não tem `conditions.json` e `manifest.json` no bundle:
   python scripts/mvp/build_bundle_jsons.py

2) Validar o bundle:
   python scripts/mvp/validate_bundle.py

3) Rodar todos os runs (matriz):
   python scripts/mvp/run_arena_matrix.py
