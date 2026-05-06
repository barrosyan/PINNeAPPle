from pinneaple_models.register_all import register_all
from pinneaple_models.registry import ModelRegistry

register_all()

names = ModelRegistry.list()
print("Total models:", len(names))
for n in names:
    print(" -", n)

pinns_models = ModelRegistry.list(family="pinns")
print("Total models:", len(pinns_models))
for n in pinns_models:
    print(" -", n)

