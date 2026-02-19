from __future__ import annotations

import argparse
import json

from pinneaple_researcher.models import RankedItem
from pinneaple_researcher.pipelines.reproduce import reproduce

run_dir=r"runs\researcher\pinn_thermal_instability_virtual_sensor\20260219_120449"
type="paper"
id="1105.3778v1"
title="Oscillatory thermal instability - the Bhopal disaster and liquid bombs"
url="https://arxiv.org/abs/1105.3778v1"

item = RankedItem(type=type, id=ids, title=title, url=url, score=1.0, why=[], meta={})
project_dir = reproduce(item=item, kb_index_dir=run_dir)
print(project_dir)
