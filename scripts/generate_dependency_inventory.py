import importlib.metadata
import json
from pathlib import Path

rows=[]
for dist in importlib.metadata.distributions():
 rows.append({"name":dist.metadata.get("Name"),"version":dist.version,"license":dist.metadata.get("License")})
Path("dependency_inventory.json").write_text(json.dumps(sorted(rows,key=lambda x:(x["name"] or "").lower()),indent=2),encoding="utf-8")
