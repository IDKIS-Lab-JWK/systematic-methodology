"""Project DenseNet recipe identity without changing strict per-run resume identity."""
from copy import deepcopy
import hashlib
import json


def recipe_identity(identity):
    result = deepcopy(dict(identity))
    if result.get("model") != "DenseNet":
        return result
    model = result["model_identity"]
    runtime = model.get("pretrained_model", {}).get("runtime", {})
    for key in ("python", "platform", "cuda_device_name", "cuda_device_capability", "device_requested"):
        runtime.pop(key, None)
    implementation = model.get("implementation", {})
    records = implementation.get("files", [])
    for record in records:
        path = record["path"].replace("\\", "/")
        if "/Code/" in path:
            path = "Code/" + path.split("/Code/", 1)[1]
        elif not path.startswith("Code/"):
            raise ValueError("Source identity is not project-relative: " + path)
        record["path"] = path
    if records:
        unique = {}
        for item in records:
            if item["path"] in unique and unique[item["path"]] != item:
                raise ValueError("Conflicting recipe source paths")
            unique[item["path"]] = item
        records = sorted(unique.values(), key=lambda item: item["path"])
        implementation["files"] = records
        implementation["sha256"] = hashlib.sha256(json.dumps(
            records, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest()
    return result
