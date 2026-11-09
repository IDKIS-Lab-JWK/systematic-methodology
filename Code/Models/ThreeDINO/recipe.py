"""CV recipe projection; execution parameters remain intact in run/checkpoint guards."""
from copy import deepcopy


def recipe_parameters(method, parameters):
    result = deepcopy(dict(parameters))
    # Scope the migration to LoRA; other active local workflows retain their contract.
    if method == "lora":
        context = result.pop("_execution_context", {})
        if "seed" in context:
            if "seed" in result and result["seed"] != context["seed"]:
                raise ValueError("Recipe and execution seeds differ")
            result["seed"] = context["seed"]
    return result
