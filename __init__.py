import sys
from pathlib import Path

here = Path(__file__).parent.resolve()
#  sys.path.append(str(Path(here, "nodes")))
import traceback
import importlib


def load_nodes():
    errors = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    for filename in (here / "nodes").iterdir():
        if filename.suffix != ".py":
            continue
        module_name = filename.stem
        try:
            module = importlib.import_module(
                f".nodes.{module_name}", package=__package__
            )
            node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                node_display_name_mappings.update(
                    getattr(module, "NODE_DISPLAY_NAME_MAPPINGS")
                )

        except AttributeError:
            pass  # wip nodes
        except Exception:
            error_message = traceback.format_exc().splitlines()[-1]
            errors.append(
                f"Failed to import module {module_name} because {error_message}"
            )

    if len(errors) > 0:
        print(
            "Some nodes failed to load:\n\t"
            + "\n\t".join(errors)
            + "\n\n"
            + "Check that you properly installed the dependencies.\n"
            + "If you think this is a bug, please report it on the github page (https://github.com/Fannovel16/comfyui_controlnet_aux/issues)"
        )
    return node_class_mappings, node_display_name_mappings


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes()
print("ComfyUI-Bedrock loaded:\n  ", list(NODE_CLASS_MAPPINGS.keys()))
