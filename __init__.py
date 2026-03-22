from .pst_node import PhaseStretchTransformNode

NODE_CLASS_MAPPINGS = {
    "PhaseStretchTransform": PhaseStretchTransformNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhaseStretchTransform": "Phase Stretch Transform",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
