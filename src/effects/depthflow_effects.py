from abc import abstractmethod
from typing import List, Optional
from ..base_flex import BaseFlex
from pydantic import BaseModel, Field

    
class DepthflowEffects(BaseFlex):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
            },
            "optional": {
                **super().INPUT_TYPES()["optional"],
                "effects": ("DEPTHFLOW_EFFECTS",)
            }
        }

    RETURN_TYPES = ("DEPTHFLOW_EFFECTS",)
    CATEGORY = "🌊 Depthflow/Effects"

    @abstractmethod
    def create_internal(self, effects, **kwargs):
        """
        Implemented by subclasses to apply their specific effect.
        """
        pass

    def apply(self, strength, feature_threshold, feature_param, feature_mode, effects=None, feature=None, **kwargs):
        # Determine if effects is a list
        effects_is_list = isinstance(effects, list)
        # Determine if we have a feature to modulate over time
        has_feature = feature is not None

        if not effects_is_list and not has_feature:
            # Case 1: Both effects and feature are not lists
            # Create a single effect by combining effects dict with our effect
            effect = self.create(0.0, strength, feature_param, feature_mode, effects=effects, feature=None, **kwargs)
            return (effect,)
        elif effects_is_list and not has_feature:
            # Case 2: Effects is a list, feature is None
            result = []
            self.start_progress(len(effects), desc=f"Applying {self.__class__.__name__}")
            for i, prev_effect in enumerate(effects):
                kwargs['frame_index'] = i
                effect = self.create(0.0, strength, feature_param, feature_mode, effects=prev_effect, feature=None, **kwargs)
                result.append(effect)
                self.update_progress()
            self.end_progress()
            return (result,)
        elif not effects_is_list and has_feature:
            # Case 3: Effects is a dict, feature is provided
            num_frames = feature.frame_count
            self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")
            result = []
            for i in range(num_frames):
                feature_value = feature.get_value_at_frame(i)
                kwargs['frame_index'] = i
                feature_value = feature_value if feature_value >= feature_threshold else 0.0
                effect = self.create(feature_value, strength, feature_param, feature_mode, effects=effects, feature=feature, **kwargs)
                result.append(effect)
                self.update_progress()
            self.end_progress()
            return (result,)
        elif effects_is_list and has_feature:
            # Case 4: Both effects is a list and feature is provided
            num_frames = feature.frame_count
            if num_frames != len(effects):
                raise ValueError("Number of frames in feature and effects list must be the same")
            self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")
            result = []
            for i in range(num_frames):
                feature_value = feature.get_value_at_frame(i)
                kwargs['frame_index'] = i
                feature_value = feature_value if feature_value >= feature_threshold else 0.0
                effect = self.create(feature_value, strength, feature_param, feature_mode, effects=effects[i], feature=feature, **kwargs)
                result.append(effect)
                self.update_progress()
            self.end_progress()
            return (result,)

    def create(self, feature_value: float, strength: float, feature_param: str, feature_mode: str, effects=None, feature=None, **kwargs):
        # Modulate the selected parameter
        if feature is not None:
            for param_name in self.get_modifiable_params():
                if param_name in kwargs:
                    if param_name == feature_param:
                        kwargs[param_name] = self.modulate_param(param_name, kwargs[param_name],
                                                                 feature_value, strength, feature_mode)

        # Ensure 'effects' is a dict
        if effects is None:
            effects = {}
        elif isinstance(effects, dict):
            effects = effects.copy()  # To avoid modifying input dict
        else:
            raise ValueError("'effects' should be a dict")

        # Call create_internal with effects
        return self.create_internal(effects=effects, **kwargs)[0]


class DepthflowEffectVignette(DepthflowEffects):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "vignette_enable": ("BOOLEAN", {"default": True}),
                "vignette_intensity": ("FLOAT", {"default": 30, "min": 0.0, "max": 100.0, "step": 0.1}),
                "vignette_decay": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    DESCRIPTION = """
    Depthflow Vignette Effect Node:
    This node applies a vignette effect to the depth flow.
    - vignette_enable: Enable the vignette effect.
    - vignette_intensity: Intensity of the vignette effect.
    - vignette_decay: Decay rate of the vignette effect.
    """

    @classmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return ["vignette_intensity", "vignette_decay", "None"]

    def create_internal(self, effects, **kwargs):
        """
        Apply the Vignette effect to the incoming DepthState.
        """
        # Update with Vignette parameters
        effects.update({
            "vignette_enable": kwargs.get("vignette_enable", True),
            "vignette_intensity": kwargs.get("vignette_intensity", 30),
            "vignette_decay": kwargs.get("vignette_decay", 0.1),
        })
        return (effects,)
    

class DepthflowEffectDOF(DepthflowEffects):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "dof_enable": ("BOOLEAN", {"default": True}),
                "dof_start": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dof_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dof_exponent": ("FLOAT", {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "dof_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "dof_quality": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "dof_directions": ("INT", {"default": 16, "min": 1, "max": 32, "step": 1}),
            },
        }

    DESCRIPTION = """
    Depthflow Effects Node:
    This node allows the user to configure the depth flow effects.
    - strength: Strength of the depth flow effect
    - feature_threshold: Minimum feature value to apply the depth flow effect
    - feature_param: Parameter to modulate the depth flow effect
    - feature_mode: Mode to use for the feature parameter
    - dof_enable: Enable a Depth of Field effect
    - dof_start: Start of the Depth of Field effect
    - dof_end: End of the Depth of Field effect
    - dof_exponent: Exponent of the Depth of Field effect
    - dof_intensity: Intensity of the Depth of Field effect
    - dof_quality: Quality of the Depth of Field effect
    - dof_directions: Directions of the Depth of Field effect
    """
    
    @classmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return ["dof_intensity", "dof_start", "dof_end", "dof_exponent", "dof_quality", "dof_directions", "None"]

    def create_internal(self, effects, **kwargs):
        """
        Apply the Depth of Field effect to the incoming DepthState.
        """
        # Update with DOF parameters
        effects.update({
            "dof_enable": kwargs.get("dof_enable", True),
            "dof_start": kwargs.get("dof_start", 0.6),
            "dof_end": kwargs.get("dof_end", 1.0),
            "dof_exponent": kwargs.get("dof_exponent", 2.0),
            "dof_intensity": kwargs.get("dof_intensity", 1.0),
            "dof_quality": kwargs.get("dof_quality", 4),
            "dof_directions": kwargs.get("dof_directions", 16),
        })
        return (effects,)
        
        