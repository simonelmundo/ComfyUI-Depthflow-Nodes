import logging
import moderngl
import os

# GPU Acceleration Environment Variables
gpu_env_vars = {
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "all",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "PYOPENGL_PLATFORM": "egl",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu",
    "LIBGL_DRIVERS_PATH": "/usr/lib/x86_64-linux-gnu/dri",
    "CUDA_VISIBLE_DEVICES": "0",
    "WINDOW_BACKEND": "headless"
}

# Set environment variables if not already set
for key, value in gpu_env_vars.items():
    if key not in os.environ:
        os.environ[key] = value

# Rest of your imports
import torch
from DepthFlow import DepthScene
from Broken.Loaders import LoaderImage
from ShaderFlow.Texture import ShaderTexture
import numpy as np
from collections import deque
from comfy.utils import ProgressBar
import gc
import subprocess
import sys
import re
import os
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


def check_compatibility():
    try:
        import pkg_resources
        
        required_packages = {
            'torch': '>=2.0.0',
            'moderngl': '>=5.6.0',
            'numpy': '>=1.19.0',
            'shaderflow': '>=0.1.0'
        }
        
        for package, version in required_packages.items():
            try:
                pkg_resources.require(f"{package}{version}")
                logger.debug(f"✅ {package} version check passed")
            except pkg_resources.VersionConflict as e:
                logger.warning(f"⚠️ Version conflict for {package}: {e}")
            except pkg_resources.DistributionNotFound:
                logger.error(f"❌ {package} not found")
    except Exception as e:
        logger.error(f"Failed to check package compatibility: {e}")

# Call compatibility check
check_compatibility()
def create_gl_context():
    backends = ['egl', 'osmesa']
    errors = {}
    
    for backend in backends:
        try:
            os.environ["PYOPENGL_PLATFORM"] = backend
            ctx = moderngl.create_standalone_context(backend=backend)
            logger.info(f"Successfully created {backend.upper()} context")
            logger.info(f"OpenGL Version: {ctx.version_code}")
            logger.info(f"Vendor: {ctx.vendor}")
            logger.info(f"Renderer: {ctx.renderer}")
            return ctx, backend
        except Exception as e:
            errors[backend] = str(e)
            continue
    
    error_msg = "\n".join([f"{b.upper()}: {e}" for b, e in errors.items()])
    raise RuntimeError(f"Failed to create OpenGL context with any backend:\n{error_msg}")

# Parse requirements.txt to extract depthflow version
def extract_depthflow_version():
    # Get the directory of the current script (depthflow.py)
    script_dir = os.path.dirname(__file__)
    # Go up one directory to reach the project root (myapp/)
    project_root = os.path.dirname(script_dir)
    # Construct the absolute path to requirements.txt
    requirements_file = os.path.join(project_root, 'requirements.txt')
    with open(requirements_file, 'r') as f:
        for line in f:
            match = re.match(r'^depthflow==(.*)$', line.strip())
            if match:
                return match.group(1)
    return None  # Return None if depthflow not found

# Extract expected depthflow version from requirements.txt
expected_version = extract_depthflow_version()
if expected_version is None:
    print("Warning: depthflow not found in requirements.txt. Cannot proceed.")
    sys.exit(1)  # Exit if depthflow version not found
    
version = importlib_metadata.version("depthflow")

if expected_version != version: 
    print(f"Depthflow version {version} does not match expected version {expected_version}")
    subprocess.run([sys.executable, "-m", "pip", "install", f"depthflow=={expected_version}"])



# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('depthflow')

class CustomDepthflowScene(DepthScene):
    def __init__(
        self,
        state=None,
        effects=None,
        progress_callback=None,
        num_frames=30,
        input_fps=30.0,
        output_fps=30.0,
        animation_speed=1.0,
        **kwargs,
    ):
        logger.debug("Environment variables:")
        for var in gpu_env_vars.keys():
            logger.debug(f"{var}: {os.environ.get(var, 'Not set')}")

        try:
            # Create OpenGL context using our helper
            ctx, backend = create_gl_context()
            logger.debug(f"Successfully created {backend.upper()} context")
            ctx.release()
        except Exception as e:
            logger.error(f"Failed to create OpenGL context: {e}")
            raise

        # Initialize scene with successful backend
        scene_kwargs = {
            "backend": os.environ["PYOPENGL_PLATFORM"],
            "state": state,
            "effects": effects
        }
        
        try:
            super().__init__(**scene_kwargs)
            logger.debug("DepthScene initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DepthScene: {e}")
            logger.exception("Full traceback:")
            raise

    
        # Rest of your initialization code...
        self.frames = deque()
        self.progress_callback = progress_callback
        self.custom_animation_frames = deque()
        self._set_effects(effects)
        self.override_state = state
        self.time = 0.00001
        self.images = None
        self.depth_maps = None
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.animation_speed = animation_speed
        self.num_frames = num_frames
        self.video_time = 0.0
        self.frame_index = 0
       
    # TODO: This is a temporary fix to while build gets fixed
    def build(self):
        self.image = ShaderTexture(scene=self, name="image").repeat(False)
        self.depth = ShaderTexture(scene=self, name="depth").repeat(False)
        self.normal = ShaderTexture(scene=self, name="normal")
        self.shader.fragment = self.DEPTH_SHADER
        self.ssaa = 1.2
        

    def input(self, image, depth):
        # Store the images and depth maps
        self.images = image  # Should be numpy arrays of shape [num_frames, H, W, C]
        self.depth_maps = depth
        # For initial setup, use the first frame
        initial_image = image[0]
        initial_depth = depth[0]
        DepthScene.input(self, initial_image, initial_depth)
        

    def setup(self):
        DepthScene.setup(self)
        self.time += 0.00001  # prevent division by zero error

    def _set_effects(self, effects):
        if effects is None:
            self.effects = None
            return
        # If effects is a list or deque, convert it to a deque
        if isinstance(effects, (list, deque)):
            self.effects = deque(effects)
        else:
            self.effects = effects

    def custom_animation(self, motion):
        # check if motion is a list, otherwise add it directly with add_animation
        if isinstance(motion, list):
            for m in motion:
                self.custom_animation_frames.append(m)
        else:
            self.add_animation(motion)

    def update(self):
        frame_duration = 1.0 / self.input_fps

        while self.time > self.video_time:
            self.video_time += frame_duration
            self.frame_index += 1

        # Set the current image and depth map based on self.frame
        if self.images is not None and self.depth_maps is not None:
            frame_index = min(self.frame_index, len(self.images) - 1)
            current_image = self.images[frame_index]
            current_depth = self.depth_maps[frame_index]

            # Convert to appropriate format if necessary
            image = self.upscaler.upscale(LoaderImage(current_image))
            depth = LoaderImage(current_depth)

            # Set the current image and depth map
            self.image.from_image(image)
            self.depth.from_image(depth)

        # If there are custom animation frames present, use them instead of the normal animation frames
        if self.custom_animation_frames:
            self.animation = [self.custom_animation_frames.popleft()]
            
        DepthScene.update(self)
        
        def set_effects_helper(effects):
            for key, value in effects.items():
                if key in self.state.__fields__:
                    setattr(self.state, key, value)

        if self.effects:
            if isinstance(self.effects, deque):
                set_effects_helper(self.effects.popleft())
            else:
                set_effects_helper(self.effects)

        if self.override_state:
            for key, value in self.override_state.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)

            if "tiling_mode" in self.override_state:
                if self.override_state["tiling_mode"] == "repeat":
                    self.image.repeat(True)
                    self.depth.repeat(True)
                    self.state.mirror = False
                elif self.override_state["tiling_mode"] == "mirror":
                    self.image.repeat(False)
                    self.depth.repeat(False)
                    self.state.mirror = True
                else:
                    self.image.repeat(False)
                    self.depth.repeat(False)
                    self.state.mirror = False

    @property
    def tau(self) -> float:
        return super().tau * self.animation_speed

    def next(self, dt):
        DepthScene.next(self, dt)
        width, height = self.resolution
        array = np.frombuffer(self._final.texture.fbo().read(), dtype=np.uint8).reshape(
            (height, width, 3)
        )
        
        array = np.flip(array, axis=0).copy()

        # To Tensor
        tensor = torch.from_numpy(array)

        del array

        # Accumulate the frame
        self.frames.append(tensor)

        if self.progress_callback:
            self.progress_callback()

        return self

    def get_accumulated_frames(self):
        # Convert the deque of frames to a tensor
        return torch.stack(list(self.frames))

    def clear_frames(self):
        self.frames.clear()
        gc.collect()


class Depthflow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image
                "depth_map": ("IMAGE",),  # Depthmap input
                "motion": ("DEPTHFLOW_MOTION",),  # Motion object
                "animation_speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "step": 0.01},
                ),
                "input_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "step": 1.0}),
                "output_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "step": 1.0}),
                "num_frames": ("INT", {"default": 30, "min": 1, "step": 1}),
                "quality": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "ssaa": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "invert": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "tiling_mode": (["mirror", "repeat", "none"], {"default": "mirror"}),
            },
            "optional": {
                "effects": ("DEPTHFLOW_EFFECTS",),  # DepthState object
            },
        }

    RETURN_TYPES = (
        "IMAGE",
    )  # Output is a batch of images (torch.Tensor with shape [B,H,W,C])
    FUNCTION = "apply_depthflow"
    CATEGORY = "🌊 Depthflow"
    DESCRIPTION = """
    Depthflow Node:
    This node applies a motion animation (Zoom, Dolly, Circle, Horizontal, Vertical) to an image
    using a depthmap and outputs an image batch as a tensor.
    - image: The input image.
    - depth_map: Depthmap corresponding to the image.
    - options: DepthState object.
    - motion: Depthflow motion object.
    - input_fps: Frames per second for the input video.
    - output_fps: Frames per second for the output video.
    - num_frames: Number of frames for the output video.
    - quality: Quality of the output video.
    - ssaa: Super sampling anti-aliasing samples.
    - invert: Invert the depthmap.
    - tiling_mode: Tiling mode for the image.
    """

    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None
    def verify_cuda_setup(self):
        try:
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                logger.warning("CUDA is not available. GPU acceleration will be disabled.")
                return False
            
            logger.info(f"CUDA Available: {cuda_available}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Current Device: {torch.cuda.current_device()}")
            logger.info(f"Device Name: {torch.cuda.get_device_name()}")
            return True
        except Exception as e:
            logger.error(f"Error checking CUDA setup: {e}")
            return False

    # ... rest of your existing code ...
    def apply_depthflow(
        self,
        image,
        depth_map,
        motion,
        animation_speed,
        input_fps,
        output_fps,
        num_frames,
        quality,
        ssaa,
        invert,
        tiling_mode,
        effects=None,
    ):
        logger.debug("BRUH  Starting apply_depthflow")
        logger.debug(f"BRUH Input parameters: fps={input_fps}, frames={num_frames}, quality={quality}")
        
        state = {"invert": invert, "tiling_mode": tiling_mode}
        logger.debug(f"BRUH Creating scene with state: {state}")
        
        try:
            scene = CustomDepthflowScene(
                state=state,
                effects=effects,
                progress_callback=self.update_progress,
                num_frames=num_frames,
                input_fps=input_fps,
                output_fps=output_fps,
                animation_speed=animation_speed,
                backend="headless"
            )
            logger.debug("BRUH  Scene created successfully")
        except Exception as e:
            logger.debug(f"BRUH  Failed to create scene: {e}")
            logger.exception("Full traceback:")
            raise
        if image.is_cuda:
            image = image.cpu().numpy()
        else:
            image = image.numpy()
        if depth_map.is_cuda:
            depth_map = depth_map.cpu().numpy()
        else:
            depth_map = depth_map.numpy()

        # Ensure the arrays have the correct shape and data type
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        elif image.ndim != 4:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if depth_map.ndim == 3:
            depth_map = np.expand_dims(depth_map, axis=0)
        elif depth_map.ndim != 4:
            raise ValueError(f"Unsupported depth_map shape: {depth_map.shape}")

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if depth_map.dtype != np.uint8:
            depth_map = (depth_map * 255).astype(np.uint8)

        # Determine the number of frames
        num_image_frames = image.shape[0]
        num_depth_frames = depth_map.shape[0]
        num_render_frames = max(num_frames, num_image_frames, num_depth_frames)

        # Expand images and depth maps to match num_render_frames
        def expand_frames(array, num_frames):
            if array.shape[0] == num_frames:
                return array
            elif array.shape[0] == 1:
                return np.broadcast_to(array, (num_frames,) + array.shape[1:])
            else:
                raise ValueError(
                    f"Cannot expand array with shape {array.shape} to {num_frames} frames"
                )

        image = expand_frames(image, num_render_frames)
        depth_map = expand_frames(depth_map, num_render_frames)

        # Get width and height of images
        height, width = image.shape[1], image.shape[2]

        # Input the image and depthmap into the scene
        scene.input(image, depth=depth_map)

        scene.custom_animation(motion)

        # Calculate the duration based on fps and num_frames
        if num_frames <= 0:
            raise ValueError("FPS and number of frames must be greater than 0")
        duration = float(num_frames) / input_fps
        total_frames = duration * output_fps

        self.start_progress(total_frames, desc="Depthflow Rendering")

        # Render the output video
        scene.main(
            render=False,
            output=None,
            fps=output_fps,
            time=duration,
            speed=1.0,
            quality=quality,
            ssaa=ssaa,
            scale=1.0,
            width=width,
            height=height,
            ratio=None,
            freewheel=True,
        )

        video = scene.get_accumulated_frames()
        scene.clear_frames()
        self.end_progress()

        # Normalize the video frames to [0, 1]
        video = video.float() / 255.0

        return (video,)
