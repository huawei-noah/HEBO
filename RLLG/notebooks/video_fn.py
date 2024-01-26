import numpy as np
import cv2
import time
from io import BytesIO
import PIL
from IPython import display
from dm_control.utils import rewards as rewards_fn
from dmc2gym.wrappers import _flatten_obs
from typing import Union, Any, Dict, List, Optional, Tuple


def grabFrame(env: Any) -> np.ndarray:
    """
    Capture and return a frame from the dm_control environment rendering.

    Parameters:
    ----------
    env : dm_control suite env
        The dm control suite environment

    Returns:
    ----------
    np.ndarray
        A NumPy array representing the RGB frame captured from the environment rendering

    """
    # Get RGB ren
    # Get RGB rendering of env
    rgbArr = env.physics.render(480, 600, camera_id=0)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)


# Use 'jpeg' instead of 'png' (~5 times faster)
def array_to_image(a: np.ndarray, fmt: Optional[str] = 'jpeg') -> display.Image:
    """
    Convert a NumPy array to an image and display it using IPython's display module.

    Parameters:
    ----------
    a : numpy.ndarray
        The input NumPy array representing an image
    fmt : str, optional
        The image format to use (default is 'jpeg')

    Returns:
    ----------
    IPython.display.Image
        An IPython Image object representing the displayed image

    """
    # Create binary stream object
    f = BytesIO()

    # Convert array to binary stream object
    PIL.Image.fromarray(a).save(f, fmt)

    return display.Image(data=f.getvalue())

def create_dm_video(env: Any,
                    policy: str = "random",
                    verbose: int = 0,
                    video_name: str = "video.mp4",
                    not_plot: bool = False) -> None:
    """
    Create a video of an episode in a dm_control environment.

    Parameters:
    ----------
    env : Any
        The dm_control environment
    policy : str or callable, optional
        The policy used to generate actions. If "random", random actions are used.
        If a callable, it should take an observation and return an action.
    verbose : int, optional
        Verbosity level. If greater than 0, print additional information during video creation.
    video_name : str, optional
        The name of the output video file (default is "video.mp4").
    not_plot : bool, optional
        If True, do not plot (default is False).

    Returns:
    ----------
    None
    """
    frame = grabFrame(env)
    height, width, layers = frame.shape
    if not not_plot:
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))

    # First pass - Step through an episode and capture each frame
    action_spec = env.action_spec()
    rewards = []
    time_step = env.reset()
    while not time_step.last():
        if policy == "random":
            action = np.random.uniform(action_spec.minimum,
                                       action_spec.maximum,
                                       size=action_spec.shape)
        else:
            action = policy(
                np.expand_dims(_flatten_obs(time_step.observation), 0), deterministic=True
            )[0, :]
        time_step = env.step(action)
        
        if verbose:
            radii = np.array([
                0.2,
                env.physics.named.model.geom_size[['finger'], 0].item()]
            ).sum()
            inside_big_goal = rewards_fn.tolerance(env.physics.finger_to_target_dist(), (0, radii))
            
            print('action:', action)
            print('reward:', float(time_step.reward))
            print('distance to easy target:', np.array([
            0.2,
            env.physics.named.model.geom_size[['finger'], 0].item()]
        ).sum())
            print('env.env.finger_to_target_dist():', env.physics.finger_to_target_dist())
            print('lambda s:', inside_big_goal)
            print()
        rewards.append(float(time_step.reward))
        if not not_plot:
            frame = grabFrame(env)
            # Render env output to video
            video.write(grabFrame(env))
        
    if verbose:
        print('rewards:', np.sum(rewards))

    # End render to video file
    if not not_plot:
        video.release()


def plot_video(d: Dict, d2: Dict, video_name: str = "video.mp4") -> None:
    """
    Plot a video usind d and d2 for display.

    Parameters:
    ----------
    d : Any
        The dictionary used to update the video frames.
    d2 : Any
        The dictionary used to update the display with additional information (e.g., FPS).
    video_name : str, optional
        The name of the input video file (default is "video.mp4").

    Returns:
    ----------
    None
    """
    cap = cv2.VideoCapture(video_name)
    while (cap.isOpened()):
        t1 = time.time()
        ret, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = array_to_image(frame)
        d.update(im)
        t2 = time.time()
        s = f"""{int(1 / (t2 - t1))} FPS"""
        d2.update(display.HTML(s))
    cap.release()


def plot_total_video(env: Any,
                     d: dict,
                     d2: dict,
                     policy: str = "random",
                     verbose: int = 0,
                     video_name: str = "video.mp4",
                     not_plot: bool = False) -> None:
    """
    Plot a total video using dictionaries for display.

    Parameters:
    ----------
    env : type
        Description of parameter `env`.
    d : dict
        The dictionary used to update the video frames.
    d2 : dict
        The dictionary used to update the display with additional information (e.g., FPS)
    policy : str, optional
        The policy to be used (default is "random")
    verbose : int, optional
        Verbosity level (default is 0).
    video_name : str, optional
        The name of the output video file (default is "video.mp4")
    not_plot : bool, optional
        If True, do not plot the video (default is False)

    Returns:
    ----------
    None

    """
    create_dm_video(env, policy=policy, verbose=verbose, video_name=video_name, not_plot=not_plot)
    if not not_plot:
        plot_video(d, d2, video_name=video_name)
