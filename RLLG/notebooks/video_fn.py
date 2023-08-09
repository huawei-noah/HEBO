import numpy as np
import cv2
import time
from io import BytesIO
import PIL
from IPython import display
from dm_control.utils import rewards as rewards_fn
from dmc2gym.wrappers import _flatten_obs


def grabFrame(env):
    # Get RGB rendering of env
    rgbArr = env.physics.render(480, 600, camera_id=0)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)


# Use 'jpeg' instead of 'png' (~5 times faster)
def array_to_image(a, fmt='jpeg'):
    # Create binary stream object
    f = BytesIO()

    # Convert array to binary stream object
    PIL.Image.fromarray(a).save(f, fmt)

    return display.Image(data=f.getvalue())


def create_dm_video(env, policy="random", verbose=0, video_name="video.mp4", not_plot=False):
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


def plot_video(d, d2, video_name="video.mp4"):
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


def plot_total_video(env, d, d2, policy="random", verbose=0, video_name="video.mp4", not_plot=False):
    """
    Note this function requires d and d2. They must be created with the following in the Jupyter Notebook:
        >>> d = display.display("", display_id=1)
        >>> d2 = display.display("", display_id=2)
    """

    create_dm_video(env, policy=policy, verbose=verbose, video_name=video_name, not_plot=not_plot)
    if not not_plot:
        plot_video(d, d2, video_name=video_name)