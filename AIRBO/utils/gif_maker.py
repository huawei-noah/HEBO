import imageio
import glob
import re
import os


def create_gif(img_dir, img_fname_pattern="*.png", sort_key=lambda x: int(x),
               gif_fname="result.gif", duration=1.0):
    img_fps = glob.glob(os.path.join(img_dir, img_fname_pattern))
    sorted_img_fps = sorted(img_fps, key=sort_key, reverse=False)

    frames = []
    for img_fp in sorted_img_fps:
        frames.append(imageio.imread(img_fp))
        # Save them as frames into a gif
    imageio.mimsave(os.path.join(img_dir, gif_fname), frames, 'GIF', duration=duration)

    return


if __name__ == '__main__':
    create_gif(
        img_dir=r"D:\yanglin\work\15_research\extended_BO\code\tb_logs\20211215_131145_CustomC1-raw_gpy_run9",
        sort_key=lambda x: int(x.split("step")[-1].split('.png')[0]),
        duration=0.7)
    print("GIF generation is done!")
