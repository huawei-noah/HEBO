# sns.color_palette("tab10")
from typing import Optional, List, Union, Tuple, NewType

import seaborn as sns

Color = NewType('Color', Union[str, Tuple[float], float])


COLORS_SNS_10 = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
]

# Colorblind
COLORS = sns.color_palette("bright") + sns.color_palette("colorblind")
MARKERS = ['o', 'v', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '^', '<', '>']


def get_color(ind: int, color_palette: Optional[List[Color]] = None, force_color: Optional[Color] = None) -> \
        Color:
    if force_color is not None:
        return force_color
    if color_palette is None:
        color_palette = COLORS
    return color_palette[ind % len(color_palette)]


def get_marker(ind: int, marker_choices: Optional[List[str]] = None, force_marker: Optional[str] = None) -> str:
    if force_marker is not None:
        return force_marker
    if marker_choices is None:
        marker_choices = MARKERS
    return marker_choices[ind % len(marker_choices)]
