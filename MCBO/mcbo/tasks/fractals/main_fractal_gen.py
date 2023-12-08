from mcbo.search_space import SearchSpace
from mcbo.tasks.fractals.fractal_utils import t_square_fractal
import matplotlib.pyplot as plt


if __name__ == "__main__":

    CHIP_SIDE = 20
    TRUE_CHIP_SIDE = CHIP_SIDE * 2

    SEARCH_SPACES = \
        {
            'minkowski-v1':
                [
                    {'name': 'a1', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'b1', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'c1', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'a2', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'b2', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'c2', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'offset', 'type': 'int', 'lb': 0, 'ub': TRUE_CHIP_SIDE},
                    {'name': 'fill_center', 'type': 'bool'},
                    {'name': 'negative', 'type': 'bool'},
                    {'name': 'rot', 'type': 'int', 'lb': 0, 'ub': CHIP_SIDE - 1},
                    {'name': 'quarter_rot', 'type': 'int', 'lb': 0, 'ub': 2},
                ]
            ,
            'minkowski-v2':
                [
                    {'name': 'a1', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'b1', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'c1', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'a2', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'b2', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'c2', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'a3', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'b3', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'c3', 'type': 'num', 'lb': 0, 'ub': 1},
                    {'name': 'offset', 'type': 'int', 'lb': 0, 'ub': TRUE_CHIP_SIDE},
                    {'name': 'fill_center', 'type': 'bool'},
                    {'name': 'negative', 'type': 'bool'},
                    {'name': 'rot', 'type': 'int', 'lb': 0, 'ub': CHIP_SIDE - 1},
                    {'name': 'quarter_rot', 'type': 'int', 'lb': 0, 'ub': 2},
                ]
        }
    rec = SearchSpace(SEARCH_SPACES["minkowski-v2"]).sample(10)
    #
    for r in rec.values:
        if len(r) == 11:
            a1, b1, c1, a2, b2, c2, offset, fill_center, negative, rot, quarter_rot = r
            coefs = [(a1, b1, c1), (a2, b2, c2)]
        elif len(r) == 14:
            a1, b1, c1, a2, b2, c2, a3, c3, b3, offset, fill_center, negative, rot, quarter_rot = r
            coefs = [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)]
        else:
            raise ValueError(r)
        offset_ratio = offset / TRUE_CHIP_SIDE
        rot = rot / CHIP_SIDE
        chip = t_square_fractal(side=TRUE_CHIP_SIDE, coefs=coefs, offset_ratio=offset_ratio,
                                fill_center=fill_center, negative=negative, rot=rot,
                                quarter_rot=quarter_rot)
        plt.imshow(chip, cmap=plt.get_cmap("binary"))
        plt.colorbar()
        plt.show()
