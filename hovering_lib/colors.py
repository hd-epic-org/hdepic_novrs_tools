
Colors = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'purple': (128, 0, 128),
    'orange': (255, 165, 0),
    'cyan': (0, 255, 255),
    'pink': (255, 192, 203),
    'gray': (128, 128, 128),
}

def code_to_rgb(clr: str) -> tuple:
    """ Convert color code to RGB tuple 0-255. """
    if isinstance(clr, tuple) and len(clr) == 3:
        return clr
    elif clr in Colors:
        return Colors[clr]
    elif clr.startswith('#'):
        clr = clr[1:]
        return tuple(int(clr[i:i+2], 16) for i in (0, 2, 4))
    else:
        raise ValueError(f"Unknown color code: {clr}")