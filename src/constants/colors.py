from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ColorPaletteEntry:
    label: str
    hex: str
    rgb: tuple[int, int, int]


COLOR_PALETTE: tuple[ColorPaletteEntry, ...] = (
    ColorPaletteEntry("black", "#202020", (32, 32, 32)),
    ColorPaletteEntry("white", "#f2f2ee", (242, 242, 238)),
    ColorPaletteEntry("cream", "#eadfc4", (234, 223, 196)),
    ColorPaletteEntry("gray", "#8a8a86", (138, 138, 134)),
    ColorPaletteEntry("red", "#b73432", (183, 52, 50)),
    ColorPaletteEntry("pink", "#e87aa6", (232, 122, 166)),
    ColorPaletteEntry("orange", "#d97931", (217, 121, 49)),
    ColorPaletteEntry("yellow", "#d9c04f", (217, 192, 79)),
    ColorPaletteEntry("green", "#5f9a55", (95, 154, 85)),
    ColorPaletteEntry("olive", "#708238", (112, 130, 56)),
    ColorPaletteEntry("teal", "#49a6a4", (73, 166, 164)),
    ColorPaletteEntry("blue", "#557fc1", (85, 127, 193)),
    ColorPaletteEntry("navy", "#263f78", (38, 63, 120)),
    ColorPaletteEntry("purple", "#8b63b7", (139, 99, 183)),
    ColorPaletteEntry("brown", "#7a5135", (122, 81, 53)),
    ColorPaletteEntry("beige", "#c8b58e", (200, 181, 142)),
    ColorPaletteEntry("gold", "#c99732", (201, 151, 50)),
    ColorPaletteEntry("silver", "#b9bec4", (185, 190, 196)),
)

MULTICOLOR_LABEL = "multicolor"
