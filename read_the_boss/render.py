"""Deterministic, asset-free arena visualization of an actual environment state."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 720
BG = '#101827'
PANEL = '#1c293b'
INK = '#edf4fc'
MUTED = '#a6b7cd'
BLUE = '#65c5ff'
ORANGE = '#ffb36b'
GREEN = '#89dfba'
INTENTS = ('SLASH', 'SWEEP', 'LUNGE', 'RECOVERY')
DISTANCES = ('CLOSE', 'MEDIUM', 'FAR')
ACTIONS = ('Approach', 'Retreat', 'Strike', 'Guard', 'Rest')


def font(size: int, bold: bool = False, mono: bool = False):
    name = 'DejaVuSansMono' if mono else 'DejaVuSans'
    if bold:
        name += '-Bold'
    candidates = (Path('/usr/share/fonts/truetype/dejavu') / (name + '.ttf'),
                  Path(name + '.ttf'))
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _text(draw, xy, text, size=20, fill=INK, bold=False, anchor=None, mono=False):
    draw.text(xy, text, font=font(size, bold, mono), fill=fill, anchor=anchor)


def _bar(draw, x, y, width, value, maximum, color):
    draw.rounded_rectangle((x, y, x + width, y + 14), radius=7, fill='#34445b')
    if value > 0:
        draw.rounded_rectangle((x, y, x + width * value / maximum, y + 14),
                               radius=7, fill=color)


def _fighter(draw, x, floor, color, facing=1, boss=False, defeated=False):
    """Simple vector characters; their separation encodes the distance band."""
    if defeated:
        draw.ellipse((x - 55, floor - 13, x + 55, floor + 7), fill='#34445b')
        draw.line((x - 42, floor - 20, x + 42, floor - 20), fill=color, width=20)
        return
    scale = 1.12 if boss else 1.0
    h = int(120 * scale)
    draw.ellipse((x - 58, floor - 9, x + 58, floor + 12), fill='#111b2c')
    draw.line((x - 13, floor - 62, x - 31, floor), fill=color, width=14)
    draw.line((x + 13, floor - 62, x + 31, floor), fill=color, width=14)
    draw.rounded_rectangle((x - 28, floor - h, x + 28, floor - 55),
                           radius=12, fill=color)
    draw.ellipse((x - 22, floor - h - 48, x + 22, floor - h - 4), fill=color)
    draw.line((x + facing * 16, floor - h + 22,
               x + facing * 61, floor - h + 44), fill=color, width=13)
    draw.line((x + facing * 57, floor - h + 48,
               x + facing * 83, floor - h - 20), fill=INK, width=7)
    draw.line((x + facing * 44, floor - h + 25,
               x + facing * 75, floor - h + 36), fill=ORANGE, width=5)
    draw.ellipse((x - facing * 62, floor - h + 18,
                  x - facing * 33 if facing == 1 else x + 73,
                  floor - h + 76), fill='#41576f') if facing == 1 else None


def render_frame(state, info, *, opponent='balanced', reward_mode='full'):
    """Return an H x W x 3 uint8 frame. No environment mutation or random draw."""
    hp, boss_hp, stamina, distance, intent, remaining = map(int, state)
    image = Image.new('RGB', (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(image)
    _text(d, (32, 22), 'READ THE BOSS', 30, bold=True)
    _text(d, (1168, 30), 'ENVIRONMENT SNAPSHOT', 15, fill=MUTED, anchor='ra')
    _text(d, (32, 62), f'Controller: {opponent.replace("_", " ")}  |  Reward: {reward_mode}',
          17, fill=MUTED)
    d.rounded_rectangle((30, 100, 1170, 235), radius=16, fill=PANEL)
    _text(d, (52, 118), f'FIGHTER  {hp}/6 HP', 21, fill=BLUE, bold=True)
    _bar(d, 52, 153, 300, hp, 6, BLUE)
    _text(d, (52, 187), 'STAMINA', 16, fill=MUTED)
    for i in range(4):
        x = 164 + i * 35
        d.rounded_rectangle((x, 185, x + 25, 210), radius=5,
                            fill=GREEN if i < stamina else '#34445b')
    _text(d, (1148, 118), f'BOSS  {boss_hp}/8 HP', 21, fill=ORANGE, bold=True, anchor='ra')
    _bar(d, 848, 153, 300, boss_hp, 8, ORANGE)
    _text(d, (1148, 187), f'{remaining} / 40 TURNS LEFT', 17, fill=MUTED, anchor='ra')
    _text(d, (600, 117), 'ANNOUNCED INTENT', 15, fill=MUTED, anchor='ma')
    cue_color = GREEN if intent == 3 else ORANGE
    _text(d, (600, 146), INTENTS[intent], 31, fill=cue_color, bold=True, anchor='ma')
    cue = ('2 damage at close range', '3 damage at close / medium',
           'Advance, then 1 if close', 'No attack this turn')[intent]
    _text(d, (600, 193), cue, 16, fill=MUTED, anchor='ma')

    # This is a schematic arena: only distance bands, not continuous physics.
    floor = 465
    d.rounded_rectangle((30, 253, 1170, 514), radius=16, fill='#172235')
    d.line((70, floor + 10, 1130, floor + 10), fill='#40516b', width=3)
    player_x = 290
    boss_x = (475, 700, 940)[distance]
    _fighter(d, player_x, floor, BLUE, defeated=hp == 0)
    _fighter(d, boss_x, floor, ORANGE, facing=-1, boss=True, defeated=boss_hp == 0)
    _text(d, (player_x, 480), 'FIGHTER', 16, fill=BLUE, anchor='ma', bold=True)
    _text(d, (boss_x, 480), 'BOSS', 16, fill=ORANGE, anchor='ma', bold=True)
    _text(d, (600, 267), 'DISTANCE: ' + DISTANCES[distance], 18,
          fill=MUTED, anchor='ma', bold=True)
    outcome = info.get('outcome', 'ongoing')
    if outcome != 'ongoing':
        d.rounded_rectangle((440, 325, 760, 390), radius=12, fill=BG, outline=GREEN, width=2)
        _text(d, (600, 339), outcome.upper(), 28, fill=GREEN, bold=True, anchor='ma')

    mask = info.get('action_mask', [0, 0, 0, 0, 0])
    for idx, name in enumerate(ACTIONS):
        x = 30 + idx * 230
        active = bool(mask[idx])
        d.rounded_rectangle((x, 531, x + 218, 586), radius=9,
                            fill=PANEL if active else '#151d2b',
                            outline=BLUE if active else '#303c4e', width=2)
        _text(d, (x + 109, 546), f'{idx}  {name}', 19,
              fill=INK if active else '#63748d', anchor='ma', bold=active)
    _text(d, (32, 604), 'Bright actions are legal; dim actions are masked.', 16, fill=MUTED)
    events = info.get('events', [])
    event_text = ' | '.join(events) if events else 'Reset complete. Choose a legal action.'
    while d.textlength(event_text, font=font(16)) > 1130 and len(event_text) > 3:
        event_text = event_text[:-4].rstrip() + '...'
    _text(d, (32, 634), event_text, 16)
    _text(d, (32, 680), f'OBS  [{hp}, {boss_hp}, {stamina}, {distance}, {intent}, {remaining}]',
          16, fill=MUTED, mono=True)
    return np.asarray(image, dtype=np.uint8).copy()
