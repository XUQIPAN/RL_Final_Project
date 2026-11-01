"""Build the milestone report from measured results and generated evidence figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
INK = colors.HexColor('#15384d')
TEXT = colors.HexColor('#233546')
MUTED = colors.HexColor('#536a7c')
PALE = colors.HexColor('#edf3f7')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author', default='[Your Name]')
    args = parser.parse_args()
    summary = json.loads((EVIDENCE / 'summary.json').read_text())
    smoke = json.loads((EVIDENCE / 'smoke_results.json').read_text())
    trace = json.loads((EVIDENCE / 'demo' / 'trace.json').read_text())
    if summary['tests']['failures'] or summary['tests']['errors']:
        raise SystemExit('Report requires passing recorded tests.')

    family = Path('/usr/share/fonts/truetype/dejavu')
    if (family / 'DejaVuSans.ttf').is_file():
        pdfmetrics.registerFont(TTFont('Body', str(family / 'DejaVuSans.ttf')))
        pdfmetrics.registerFont(TTFont('BodyBold', str(family / 'DejaVuSans-Bold.ttf')))
        pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold',
                                       italic='Body', boldItalic='BodyBold')
        regular, bold = 'Body', 'BodyBold'
    else:
        regular, bold = 'Helvetica', 'Helvetica-Bold'
    styles = {
        'body': ParagraphStyle('body', fontName=regular, fontSize=9.3, leading=13,
                               textColor=TEXT, spaceAfter=5),
        'small': ParagraphStyle('small', fontName=regular, fontSize=8, leading=11,
                                textColor=MUTED, spaceAfter=5),
        'heading': ParagraphStyle('heading', fontName=bold, fontSize=12.5, leading=16,
                                  textColor=INK, spaceBefore=9, spaceAfter=6),
        'title': ParagraphStyle('title', fontName=bold, fontSize=23, leading=28,
                                textColor=INK, spaceAfter=5),
        'eyebrow': ParagraphStyle('eyebrow', fontName=bold, fontSize=8, leading=11,
                                  textColor=MUTED, spaceAfter=4),
        'bullet': ParagraphStyle('bullet', fontName=regular, fontSize=9.3, leading=13,
                                 textColor=TEXT, leftIndent=11, firstLineIndent=-9,
                                 spaceAfter=5),
    }
    p = lambda text, style='body': Paragraph(text, styles[style])
    width = 528

    def picture(filename, cap, max_height=None):
        from PIL import Image as PILImage
        with PILImage.open(EVIDENCE / filename) as im:
            w, h = im.size
        scale = width / w
        if max_height and h * scale > max_height:
            scale = max_height / h
        fig = Image(str(EVIDENCE / filename), width=w * scale, height=h * scale)
        fig.hAlign = 'CENTER'
        return [fig, Spacer(1, 5), p(cap, 'small')]

    def bullet(text):
        return p('- ' + text, 'bullet')

    story = [
        p('MILESTONE 2  /  ENVIRONMENT IMPLEMENTATION', 'eyebrow'),
        p('Read the Boss', 'title'),
        p('Stamina-constrained tactical combat | ' + escape(args.author), 'small'),
        p('1. Project Overview', 'heading'),
        bullet('<b>Idea:</b> A turn-based 1v1 duel tests whether an RL fighter can manage '
               'stamina and transfer its decisions to unfamiliar boss behaviors.'),
        bullet('<b>Implemented:</b> A custom Gymnasium environment, seven scripted boss modes, '
               'two reward variants, legal-action masks, a visual renderer, human terminal play, '
               'and three heuristic policies. Automated tests and reproducible execution evidence are included.'),
        bullet('<b>Remaining:</b> Q-learning/SARSA training, baseline and difficulty calibration, '
               'equal-budget robustness experiments, and final reward analysis. No trained RL policy is claimed here.'),
        p('2. Environment Design', 'heading'),
        p('<b>Environment.</b> One fighter (6 HP, 4 maximum stamina) faces an 8-HP boss at '
          'close, medium, or far distance. Episodes end at victory, death, or the intrinsic 40-turn deadline. '
          'Balanced, aggressive, and distance-aware controllers plus their per-turn mixture are implemented; '
          'three additional controllers are reserved for later policy evaluation.'),
        p('<b>State / observation.</b> The integer vector is '
          '<b>[player HP, boss HP, stamina, distance, intent, turns remaining]</b>. '
          'Reset returns <b>[6, 8, 4, 1, 3, 40]</b>; distance codes 0/1/2 mean close/medium/far, '
          'and intent codes 0/1/2/3 mean slash/sweep/lunge/recovery. The observation includes '
          'the visible next attack and remaining time; no hidden persistent boss identity is used.'),
        p('<b>Actions.</b> Approach moves closer for free; retreat moves away for 1 stamina; '
          'strike costs 2 and deals 2 damage when close; guard costs 1 and blocks up to 2 damage; '
          'rest restores 2 stamina, capped at 4. All supplied policies and the human interface obey action masks.'),
    ]
    story += picture('figure1_environment.png',
        '<b>Figure 1.</b> Actual RGB rendering immediately after <b>reset(seed=7)</b>. '
        'Strike is dimmed because the fighter starts at medium distance. The schematic uses discrete distance bands.',
        max_height=252)

    story += [PageBreak(), p('2. Environment Design (continued)', 'heading'),
        p('<b>Reward.</b> Every step gives <b>-0.1 - 0.02c</b>, where <i>c</i> is stamina spent. '
          'Victory adds <b>100 + 10h/6</b> using remaining player health; death or timeout adds <b>-100</b>. '
          'Waiting/resting earns no bonus. The optional terminal/time-only mode removes the health and stamina terms.'),
        p('<b>Transition.</b> Apply the player action and stamina cost, then resolve the previously announced '
          'boss intent at the new distance. A lethal player strike cancels retaliation. Slash deals 2 damage '
          'at close range; sweep deals 3 at close/medium; lunge advances one band then deals 1 if close; '
          'recovery deals none. Decrease the clock, check victory/death before timeout, and sample the next '
          'visible intent only if combat continues.'),
    ]
    code = summary['code_figure']
    story += picture('figure2_code.png',
        f'<b>Figure 2.</b> Verbatim source excerpt from <b>{code["path"]}</b>, lines '
        f'<b>{code["first_line"]}-{code["last_line"]}</b>. Full validation and helper methods are in the source ZIP.',
        max_height=330)
    story += [
        p('The excerpt shows the executed transition order: mask checking, player action, conditional '
          'boss retaliation, clock/outcome update, and stochastic next intent. Reward terms are summed '
          'explicitly before returning Gymnasium\'s observation/reward/termination/info tuple.'),
        p('3. Environment Testing', 'heading'),
        bullet(f'<b>{summary["tests"]["run"]}/{summary["tests"]["run"]} tests passed:</b> '
               'Gymnasium API/render checks, reset/reseeding, observation bounds, masks, stamina, '
               'attack ranges, guard effects, rewards, and victory/death/timeout ordering.'),
        bullet('<b>Reset and action checks:</b> reset restores the six initial values and masks; identical '
               'seeds/actions reproduce trajectories. Controlled cases verify lethal-hit priority, '
               'last-turn timeouts, zero stamina, and invalid-action handling.'),
        bullet('<b>Constraints and uncertainty:</b> 280 complete random-action episodes check invariants; '
               '60,000 sampled intents check controller probabilities. Mixed rules are resampled each turn, '
               'so stochastic behavior does not introduce an unobserved persistent opponent type.'),
    ]

    story += [PageBreak(), p('3. Environment Testing (continued)', 'heading')]
    story += picture('figure3_execution.png',
        '<b>Figure 3.</b> Summary rendered from actual test, demo, and smoke outputs. '
        'Full records: <b>evidence/tests.txt</b>, <b>evidence/demo/trace.json</b>, and '
        '<b>evidence/smoke_results.json</b>. The abbreviated trace omits intermediate turns, not failures.',
        max_height=240)
    step1, step2 = trace['steps'][:2]
    story += [
        p('<b>Observed action effects.</b> In the seed-7 demonstration, approach changes medium to close '
          f'and reduces the clock to {step1["observation_after"][5]} (reward {step1["reward"]:.2f}). '
          f'The following guarded sweep leaves player HP at {step2["observation_after"][0]} and stamina at '
          f'{step2["observation_after"][2]} against the sweep (reward {step2["reward"]:.2f}); '
          'the guard blocks 2 of its 3 damage. The fixed reactive heuristic wins this episode in '
          f'<b>{summary["demo"]["turns"]} turns</b>, with <b>{summary["demo"]["remaining_player_health"]} HP</b> '
          f'and return <b>{summary["demo"]["return"]:.3f}</b>.'),
        p('<b>Broader execution check.</b> Three untrained policies were run for 100 episodes per '
          'training mode, using shared seeds 1000-1099. All 1,200 episodes completed within 40 turns; '
          f'the run used {summary["smoke"]["environment_steps"]:,} transitions and submitted '
          f'{summary["smoke"]["invalid_actions"]} masked actions. Measured win rates were:'),
    ]
    rows = [[p('Boss controller', 'small'), p('Random', 'small'), p('Aggressive', 'small'), p('Reactive', 'small')]]
    for opponent, name in [('balanced', 'Balanced'), ('aggressive', 'Aggressive'),
                           ('distance_aware', 'Distance-aware'), ('mixed', 'Mixed')]:
        rates = {r['policy']: r['win_rate'] for r in smoke['results'] if r['opponent'] == opponent}
        rows.append([p(name), *[p(f'{rates[k] * 100:.0f}%') for k in ('random', 'aggressive', 'reactive')]])
    table = Table(rows, colWidths=[192, 112, 112, 112], hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PALE),
        ('LINEBELOW', (0, 0), (-1, 0), 0.7, INK),
        ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor('#bbc9d4')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ]))
    story += [table, Spacer(1, 6),
        p('These small smoke checks demonstrate execution, not learned-policy quality or the benefit '
          'of mixed training. Held-out mechanics are unit-tested, but held-out policy benchmarking '
          'remains reserved. Optional animated replay: <b>evidence/demo/demo.gif</b> (actual recorded frames).', 'small'),
        p('4. Next Steps', 'heading'),
        bullet('<b>Implement:</b> masked Q-learning and SARSA, equal-step-budget single/mixed training, '
               'multiple training seeds, and the reward ablation. Freeze the design and validation choices '
               'before benchmarking on reserved enemy behaviors.'),
        bullet('<b>Current problem:</b> combat is demanding: reactive-heuristic win rates range from '
               '13% to 56% in these checks. Calibrate the baseline and difficulty on development/validation '
               'episodes, document any rule changes, and establish learning progress before final evaluation.'),
        p('<b>Implementation clarifications.</b> A directly submitted masked action wastes a turn; '
          'the provided policies/UI prevent it. Timeout is a terminal loss, not external truncation, '
          'because the deadline is intrinsic and observed. Core combat values match the revised proposal.', 'small'),
        p('API references: <link href="https://gymnasium.farama.org/introduction/create_custom_env/" '
          'color="#15384d">Gymnasium custom environments</link>; '
          '<link href="https://farama.org/Gymnasium-Terminated-Truncated-Step-API" '
          'color="#15384d">Farama termination/truncation guidance</link>.', 'small'),
    ]

    output = ROOT / 'report' / 'milestone2_report.pdf'
    output.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=42, leftMargin=42,
                            topMargin=34, bottomMargin=38,
                            title='Read the Boss - Milestone 2 Environment Implementation',
                            author=args.author)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#b7c8d5'))
        canvas.line(42, 28, 570, 28)
        canvas.setFont(regular, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(42, 16, 'READ THE BOSS  |  MILESTONE 2  |  ENVIRONMENT EVIDENCE')
        canvas.drawRightString(570, 16, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Created {output}')


if __name__ == '__main__':
    main()
