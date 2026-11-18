"""Audit: rule edge cases via fan calculator and judge/observer gates.

Read-only checks (no project files modified). Outputs audit/rule_checks.json.
Covers: 8-fan gate, 抢杠胡 flag propagation, 杠上开花, 海底/妙手, 七对,
fourth-tile counting, invalid-Hu penalty, exhaustive draw.
"""

import json
from pathlib import Path

from MahjongGB import MahjongFanCalculator

from mahjong.gamecore.engine import GameCore
from mahjong.observation import SeatView

ROOT = Path(__file__).resolve().parent.parent
results = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail})


def fan_total(**kw):
    base = dict(
        pack=(), hand=(), winTile="", flowerCount=0, isSelfDrawn=False,
        is4thTile=False, isAboutKong=False, isWallLast=False,
        seatWind=0, prevalentWind=0, verbose=True,
    )
    base.update(kw)
    fans = MahjongFanCalculator(**base)
    return sum(f[0] * f[1] for f in fans), [(f[0], f[2]) for f in fans]


# 1. Fan calculator known values.
total, names = fan_total(
    hand=("W1", "W1", "W1", "B5", "B5", "B5", "T7", "T7", "T7", "J2"),
    pack=(("PENG", "J1", 1),), winTile="J2",
)
check("fan_pengpenghu_case", total == 26, "total=%d %s" % (total, names))

total, names = fan_total(
    hand=("W1", "W1", "B2", "B2", "T3", "T3", "J1", "J1", "F1", "F1", "W5", "W5", "B9"),
    winTile="B9", isSelfDrawn=True,
)
check("fan_seven_pairs", total >= 24, "total=%d %s" % (total, names))

total, names = fan_total(
    hand=("W1", "W2", "W3", "B4", "B5", "B6", "T7", "T8", "T9", "J1"),
    pack=(("GANG", "J2", 0),), winTile="J1", isSelfDrawn=True, isAboutKong=True,
)
check("fan_kong_bloom_flag", any("杠上开花" in n for _, n in names), "total=%d %s" % (total, names))

total, names = fan_total(
    hand=("W1", "W2", "W3", "W4", "W5", "W6", "B1", "B2", "B3", "T4", "T5", "T6", "J1"),
    winTile="J1", isSelfDrawn=True, isWallLast=True,
)
check("fan_wall_last_selfdraw", any("妙手回春" in n for _, n in names), "total=%d %s" % (total, names))

total, names = fan_total(
    hand=("W1", "W2", "W3", "W4", "W5", "W6", "B1", "B2", "B3", "T4", "T5", "T6", "J1"),
    winTile="J1", isSelfDrawn=False, isWallLast=True,
)
check("fan_wall_last_ron", any("海底捞月" in n for _, n in names), "total=%d %s" % (total, names))

# 2. 8-fan gate in SeatView: low-fan ron must NOT offer Hu.
low_hand = "W1 W2 W3 W4 W5 W6 B1 B2 B3 T4 T5 T6 J1"  # 平和2+喜相逢x2=4 fan on J1
view = SeatView(0)
view.request2obs("Wind 0")
view.request2obs("Deal " + low_hand)
view.request2obs("Player 3 Play J1")
check("gate_low_fan_ron_no_hu", 1 not in view.valid, "valid=%s" % view.valid)
total, names = fan_total(hand=tuple(low_hand.split()), winTile="J1")
check("gate_low_fan_value", total < 8, "total=%d %s" % (total, names))

# Same-shape robbing a kong: 抢杠和 +8 must make Hu legal. The wait must be
# two-sided (winner holds 0 of the win tile); a single-wait rob is physically
# impossible (the kong player could not hold 3+draw the 4th copy).
rob_hand = "W1 W2 W3 W4 W5 W6 B1 B2 B3 T4 T5 J1 J1"  # waits T3/T6
view2 = SeatView(0)
view2.request2obs("Wind 0")
view2.request2obs("Deal " + rob_hand)
view2.request2obs("Player 1 BuGang T6")
check("gate_rob_kong_hu_offered", 1 in view2.valid, "valid=%s" % view2.valid)
total, names = fan_total(hand=tuple(rob_hand.split()), winTile="T6", isAboutKong=True)
check("gate_rob_kong_fan", total >= 8 and any("抢杠和" in n for _, n in names), "total=%d %s" % (total, names))

# 3. Judge rejects a Hu with insufficient fans (invalid penalty -30/+10).
core = GameCore()
core.r.seed(5)
core.reset()
# Force player 0's hand to a 6-fan self-draw shape and current tile to J1.
core.hands[0] = "W1 W2 W3 B5 B6 B7 T2 T3 T4 B9 B9 B9 J1".split()
core.curTile = "J1"
core.state = 1
core.step({core.agent_names[0]: "Hu"})
check(
    "judge_invalid_hu_penalty",
    core.done and core.reward[0] == -30 and core.reward[1] == 10,
    "reward=%s invalid=%s" % (core.reward, core.invalid_info),
)

# 4. Fourth-tile counting on self-draw (shownTiles 3 + drawn = 4).
core2 = GameCore()
core2.r.seed(0)
core2.reset()
core2.shownTiles[core2.curTile] = 3
captured = {}
orig = MahjongFanCalculator


def spy(**kw):
    captured.update(kw)
    return [(8, 1, "test", "test")]


import mahjong.gamecore.engine as eng

eng.MahjongFanCalculator = spy
try:
    core2.step({core2.agent_names[core2.curPlayer]: "Hu"})
finally:
    eng.MahjongFanCalculator = orig
check("judge_fourth_tile_selfdraw", captured.get("is4thTile") is True, str(captured.get("is4thTile")))

# 5. Exhaustive draw gives zero rewards (find a seed that draws).
draw_found = False
for seed in range(200):
    c = GameCore()
    c.r.seed(seed)
    c.reset()
    import numpy as np

    rng = np.random.RandomState(seed)
    views = [SeatView(i) for i in range(4)]
    offsets = [0] * 4
    steps = 0
    while not c.done and steps < 600:
        for i in range(4):
            ev = c.agents[i].events
            for e in ev[offsets[i]:]:
                views[i].request2obs(e)
            offsets[i] = len(ev)
        cmds = {}
        for i in c.obs:
            legal = views[i].valid
            a = int(rng.choice(legal))
            cmds[c.agent_names[i]] = views[i].action2response(a)
        c.step(cmds)
        steps += 1
    if c.done and c.reward == [0, 0, 0, 0]:
        draw_found = True
        break
check("exhaustive_draw_zero_reward", draw_found, "seed=%d" % seed)

(ROOT / "audit/rule_checks.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
for r in results:
    print(("PASS" if r["pass"] else "FAIL"), r["check"], "|", r["detail"])
print("%d/%d passed" % (sum(r["pass"] for r in results), len(results)))