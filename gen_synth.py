import json, time, random

"""
Synthetic data generator matching expected human patterns:

- Two teacher styles via `style`: 0=Candid, 1=Tactful
- Student confidence via `r`: 0=Unconfident, 1=Confident
- Worlds:
  * h>t, D=1  -> Only D true
  * h<t, D=1  -> Both T and D true
  * h<t, D=0  -> Only T true

Behavioral rules to emulate:
- Candid (baseline): choose the truthful utterance(s) near 1.0.
- Tactful, Confident: nearly identical to Candid.
- Tactful, Unconfident: key warp in h<t & D=1 → prefer u_D (~0.8–0.9);
  still choose u_D (~1.0) when only D true; choose u_T (~1.0) when only T true.
"""

# Domains from the current model setup
F_vals = [0, 1]  # 0=BOTH, 1=EITHER
R_TEACHER = [0, 1]  # 0=Candid, 1=Tactful
D_vals = [0, 1]  # 0=Easy, 1=Difficult
T_vals = [1, 3]  # threshold values, index: 0->1 ("insufficient"), 1->3 ("sufficient")
H_vals = [1, 3]  # hours, index: 0->1 (low), 1->3 (high)

# Guideline-based selection rules
def choose_outputs_guideline_prob(style, student_c, h, t, d):
    """Guideline-driven outputs with probabilistic noise.

    style: 0=candid, 1=tactful (teacher style)
    student_c: 0=unconfident, 1=confident (student type C)
    Returns (out_t, out_v) as 'y'/'n' flags.
    """
    stuck = h < t

    # Base probabilities centered on the guideline's optimal utterance.
    if style == 0:  # Candid (epistemic-first, independent of confidence)
        if not stuck and d == 1:      # Only D true => u_D
            p_t, p_v = 0.10, 0.92
        elif stuck and d == 1:        # Both true => u_TD
            p_t, p_v = 0.88, 0.88
        elif stuck and d == 0:        # Only T true => u_T
            p_t, p_v = 0.92, 0.10
        else:
            p_t, p_v = 0.50, 0.50
    else:  # Tactful (social-aware; depends on student confidence)
        if student_c == 1:  # Confident → looks like Candid
            if not stuck and d == 1:
                p_t, p_v = 0.10, 0.92
            elif stuck and d == 1:
                p_t, p_v = 0.88, 0.88
            elif stuck and d == 0:
                p_t, p_v = 0.92, 0.10
            else:
                p_t, p_v = 0.50, 0.50
        else:  # Unconfident → empathy effect in high-cost worlds
            if not stuck and d == 1:      # Only D true → u_D (~1.0)
                p_t, p_v = 0.06, 0.94
            elif stuck and d == 1:        # Both true → prefer u_D (0.8–0.9)
                p_t, p_v = 0.14, 0.88
            elif stuck and d == 0:        # Only T true → u_T (~1.0)
                p_t, p_v = 0.94, 0.06
            else:
                p_t, p_v = 0.50, 0.50

    # Preserve probabilistic jitter around the base preference
    jitter_t = random.uniform(-0.12, 0.12)
    jitter_v = random.uniform(-0.12, 0.12)
    p_t = max(0.05, min(0.95, p_t + jitter_t))
    p_v = max(0.05, min(0.95, p_v + jitter_v))

    out_t = 'y' if random.random() < p_t else 'n'
    out_v = 'y' if random.random() < p_v else 'n'
    return out_t, out_v

def make_explanation(f, r, d, t, v, out_t, out_v):
    f_txt = 'BOTH' if f == 0 else 'EITHER'
    r_txt = 'Confident (r=1)' if r == 1 else 'Unconfident (r=0)'
    d_txt = 'Difficult (d=1)' if d == 1 else 'Easy (d=0)'
    stuck_txt = 'STUCK' if v < t else 'FLUENT'

    # Slightly varied templates to introduce linguistic variance
    templates = [
        "I speak about both time/threshold and difficulty for clarity.",
        "I address both threshold/time and the task difficulty to help.",
        "I mention time vs threshold and note the difficulty explicitly.",
    ]
    t_only = [
        "I focus on time/threshold and keep difficulty unspecified.",
        "I prefer discussing threshold/time without labeling difficulty.",
        "I talk about time needed rather than difficulty per se.",
    ]
    v_only = [
        "I avoid time/threshold to protect face and cite difficulty.",
        "I emphasize difficulty while skipping threshold/time talk.",
        "I comment on difficulty and avoid time/threshold specifics.",
    ]
    none_tpl = [
        "I keep both dimensions unspecified to stay neutral.",
        "I avoid labeling either threshold/time or difficulty here.",
        "I remain neutral without calling out difficulty or threshold.",
    ]

    parts = [
        f"Teacher in {f_txt}; student {r_txt}; {d_txt}; hours v={v}; threshold t={t} ⇒ {stuck_txt}.",
    ]
    if out_t == 'y' and out_v == 'y':
        parts.append(random.choice(templates))
    elif out_t == 'y' and out_v == 'n':
        parts.append(random.choice(t_only))
    elif out_t == 'n' and out_v == 'y':
        parts.append(random.choice(v_only))
    else:
        parts.append(random.choice(none_tpl))
    return ' '.join(parts)


def generate(path, participants_per_panel=8):
    """
    Generate one trial per participant across F × TeacherStyle × WorldState.

    - A "panel" refers to one fixed combination of (F, TeacherStyle, WorldState)
      which corresponds to one small subplot cell in the bar-chart grid.
    - participants_per_panel controls how many participants you place in each panel.
    """
    # Base epoch and jitter for more human-like timing
    now = 1726000000.0
    pid_base = 6000000
    rows = []

    # Panels: F × TeacherStyle(r) × WorldState
    WORLD_STATES = [
        (3, 1, 1),  # h>t, D=1
        (1, 3, 1),  # h<t, D=1
        (1, 3, 0),  # h<t, D=0
    ]

    i = 0
    for f in F_vals:
        for style in R_TEACHER:  # 0=candid, 1=tactful
            for (h_val, t_val, d_val) in WORLD_STATES:
                for j in range(participants_per_panel):
                    d = d_val
                    t = t_val
                    v = h_val
                    # Alternate student confidence r across participants to populate both rows
                    r = j % 2  # 0=Unconfident, 1=Confident
                    out_t, out_v = choose_outputs_guideline_prob(style, r, v, t, d)
                    exp = make_explanation(f, r, d, t, v, out_t, out_v)
                    pid = f"5f48{pid_base + i:07d}"

                    # Add small random jitter to timestamps and response times
                    base_ts = now + i * 0.123
                    ts_jitter = random.uniform(-0.9, 0.9)
                    rt_jitter = random.uniform(-1.5, 1.5)

                    row = {
                        "prolific": [pid],
                        "time_": base_ts + ts_jitter,
                        "f": f,
                        "r": r,               # student confidence
                        "style": style,       # teacher style (0=candid,1=tactful)
                        "d": d,
                        "t": t,
                        "v": v,
                        "out-t": [out_t],
                        "out-v": [out_v],
                        "explanation": [exp],
                        "time_response": (base_ts + 7.5) + rt_jitter,
                    }
                    rows.append(row)
                    i += 1
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)

if __name__ == '__main__':
    import sys
    # Default to a smaller sample; override with a CLI arg if provided
    per = 8
    if len(sys.argv) > 1:
        try:
            per = int(sys.argv[1])
        except Exception:
            pass
    n = generate('synthetic_log.jsonl', participants_per_panel=per)
    print('wrote', n, 'rows')
