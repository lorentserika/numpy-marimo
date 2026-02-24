import marimo

__generated_with = "0.20.2"
app = marimo.App(width="compact")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import ast
    import re
    import matplotlib.pyplot as plt

    return ast, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # NumPy Exercises — interaktiivne Marimo

    - ✅ **Iga rida on lahti seletatud** slidrite ja visuaalidega
    - ✅ **Tudengi sisestuskastid** kõigile 21 reale
    - ✅ **"Kontrolli" nupp** — vastust kontrollitakse ainult nupuvajutusel
    - ✅ **3-tasemeline vihjesüsteem** (accordion)
    """)
    return


@app.cell
def _(ast, np):
    def safe_eval_expr(expr, local_env):
        expr = expr.strip()
        if not expr:
            return {"ok": False, "error": "Tühi vastus."}
        try:
            tree = ast.parse(expr, mode="eval")
            allowed = (
                ast.Expression, ast.Call, ast.Name, ast.Load, ast.Attribute,
                ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice,
                ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                ast.USub, ast.UnaryOp, ast.Compare, ast.Eq, ast.NotEq,
                ast.Gt, ast.GtE, ast.Lt, ast.LtE
            )
            for node in ast.walk(tree):
                if not isinstance(node, allowed):
                    return {"ok": False, "error": f"Lubamatu süntaks: {type(node).__name__}"}
            value = eval(compile(tree, "<student>", "eval"), {"np": np}, local_env)
            return {"ok": True, "value": value}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    def pretty(x):
        try:
            return np.array2string(np.asarray(x), separator=", ")
        except Exception:
            return repr(x)

    def review_expr(student_expr, expected_value, local_env=None, hint=None):
        if local_env is None:
            local_env = {}
        r = safe_eval_expr(student_expr, local_env)
        if not r["ok"]:
            fb = [f"Viga: {r['error']}"]
            if hint:
                fb.append(f"Vihje: {hint}")
            return {"passed": False, "feedback": fb}
        val = r["value"]
        ok = False
        try:
            if isinstance(expected_value, np.ndarray):
                ok = np.array_equal(np.asarray(val), expected_value)
            else:
                ok = (val == expected_value) or (str(val) == str(expected_value))
        except Exception:
            ok = (str(val) == str(expected_value))
        if ok:
            return {"passed": True, "feedback": ["Õige!"]}
        fb = [f"Oodatud: {pretty(expected_value)}", f"Sinu tulemus: {pretty(val)}"]
        if hint:
            fb.append(f"Vihje: {hint}")
        return {"passed": False, "feedback": fb}

    def review_rand(student_expr, expected_shape, local_env=None):
        if local_env is None:
            local_env = {}
        r = safe_eval_expr(student_expr, local_env)
        if not r["ok"]:
            return {"passed": False, "feedback": [f"Viga: {r['error']}"]}
        try:
            arr = np.asarray(r["value"], dtype=float)
            shape_ok = arr.shape == expected_shape
            range_ok = np.all(arr >= 0) and np.all(arr < 1)
            if shape_ok and range_ok:
                return {"passed": True, "feedback": [f"Õige! Kuju {arr.shape}, väärtused [0,1)."]}
            fb = []
            if not shape_ok:
                fb.append(f"Oodatud kuju: {expected_shape}, sinu kuju: {arr.shape}")
            if not range_ok:
                fb.append("Väärtused peaksid olema vahemikus [0, 1).")
            return {"passed": False, "feedback": fb}
        except Exception:
            return {"passed": False, "feedback": ["Tulemus ei ole numbriline massiiv."]}

    def review_randn(student_expr, expected_shape, local_env=None):
        if local_env is None:
            local_env = {}
        r = safe_eval_expr(student_expr, local_env)
        if not r["ok"]:
            return {"passed": False, "feedback": [f"Viga: {r['error']}"]}
        try:
            arr = np.asarray(r["value"], dtype=float)
            if arr.shape == expected_shape:
                return {"passed": True, "feedback": [f"Õige! Kuju {arr.shape}, normaaljaotuse valim."]}
            return {"passed": False, "feedback": [f"Oodatud kuju: {expected_shape}, sinu kuju: {arr.shape}"]}
        except Exception:
            return {"passed": False, "feedback": ["Tulemus ei ole numbriline massiiv."]}

    return review_expr, review_rand, review_randn


@app.cell
def _(mo):
    mo.md("""
    ## 1) `import numpy as np` — miks alias `np`?
    """)
    return


@app.cell
def _(mo):
    alias = mo.ui.text(value="np", label="Aliase nimi (tavaliselt `np`)")
    module_name = mo.ui.text(value="numpy", label="Mooduli nimi")
    mo.hstack([module_name, alias], justify="start")
    return alias, module_name


@app.cell
def _(alias, mo, module_name):
    mo.md(f"""
    ```python
    import {module_name.value} as {alias.value}
    ```
    **Selgitus:** Alias teeb kirjutamise lühemaks. Kogu NumPy ökosüsteem kasutab aliasena **np**.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2) `np.zeros`, `np.ones`, `np.ones()*5` — massiivi loomine
    """)
    return


@app.cell
def _(mo):
    n_basic = mo.ui.slider(1, 30, value=10, label="Massiivi pikkus `n`")
    multiplier = mo.ui.slider(-5, 10, value=5, label="Korrutaja (`np.ones(n) * x`)")
    mo.vstack([n_basic, multiplier])
    return multiplier, n_basic


@app.cell
def _(mo, multiplier, n_basic, np, plt):
    _z = np.zeros(n_basic.value)
    _o = np.ones(n_basic.value)
    _f = np.ones(n_basic.value) * multiplier.value
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3))
    for _ax, _data, _title in zip(_axes, [_z, _o, _f], ["zeros", "ones", f"ones * {multiplier.value}"]):
        _ax.bar(range(len(_data)), _data, color=["#4C72B0", "#55A868", "#C44E52"][_axes.tolist().index(_ax)])
        _ax.set_title(_title)
        _ax.set_ylim(min(-1, _data.min() - 1), max(2, _data.max() + 1))
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
    ```python
    np.zeros({n_basic.value}) -> {_z}
    np.ones({n_basic.value})  -> {_o}
    np.ones({n_basic.value}) * {multiplier.value} -> {_f}
    ```
    """),
        _fig
    ])
    return


@app.cell
def _(mo):
    q_zeros = mo.ui.text_area(value="", label="Harjutus 2: kirjuta avaldis, mis loob 10 nulliga massiivi", rows=2, full_width=True)
    btn_zeros = mo.ui.button(label="Kontrolli")
    mo.vstack([q_zeros, btn_zeros])
    return btn_zeros, q_zeros


@app.cell
def _(btn_zeros, mo, np, q_zeros, review_expr):
    mo.stop(btn_zeros.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_zeros.value, np.zeros(10), hint="Kasuta `np.zeros(...)`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`np.zeros(n)` loob massiivi, kus kõik elemendid on 0."),
        "Vihje 2": mo.md("Argument on elementide arv. Sul on vaja 10 elementi."),
        "Täielik lahendus": mo.md("```python\nnp.zeros(10)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_ones = mo.ui.text_area(value="", label="Harjutus 3: kirjuta avaldis, mis loob 10 ühtedega massiivi", rows=2, full_width=True)
    btn_ones = mo.ui.button(label="Kontrolli")
    mo.vstack([q_ones, btn_ones])
    return btn_ones, q_ones


@app.cell
def _(btn_ones, mo, np, q_ones, review_expr):
    mo.stop(btn_ones.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_ones.value, np.ones(10), hint="Kasuta `np.ones(...)`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("Sarnane `np.zeros`-ile, aga väärtus on 1."),
        "Vihje 2": mo.md("`np.ones(n)` loob `n`-elemendilise massiivi ühtedega."),
        "Täielik lahendus": mo.md("```python\nnp.ones(10)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_fives = mo.ui.text_area(value="", label="Harjutus 4: kirjuta avaldis, mis loob 10 viiega massiivi (kasutades np.ones)", rows=2, full_width=True)
    btn_fives = mo.ui.button(label="Kontrolli")
    mo.vstack([q_fives, btn_fives])
    return btn_fives, q_fives


@app.cell
def _(btn_fives, mo, np, q_fives, review_expr):
    mo.stop(btn_fives.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_fives.value, np.ones(10) * 5, hint="Kõigepealt loo ühtede massiiv, siis korruta.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("NumPy massiive saab korrutada skalaarsega: `massiiv * arv`."),
        "Vihje 2": mo.md("Loo `np.ones(10)`, siis korruta `5`-ga."),
        "Täielik lahendus": mo.md("```python\nnp.ones(10) * 5\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3) `np.arange(...)` — vahemikud ja samm
    """)
    return


@app.cell
def _(mo):
    start = mo.ui.slider(-10, 20, value=10, label="Algus `start`")
    stop = mo.ui.slider(-5, 60, value=51, label="Lõpp `stop` (EI kuulu hulka)")
    step = mo.ui.slider(1, 10, value=1, label="Samm `step`")
    mo.vstack([start, stop, step])
    return start, step, stop


@app.cell
def _(mo, np, plt, start, step, stop):
    _arr = np.arange(start.value, stop.value, step.value)
    _fig, _ax = plt.subplots(figsize=(10, 2))
    _ax.scatter(_arr, np.zeros_like(_arr), c=_arr, cmap="viridis", s=40, zorder=2)
    _ax.set_yticks([])
    _ax.set_title(f"np.arange({start.value}, {stop.value}, {step.value}) — {len(_arr)} elementi")
    _ax.axhline(0, color="gray", linewidth=0.5)
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"```python\nnp.arange({start.value}, {stop.value}, {step.value}) -> {_arr}\n```\n**`stop` on exclusive** (piir ei kuulu sisse)."),
        _fig
    ])
    return


@app.cell
def _(mo):
    even_toggle = mo.ui.checkbox(value=True, label="Näita paarisarvude näidet (samm = 2)")
    even_start = mo.ui.slider(0, 30, value=10, label="Paarisarvude algus")
    even_stop = mo.ui.slider(2, 60, value=51, label="Paarisarvude lõpp (exclusive)")
    mo.vstack([even_toggle, even_start, even_stop])
    return even_start, even_stop, even_toggle


@app.cell
def _(even_start, even_stop, even_toggle, mo, np):
    if even_toggle.value:
        _arr_even = np.arange(even_start.value, even_stop.value, 2)
        mo.md(f"### Samm 2 (paarisarvud)\n```python\nnp.arange({even_start.value}, {even_stop.value}, 2) -> {_arr_even}\n```")
    else:
        mo.md("_Paarisarvude plokk on peidetud._")
    return


@app.cell
def _(mo):
    q_arange = mo.ui.text_area(value="", label="Harjutus 5: kirjuta avaldis, mis loob massiivi täisarvudega 5 kuni 10", rows=2, full_width=True)
    btn_arange = mo.ui.button(label="Kontrolli")
    mo.vstack([q_arange, btn_arange])
    return btn_arange, q_arange


@app.cell
def _(btn_arange, mo, np, q_arange, review_expr):
    mo.stop(btn_arange.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_arange.value, np.arange(5, 11), hint="Kasuta `np.arange(start, stop)` — stop EI kuulu sisse.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`np.arange(start, stop)` loob järjestikused arvud `start` kuni `stop-1`."),
        "Vihje 2": mo.md("Kui tahad arve 5 kuni 10, siis `stop` peab olema **11**."),
        "Täielik lahendus": mo.md("```python\nnp.arange(5, 11)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_arange_even = mo.ui.text_area(value="", label="Harjutus 6: kirjuta avaldis, mis loob paarisarvude massiivi 2-st kuni 8-ni", rows=2, full_width=True)
    btn_arange_even = mo.ui.button(label="Kontrolli")
    mo.vstack([q_arange_even, btn_arange_even])
    return btn_arange_even, q_arange_even


@app.cell
def _(btn_arange_even, mo, np, q_arange_even, review_expr):
    mo.stop(btn_arange_even.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_arange_even.value, np.arange(2, 9, 2), hint="Kasuta kolmandat argumenti sammuna.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`np.arange` kolmas argument on **samm** (step)."),
        "Vihje 2": mo.md("Paarisarvud 2,4,6,8: `np.arange(2, lõpp, 2)`. Mis peab olema `lõpp`, et 8 sisse jääks?"),
        "Täielik lahendus": mo.md("```python\nnp.arange(2, 9, 2)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4) `reshape` ja `np.eye` — maatriksid
    """)
    return


@app.cell
def _(mo):
    dim = mo.ui.slider(2, 6, value=3, label="Maatriksi mõõt `n` (n×n)")
    dim
    return (dim,)


@app.cell
def _(dim, mo, np, plt):
    _n = dim.value
    _m = np.arange(_n * _n).reshape(_n, _n)
    _eye = np.eye(_n)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(8, 3))
    _ax1.imshow(_m, cmap="YlOrRd")
    for _i in range(_n):
        for _j in range(_n):
            _ax1.text(_j, _i, str(_m[_i, _j]), ha="center", va="center", fontsize=10)
    _ax1.set_title(f"arange({_n*_n}).reshape({_n},{_n})")
    _ax2.imshow(_eye, cmap="Blues")
    for _i in range(_n):
        for _j in range(_n):
            _ax2.text(_j, _i, f"{_eye[_i,_j]:.0f}", ha="center", va="center", fontsize=10)
    _ax2.set_title(f"eye({_n})")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"```python\nnp.arange({_n*_n}).reshape({_n},{_n})\n{_m}\n\nnp.eye({_n})\n{_eye}\n```"),
        _fig
    ])
    return


@app.cell
def _(mo):
    q_reshape = mo.ui.text_area(value="", label="Harjutus 7: kirjuta avaldis, mis loob 4×4 maatriksi arvudega 0-15", rows=2, full_width=True)
    btn_reshape = mo.ui.button(label="Kontrolli")
    mo.vstack([q_reshape, btn_reshape])
    return btn_reshape, q_reshape


@app.cell
def _(btn_reshape, mo, np, q_reshape, review_expr):
    mo.stop(btn_reshape.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_reshape.value, np.arange(16).reshape(4, 4), hint="Loo 1D massiiv, siis `.reshape()`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("Kõigepealt loo 16 arvu (0..15), siis muuda kuju 4×4-ks."),
        "Vihje 2": mo.md("`np.arange(16)` annab `[0,1,...,15]`. Lisa `.reshape(4, 4)`."),
        "Täielik lahendus": mo.md("```python\nnp.arange(16).reshape(4, 4)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_eye = mo.ui.text_area(value="", label="Harjutus 8: kirjuta avaldis, mis loob 5×5 identiteedimaatriksi", rows=2, full_width=True)
    btn_eye = mo.ui.button(label="Kontrolli")
    mo.vstack([q_eye, btn_eye])
    return btn_eye, q_eye


@app.cell
def _(btn_eye, mo, np, q_eye, review_expr):
    mo.stop(btn_eye.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_eye.value, np.eye(5), hint="Kasuta `np.eye(...)`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("Identiteedimaatriks on ruutmaatriks, kus diagonaalil on 1-d."),
        "Vihje 2": mo.md("`np.eye(n)` loob n×n identiteedimaatriksi."),
        "Täielik lahendus": mo.md("```python\nnp.eye(5)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5) Juhuarvud: `rand`, `randn`
    """)
    return


@app.cell
def _(mo):
    seed = mo.ui.slider(0, 999, value=42, label="Seeme (`seed`)")
    rand_count = mo.ui.slider(1, 500, value=100, label="`np.random.rand(n)` -> n")
    randn_count = mo.ui.slider(1, 500, value=100, label="`np.random.randn(n)` -> n")
    mo.vstack([seed, rand_count, randn_count])
    return rand_count, randn_count, seed


@app.cell
def _(mo, np, plt, rand_count, randn_count, seed):
    np.random.seed(seed.value)
    _r1 = np.random.rand(rand_count.value)
    np.random.seed(seed.value + 1)
    _r2 = np.random.randn(randn_count.value)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _ax1.hist(_r1, bins=25, color="#4C72B0", edgecolor="white", alpha=0.8)
    _ax1.set_title(f"np.random.rand({rand_count.value}) — ühtlane jaotus [0,1)")
    _ax1.set_xlabel("Väärtus")
    _ax1.set_ylabel("Sagedus")
    _ax2.hist(_r2, bins=25, color="#C44E52", edgecolor="white", alpha=0.8)
    _ax2.set_title(f"np.random.randn({randn_count.value}) — normaaljaotus")
    _ax2.set_xlabel("Väärtus")
    _ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
    ```python
    np.random.seed({seed.value})
    np.random.rand({rand_count.value})   # {rand_count.value} ühtlase jaotuse arvu [0, 1)
    np.random.randn({randn_count.value})  # {randn_count.value} normaaljaotuse arvu (μ=0, σ=1)
    ```
    **`rand`** → ühtlane jaotus [0,1) | **`randn`** → normaaljaotus (keskmine 0, std 1)

    Sama `seed` = sama tulemus — hea testimiseks.
    """),
        _fig
    ])
    return


@app.cell
def _(mo):
    q_rand = mo.ui.text_area(value="", label="Harjutus 9: kirjuta avaldis, mis genereerib 2 juhuslikku reaalarvu vahemikus [0,1)", rows=2, full_width=True)
    btn_rand = mo.ui.button(label="Kontrolli")
    mo.vstack([q_rand, btn_rand])
    return btn_rand, q_rand


@app.cell
def _(btn_rand, mo, q_rand, review_rand):
    mo.stop(btn_rand.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_rand(q_rand.value, (2,))
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`np.random.rand(n)` genereerib `n` juhuslikku arvu vahemikus [0,1)."),
        "Vihje 2": mo.md("Sul on vaja 2 arvu, seega argument on `2`."),
        "Täielik lahendus": mo.md("```python\nnp.random.rand(2)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_randn = mo.ui.text_area(value="", label="Harjutus 10: kirjuta avaldis, mis genereerib 30 normaaljaotuse juhuarvu", rows=2, full_width=True)
    btn_randn = mo.ui.button(label="Kontrolli")
    mo.vstack([q_randn, btn_randn])
    return btn_randn, q_randn


@app.cell
def _(btn_randn, mo, q_randn, review_randn):
    mo.stop(btn_randn.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_randn(q_randn.value, (30,))
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`randn` genereerib normaaljaotuse (Gaussi) valimeid."),
        "Vihje 2": mo.md("`np.random.randn(30)` annab 30 arvu keskmisega ~0 ja std ~1."),
        "Täielik lahendus": mo.md("```python\nnp.random.randn(30)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6) `reshape` + jagamine ja `linspace`
    """)
    return


@app.cell
def _(mo):
    size10 = mo.ui.slider(2, 12, value=10, label="Maatriksi mõõt (n×n)")
    denom_mode = mo.ui.dropdown(options={"100":100, "10":10, "n*n":0}, value="100", label="Jagaja")
    lin_n = mo.ui.slider(2, 40, value=20, label="`linspace` punktide arv")
    lin_start = mo.ui.slider(-2.0, 2.0, value=0.0, step=0.1, label="`linspace` algus")
    lin_stop = mo.ui.slider(-2.0, 5.0, value=1.0, step=0.1, label="`linspace` lõpp")
    mo.vstack([size10, denom_mode, lin_n, lin_start, lin_stop])
    return denom_mode, lin_n, lin_start, lin_stop, size10


@app.cell
def _(denom_mode, lin_n, lin_start, lin_stop, mo, np, plt, size10):
    _n = size10.value
    _den = (_n*_n) if denom_mode.value == 0 else denom_mode.value
    _mat = np.arange(1, _n*_n + 1).reshape(_n, _n) / _den
    _lin = np.linspace(lin_start.value, lin_stop.value, lin_n.value)
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _ax1.imshow(_mat, cmap="YlOrRd", aspect="auto")
    _ax1.set_title(f"arange(1,{_n*_n+1}).reshape({_n},{_n}) / {_den}")
    _ax1.set_xlabel("Veerg")
    _ax1.set_ylabel("Rida")
    _ax2.scatter(_lin, np.zeros_like(_lin), c=range(len(_lin)), cmap="viridis", s=60, zorder=2)
    _ax2.set_yticks([])
    _ax2.axhline(0, color="gray", linewidth=0.5)
    _ax2.set_title(f"linspace({lin_start.value}, {lin_stop.value}, {lin_n.value})")
    _ax2.set_xlabel("Väärtus")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"```python\nnp.arange(1, {_n*_n+1}).reshape({_n}, {_n}) / {_den}\nnp.linspace({lin_start.value}, {lin_stop.value}, {lin_n.value})\n```"),
        _fig
    ])
    return


@app.cell
def _(mo):
    q_matdiv = mo.ui.text_area(value="", label="Harjutus 11: kirjuta avaldis, mis loob 5×5 maatriksi väärtustega 0.04..1.0 (arvud 1..25 jagatud 25-ga)", rows=2, full_width=True)
    btn_matdiv = mo.ui.button(label="Kontrolli")
    mo.vstack([q_matdiv, btn_matdiv])
    return btn_matdiv, q_matdiv


@app.cell
def _(btn_matdiv, mo, np, q_matdiv, review_expr):
    mo.stop(btn_matdiv.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_matdiv.value, np.arange(1, 26).reshape(5, 5) / 25, hint="Kombineeri `arange`, `reshape` ja jagamine.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("Kõigepealt loo arvud 1 kuni 25, siis muuda kuju, siis jaga."),
        "Vihje 2": mo.md("`np.arange(1, 26)` → `.reshape(5, 5)` → `/ 25`"),
        "Täielik lahendus": mo.md("```python\nnp.arange(1, 26).reshape(5, 5) / 25\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_linspace = mo.ui.text_area(value="", label="Harjutus 12: kirjuta avaldis, mis loob 10 võrdselt jaotatud punkti 0 ja 1 vahel", rows=2, full_width=True)
    btn_linspace = mo.ui.button(label="Kontrolli")
    mo.vstack([q_linspace, btn_linspace])
    return btn_linspace, q_linspace


@app.cell
def _(btn_linspace, mo, np, q_linspace, review_expr):
    mo.stop(btn_linspace.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_linspace.value, np.linspace(0, 1, 10), hint="Kasuta `np.linspace(algus, lõpp, punktide_arv)`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`np.linspace` loob kindla arvu **võrdselt jaotatud** punkte."),
        "Vihje 2": mo.md("Algus 0, lõpp 1, punkte 10."),
        "Täielik lahendus": mo.md("```python\nnp.linspace(0, 1, 10)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7) NumPy indekseerimine — `mat = np.arange(1,26).reshape(5,5)`
    """)
    return


@app.cell
def _(np):
    mat5 = np.arange(1, 26).reshape(5, 5)
    return (mat5,)


@app.cell
def _(mat5, mo, plt):
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.imshow(mat5, cmap="YlOrRd")
    for _i in range(5):
        for _j in range(5):
            _ax.text(_j, _i, str(mat5[_i, _j]), ha="center", va="center", fontsize=12, fontweight="bold")
    _ax.set_xticks(range(5))
    _ax.set_yticks(range(5))
    _ax.set_xlabel("Veeru indeks")
    _ax.set_ylabel("Rea indeks")
    _ax.set_title("mat = np.arange(1,26).reshape(5,5)")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
    ```python
    mat = np.arange(1, 26).reshape(5, 5)
    # Indeksid algavad 0-st: mat[rida, veerg]
    # mat[0, 0] = 1,  mat[0, 4] = 5,  mat[4, 4] = 25
    ```
    """),
        _fig
    ])
    return


@app.cell
def _(mo):
    q_matcreate = mo.ui.text_area(value="", label="Harjutus 13: kirjuta avaldis, mis loob 6×6 maatriksi väärtustega 10-45", rows=2, full_width=True)
    btn_matcreate = mo.ui.button(label="Kontrolli")
    mo.vstack([q_matcreate, btn_matcreate])
    return btn_matcreate, q_matcreate


@app.cell
def _(btn_matcreate, mo, np, q_matcreate, review_expr):
    mo.stop(btn_matcreate.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_matcreate.value, np.arange(10, 46).reshape(6, 6), hint="Kombineeri `arange` ja `reshape`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("Loo esmalt 1D massiiv arvudega 10..45 (36 arvu = 6×6)."),
        "Vihje 2": mo.md("`np.arange(10, 46)` annab arvud 10..45. Lisa `.reshape(6, 6)`."),
        "Täielik lahendus": mo.md("```python\nnp.arange(10, 46).reshape(6, 6)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Slicing-demonstratsioon (slidrid)
    """)
    return


@app.cell
def _(mo):
    r_start = mo.ui.slider(0, 5, value=2, label="Ridade algus")
    r_stop = mo.ui.slider(0, 5, value=5, label="Ridade lõpp (exclusive)")
    c_start = mo.ui.slider(0, 5, value=1, label="Veergude algus")
    c_stop = mo.ui.slider(0, 5, value=5, label="Veergude lõpp (exclusive)")
    mo.vstack([r_start, r_stop, c_start, c_stop])
    return c_start, c_stop, r_start, r_stop


@app.cell
def _(c_start, c_stop, mat5, mo, np, plt, r_start, r_stop):
    _rs, _re = sorted((r_start.value, r_stop.value))
    _cs, _ce = sorted((c_start.value, c_stop.value))
    _sub = mat5[_rs:_re, _cs:_ce]
    _mask = np.zeros((5, 5))
    _mask[_rs:_re, _cs:_ce] = 1
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(9, 4))
    _ax1.imshow(_mask, cmap="Blues", alpha=0.4)
    _ax1.imshow(mat5, cmap="YlOrRd", alpha=0.6)
    for _i in range(5):
        for _j in range(5):
            _c = "white" if _mask[_i, _j] else "black"
            _w = "bold" if _mask[_i, _j] else "normal"
            _ax1.text(_j, _i, str(mat5[_i, _j]), ha="center", va="center", fontsize=11, color=_c, fontweight=_w)
    _ax1.set_title(f"mat[{_rs}:{_re}, {_cs}:{_ce}]")
    if _sub.size > 0:
        _ax2.imshow(_sub, cmap="YlOrRd")
        for _i in range(_sub.shape[0]):
            for _j in range(_sub.shape[1]):
                _ax2.text(_j, _i, str(_sub[_i, _j]), ha="center", va="center", fontsize=12, fontweight="bold")
        _ax2.set_title("Tulemus")
    else:
        _ax2.text(0.5, 0.5, "Tühi valik", ha="center", va="center", transform=_ax2.transAxes)
        _ax2.set_title("Tulemus")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
    ```python
    mat[{_rs}:{_re}, {_cs}:{_ce}]
    # Read {_rs}..{_re-1}, veerud {_cs}..{_ce-1}
    # Tulemus: {_sub.shape[0]}×{_sub.shape[1]} alammaatriks
    ```
    """ if _sub.size > 0 else f"""
    ```python
    mat[{_rs}:{_re}, {_cs}:{_ce}]  # Tühi valik
    ```
    """),
        _fig
    ])
    return


@app.cell
def _(mo):
    q_sub = mo.ui.text_area(value="", label="Harjutus 14: kirjuta avaldis `mat[...]`, mis valib read alates 2. reast ja veerud alates 3. veerust", rows=2, full_width=True)
    btn_sub = mo.ui.button(label="Kontrolli")
    mo.vstack([q_sub, btn_sub])
    return btn_sub, q_sub


@app.cell
def _(btn_sub, mat5, mo, q_sub, review_expr):
    mo.stop(btn_sub.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_sub.value, mat5[1:, 2:], local_env={"mat": mat5}, hint="Kasuta slicing'ut mõlemal teljel.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`mat[ridade_algus:, veergude_algus:]` — tühja lõpuga tähendab 'lõpuni'."),
        "Vihje 2": mo.md("2. rida on indeksiga **1**, 3. veerg on indeksiga **2** (nullist algav!)."),
        "Täielik lahendus": mo.md("```python\nmat[1:, 2:]\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_single = mo.ui.text_area(value="", label="Harjutus 15: kirjuta avaldis, mis valib 1. rea ja 5. veeru elemendi", rows=2, full_width=True)
    btn_single = mo.ui.button(label="Kontrolli")
    mo.vstack([q_single, btn_single])
    return btn_single, q_single


@app.cell
def _(btn_single, mat5, mo, q_single, review_expr):
    mo.stop(btn_single.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_single.value, mat5[0, 4], local_env={"mat": mat5}, hint="Kasuta `mat[rida, veerg]`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`mat[rida, veerg]` — mõlemad nullist algavad."),
        "Vihje 2": mo.md("1. rida = indeks **0**, 5. veerg = indeks **4**."),
        "Täielik lahendus": mo.md("```python\nmat[0, 4]\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_col = mo.ui.text_area(value="", label="Harjutus 16: kirjuta avaldis, mis valib 3. veerust esimesed 4 elementi veerutulbana (4×1)", rows=2, full_width=True)
    btn_col = mo.ui.button(label="Kontrolli")
    mo.vstack([q_col, btn_col])
    return btn_col, q_col


@app.cell
def _(btn_col, mat5, mo, q_col, review_expr):
    mo.stop(btn_col.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_col.value, mat5[:4, 2].reshape(4, 1), local_env={"mat": mat5}, hint="Vali veerg, siis `.reshape()`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("Vali esimesed 4 rida ja 3. veerg (indeks **2**)."),
        "Vihje 2": mo.md("`mat[:4, 2]` annab 1D massiivi. Lisa `.reshape(4, 1)` veerutulba jaoks."),
        "Täielik lahendus": mo.md("```python\nmat[:4, 2].reshape(4, 1)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_lastrow = mo.ui.text_area(value="", label="Harjutus 17: kirjuta avaldis, mis valib eelviimase rea", rows=2, full_width=True)
    btn_lastrow = mo.ui.button(label="Kontrolli")
    mo.vstack([q_lastrow, btn_lastrow])
    return btn_lastrow, q_lastrow


@app.cell
def _(btn_lastrow, mat5, mo, q_lastrow, review_expr):
    mo.stop(btn_lastrow.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_lastrow.value, mat5[3], local_env={"mat": mat5}, hint="Eelviimane rida 5×5 maatriksis?")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`mat[indeks]` valib terve rea. Eelviimane = üks enne viimast."),
        "Vihje 2": mo.md("Read on 0,1,2,3,4. Eelviimane on indeksiga **3**."),
        "Täielik lahendus": mo.md("```python\nmat[3]\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_first2 = mo.ui.text_area(value="", label="Harjutus 18: kirjuta avaldis, mis valib esimesed kaks rida", rows=2, full_width=True)
    btn_first2 = mo.ui.button(label="Kontrolli")
    mo.vstack([q_first2, btn_first2])
    return btn_first2, q_first2


@app.cell
def _(btn_first2, mat5, mo, q_first2, review_expr):
    mo.stop(btn_first2.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _r = review_expr(q_first2.value, mat5[:2], local_env={"mat": mat5}, hint="Kasuta slicing'ut `mat[:lõpp]`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`mat[:n]` valib esimesed `n` rida."),
        "Vihje 2": mo.md("Esimesed 2 rida: `mat[:2]` (indeksid 0 ja 1)."),
        "Täielik lahendus": mo.md("```python\nmat[:2]\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8) Agregeerivad tehted: `sum`, `std`, `sum(axis=0)`
    """)
    return


@app.cell
def _(mo):
    n_ag = mo.ui.slider(2, 8, value=5, label="Maatriksi mõõt `n` (n×n)")
    start_ag = mo.ui.slider(1, 10, value=1, label="Algväärtus")
    axis_choice = mo.ui.dropdown(options={"axis=None (kõik elemendid)": "none", "axis=0 (veerud)": "axis0", "axis=1 (read)": "axis1"}, value="axis=None (kõik elemendid)", label="Telg summeerimiseks")
    mo.vstack([n_ag, start_ag, axis_choice])
    return axis_choice, n_ag, start_ag


@app.cell
def _(axis_choice, mo, n_ag, np, plt, start_ag):
    _total = n_ag.value * n_ag.value
    _mat = np.arange(start_ag.value, start_ag.value + _total).reshape(n_ag.value, n_ag.value)
    _s_all = _mat.sum()
    _s0 = _mat.sum(axis=0)
    _s1 = _mat.sum(axis=1)
    _st = _mat.std()
    if axis_choice.value == "none":
        _sum_selected = _s_all
        _axis_text = "sum() = kõik elemendid kokku"
    elif axis_choice.value == "axis0":
        _sum_selected = _s0
        _axis_text = "sum(axis=0) = iga veeru summa"
    else:
        _sum_selected = _s1
        _axis_text = "sum(axis=1) = iga rea summa"
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.imshow(_mat, cmap="YlOrRd")
    for _i in range(n_ag.value):
        for _j in range(n_ag.value):
            _ax1.text(_j, _i, str(_mat[_i, _j]), ha="center", va="center", fontsize=9)
    _ax1.set_title(f"Maatriks {n_ag.value}×{n_ag.value}")
    if axis_choice.value == "axis0":
        _ax2.bar(range(n_ag.value), _s0, color="#4C72B0")
        _ax2.set_title("sum(axis=0) — veergude summad")
        _ax2.set_xlabel("Veeru indeks")
    elif axis_choice.value == "axis1":
        _ax2.barh(range(n_ag.value), _s1, color="#55A868")
        _ax2.set_title("sum(axis=1) — ridade summad")
        _ax2.set_ylabel("Rea indeks")
        _ax2.invert_yaxis()
    else:
        _ax2.text(0.5, 0.5, f"sum() = {_s_all}\nstd() = {_st:.2f}", ha="center", va="center", fontsize=16, transform=_ax2.transAxes)
        _ax2.set_title("Kogu maatriksi statistika")
        _ax2.axis("off")
    _fig.tight_layout()
    _formula = f"""
    ```python
    mat = np.arange({start_ag.value}, {start_ag.value + _total}).reshape({n_ag.value}, {n_ag.value})
    mat.sum()       = {_s_all}       # kõigi elementide summa
    mat.sum(axis=0) = {_s0}  # veergude summad
    mat.sum(axis=1) = {_s1}  # ridade summad
    mat.std()       = {_st:.4f}   # standardhälve
    ```
    """
    mo.vstack([mo.md(_formula), mo.md(f"**{_axis_text}** → `{_sum_selected}`"), _fig])
    return


@app.cell
def _(mo):
    q_matsum = mo.ui.text_area(value="", label="Harjutus 19: kirjuta avaldis, mis arvutab 6×6 maatriksi (10..45) kõigi elementide summa", rows=2, full_width=True)
    btn_matsum = mo.ui.button(label="Kontrolli")
    mo.vstack([q_matsum, btn_matsum])
    return btn_matsum, q_matsum


@app.cell
def _(btn_matsum, mo, np, q_matsum, review_expr):
    mo.stop(btn_matsum.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _mat = np.arange(10, 46).reshape(6, 6)
    _r = review_expr(q_matsum.value, _mat.sum(), local_env={"mat": _mat}, hint="Kasuta `.sum()` ilma argumentideta.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`.sum()` ilma argumentideta summeerib **kõik** elemendid."),
        "Vihje 2": mo.md("`mat.sum()` — annab ühe arvu."),
        "Täielik lahendus": mo.md("```python\nmat.sum()\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_matstd = mo.ui.text_area(value="", label="Harjutus 20: kirjuta avaldis, mis arvutab 6×6 maatriksi (10..45) standardhälbe", rows=2, full_width=True)
    btn_matstd = mo.ui.button(label="Kontrolli")
    mo.vstack([q_matstd, btn_matstd])
    return btn_matstd, q_matstd


@app.cell
def _(btn_matstd, mo, np, q_matstd, review_expr):
    mo.stop(btn_matstd.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _mat = np.arange(10, 46).reshape(6, 6)
    _r = review_expr(q_matstd.value, _mat.std(), local_env={"mat": _mat}, hint="Kasuta `.std()`.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("Standardhälve näitab väärtuste hajuvust keskmise ümber."),
        "Vihje 2": mo.md("`mat.std()` — sarnane `.sum()`-ile."),
        "Täielik lahendus": mo.md("```python\nmat.std()\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    q_sumaxis = mo.ui.text_area(value="", label="Harjutus 21: kirjuta avaldis, mis summeerib 6×6 maatriksi (10..45) veergude kaupa", rows=2, full_width=True)
    btn_sumaxis = mo.ui.button(label="Kontrolli")
    mo.vstack([q_sumaxis, btn_sumaxis])
    return btn_sumaxis, q_sumaxis


@app.cell
def _(btn_sumaxis, mo, np, q_sumaxis, review_expr):
    mo.stop(btn_sumaxis.value == 0, mo.md("*Kirjuta vastus ja vajuta Kontrolli.*"))
    _mat = np.arange(10, 46).reshape(6, 6)
    _r = review_expr(q_sumaxis.value, _mat.sum(axis=0), local_env={"mat": _mat}, hint="Kasuta `axis=0` veergude summeerimiseks.")
    _bullets = "\n".join([f"- {x}" for x in _r["feedback"]])
    _s = "✅" if _r["passed"] else "❌"
    _hints = mo.accordion({
        "Vihje 1": mo.md("`axis` määrab telje suuna."),
        "Vihje 2": mo.md("`axis=0` = summeeri mööda ridu → iga **veeru** summa."),
        "Täielik lahendus": mo.md("```python\nmat.sum(axis=0)\n```"),
    })
    mo.vstack([mo.md(f"### {_s} Review\n{_bullets}"), _hints])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Kokkuvõte — kõik 21 rida

    | # | Avaldis | Selgitus |
    |---|---------|----------|
    | 1 | `import numpy as np` | NumPy alias |
    | 2 | `np.zeros(10)` | 10 nulli |
    | 3 | `np.ones(10)` | 10 ühte |
    | 4 | `np.ones(10) * 5` | elementhaaval korrutamine |
    | 5 | `np.arange(10, 51)` | täisarvud 10..50 |
    | 6 | `np.arange(10, 51, 2)` | paarisarvud |
    | 7 | `np.arange(9).reshape(3,3)` | 3×3 maatriks |
    | 8 | `np.eye(3)` | identiteedimaatriks |
    | 9 | `np.random.rand(1)` | ühtlane jaotus [0,1) |
    | 10 | `np.random.randn(25)` | normaaljaotus |
    | 11 | `np.arange(1,101).reshape(10,10)/100` | skaleeritud maatriks |
    | 12 | `np.linspace(0,1,20)` | võrdsed punktid |
    | 13 | `np.arange(1,26).reshape(5,5)` | 5×5 töömaatriks |
    | 14 | `mat[2:,1:]` | alammaatriks |
    | 15 | `mat[3,4]` | üksik element |
    | 16 | `mat[:3,1].reshape(3,1)` | veerutulp |
    | 17 | `mat[4]` | viimane rida |
    | 18 | `mat[3:]` | viimased read |
    | 19 | `mat.sum()` | elementide summa |
    | 20 | `mat.std()` | standardhälve |
    | 21 | `mat.sum(axis=0)` | veergude summad |
    """)
    return


if __name__ == "__main__":
    app.run()
