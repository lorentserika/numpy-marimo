import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import ast
    import re

    return ast, mo, np, re


@app.cell
def _(mo):
    mo.md("""
    # NumPy interaktiivne õpifail (tudengi kood + automaatne review + juhendamine)

    See versioon sisaldab:
    - ✅ selgitusi
    - ✅ tudengi sisestuskaste (kirjuta oma kood)
    - ✅ automaatseid kontrolle
    - ✅ kommentaare ja vihjeid (reeglipõhine review)

    > Lisasoovitus tudengile: kasuta **terminalis GitHub Copilotit** (või VS Code/PyCharm Copilotit), et küsida oma lahenduse kohta lisakommentaare ja selgitusi.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Tudengile: kuidas saada lisakommentaare GitHub Copilotilt (ilma API-ta)

    Pärast seda, kui oled oma lahenduse siia sisestanud, võid küsida lisaselgitust **terminalis GitHub Copilotilt**.

    ### Näited küsimustest Copilotile
    - "Selgita, miks `np.arange(10)` annab väärtused 0 kuni 9."
    - "Kas minu NumPy lahendus on korrektne? Selgita samm-sammult."
    - "Mis vahe on `np.eye(4)` ja `np.ones((4,4))` vahel?"
    - "Miks `arr.dtype` võib olla mõnes arvutis `int32`, teises `int64`?"

    ### Soovituslik töövoog
    1. Kirjuta lahendus selles notebookis
    2. Vaata automaatset review'd (õige/vale + vihjed)
    3. Kopeeri oma lahendus või error Copilotisse
    4. Küsi *miks* ja *kuidas parandada*

    👉 Nii saad **kiire kontrolli** siit notebookist ja **pikema selgituse** Copilotilt.
    """)
    return


@app.cell
def _(ast, np, re):
    def safe_eval_expr(expr: str, local_env: dict):
        try:
            tree = ast.parse(expr, mode="eval")
            allowed = (
                ast.Expression, ast.Call, ast.Name, ast.Load, ast.Attribute,
                ast.Constant, ast.Tuple, ast.List, ast.Subscript, ast.Slice,
                ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                ast.USub, ast.UnaryOp
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

    def review_arange(student_code: str):
        text = student_code.strip()
        if not text:
            return {"passed": False, "score": 0, "feedback": ["Kirjuta avaldis, nt `np.arange(0,10)`."], "result": None}
        r = safe_eval_expr(text, {})
        if not r["ok"]:
            return {"passed": False, "score": 20, "feedback": [f"Viga: {r['error']}", "Vihje: kasuta `np.arange(...)`."], "result": None}
        expected = np.arange(10)
        try:
            arr = np.asarray(r["value"])
            ok = np.array_equal(arr, expected)
        except Exception:
            ok = False
        if ok:
            fb = ["Õige! Lood massiivi väärtustega 0 kuni 9."]
            compact = re.sub(r"\s+", "", text)
            if compact == "np.arange(0,10)":
                fb.append("Väga hea — täpselt oodatud kuju `np.arange(0,10)`.")
            elif compact == "np.arange(10)":
                fb.append("Õige tulemus — `np.arange(10)` on samuti korrektne ja lühem variant.")
            return {"passed": True, "score": 100, "feedback": fb, "result": pretty(arr)}
        return {"passed": False, "score": 60, "feedback": [f"Oodatud: {pretty(expected)}", f"Sinu tulemus: {pretty(r['value'])}", "Vihje: järjestikuste täisarvude jaoks kasuta `np.arange`."], "result": pretty(r['value'])}

    def review_eye4(student_code: str):
        text = student_code.strip()
        if not text:
            return {"passed": False, "score": 0, "feedback": ["Kirjuta avaldis, nt `np.eye(4)`."], "result": None}
        r = safe_eval_expr(text, {})
        if not r["ok"]:
            return {"passed": False, "score": 20, "feedback": [f"Viga: {r['error']}", "Vihje: identiteedimaatriksi jaoks kasuta `np.eye(...)`."], "result": None}
        expected = np.eye(4)
        try:
            arr = np.asarray(r["value"])
            ok = np.array_equal(arr, expected)
        except Exception:
            ok = False
        if ok:
            return {"passed": True, "score": 100, "feedback": ["Õige! See on 4×4 identiteedimaatriks.", "Peadiagonaalil on 1-d ja mujal 0-d."], "result": pretty(arr)}
        return {"passed": False, "score": 60, "feedback": ["Tulemus ei ole 4×4 identiteedimaatriks.", f"Oodatud:\n{pretty(expected)}", f"Sinu tulemus:\n{pretty(r['value'])}"], "result": pretty(r['value'])}

    def review_rand2(student_code: str):
        text = student_code.strip()
        if not text:
            return {"passed": False, "score": 0, "feedback": ["Kirjuta avaldis, nt `np.random.rand(2)`."], "result": None}
        r = safe_eval_expr(text, {})
        if not r["ok"]:
            return {"passed": False, "score": 20, "feedback": [f"Viga: {r['error']}", "Vihje: `np.random.rand(2)`."], "result": None}
        try:
            arr = np.asarray(r["value"], dtype=float)
            ok = arr.shape == (2,) and np.all(arr >= 0) and np.all(arr < 1)
        except Exception:
            ok = False
        if ok:
            return {"passed": True, "score": 100, "feedback": ["Õige! Tulemus on 2 reaalarvu vahemikus [0,1)."], "result": pretty(arr)}
        return {"passed": False, "score": 60, "feedback": ["Tulemus ei vasta tingimusele: 2 reaalarvu vahemikus [0,1).", "Kontrolli kuju `(2,)` ja vahemikku."], "result": pretty(r.get("value"))}

    def review_dtype(student_code: str):
        text = student_code.strip()
        if not text:
            return {"passed": False, "score": 0, "feedback": ["Kirjuta avaldis, nt `arr.dtype`."], "result": None}
        local_env = {"arr": np.arange(25)}
        r = safe_eval_expr(text, local_env)
        if not r["ok"]:
            return {"passed": False, "score": 20, "feedback": [f"Viga: {r['error']}", "Vihje: kasuta `arr.dtype`."], "result": None}
        exp = local_env["arr"].dtype
        if str(r["value"]) == str(exp):
            return {"passed": True, "score": 100, "feedback": ["Õige! `arr.dtype` tagastab massiivi andmetüübi.", "Märkus: tulemus võib olla `int32` või `int64`."], "result": repr(r["value"])}
        return {"passed": False, "score": 50, "feedback": [f"Oodatud andmetüüp oli `{exp}` (kujul `dtype(...)`).", "Vihje: küsi otse `arr.dtype`."], "result": repr(r['value'])}

    TASKS = [
        {"id":"arange","title":"Massiiv 0 kuni 9","prompt":"Kirjuta avaldis, mis loob massiivi `[0,1,2,3,4,5,6,7,8,9]`.","expected":"np.arange(0,10)","review_fn":review_arange},
        {"id":"eye4","title":"4×4 identiteedimaatriks","prompt":"Kirjuta avaldis, mis loob 4×4 identiteedimaatriksi.","expected":"np.eye(4)","review_fn":review_eye4},
        {"id":"rand2","title":"2 juhuslikku reaalarvu [0,1)","prompt":"Kirjuta avaldis, mis genereerib 2 juhuslikku reaalarvu vahemikus [0,1).","expected":"np.random.rand(2)","review_fn":review_rand2},
        {"id":"dtype","title":"`arr.dtype` kui `arr = np.arange(25)`","prompt":"Kirjuta avaldis, mis tagastab `arr.dtype`, kui `arr = np.arange(25)`.","expected":"arr.dtype","review_fn":review_dtype},
    ]
    return (TASKS,)


@app.cell
def _(TASKS, mo):
    task_selector = mo.ui.dropdown(options={t["title"]: t["id"] for t in TASKS}, value=TASKS[0]["title"], label="Vali ülesanne")
    task_selector
    return (task_selector,)


@app.cell
def _(TASKS, task_selector):
    current_task = next(t for t in TASKS if t["id"] == task_selector.value)
    return (current_task,)


@app.cell
def _(current_task, mo, task_selector):
    defaults = {
        "arange":"np.arange(0,10)",
        "eye4":"np.eye(4)",
        "rand2":"np.random.rand(2)",
        "dtype":"arr.dtype",
    }
    student_code = mo.ui.text_area(
        value=defaults.get(task_selector.value, ""),
        label="Tudengi lahendus (avaldis)",
        rows=4,
        full_width=True,
    )
    mo.vstack([
        mo.md(f"## {current_task['title']}"),
        mo.md(f"**Ülesanne:** {current_task['prompt']}"),
        mo.md("Kirjuta ainult avaldis (mitte `print(...)`)."),
        student_code
    ])
    return (student_code,)


@app.cell
def _(current_task, student_code):
    review = current_task["review_fn"](student_code.value)
    return (review,)


@app.cell
def _(mo, review):
    status = "✅ Korrektne" if review["passed"] else "❌ Vajab parandust"
    bullets = "\n".join([f"- {b}" for b in review["feedback"]])
    res = f"\n\n**Tulemus**\n```python\n{review['result']}\n```" if review.get("result") is not None else ""
    mo.md(f"""### Automaatne review
    **Staatus:** {status}  
    **Punktid:** {review.get("score",0)}/100

    {bullets}{res}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Õpetajale: miks see lahendus on kasulik ilma API-ta?

    - **Automaatne kontroll** annab kiire õige/vale tagasiside
    - **Vihjed** suunavad parandama
    - **GitHub Copilot** saab anda lisakommentaare terminalis/IDE-s ilma eraldi API integratsioonita notebooki sees

    See tähendab:
    - notebook = kontroll + struktuur
    - Copilot = lisaselgitus ja arutelu
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Käivitamine
    ```bash
    pip install marimo numpy
    python -m marimo edit numpy_marimo_review_no_api.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
