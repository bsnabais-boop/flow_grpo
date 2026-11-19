"""Builds the visual companion of the meeting script (same points as the guião), with tables, charts and images.
Numbers are read from the result files; run from anywhere:  python scripts/report/build_visual_guide.py
Output: scripts/report/latent_spark_visual.html (publish it as an artifact)."""
import base64
import glob
import html
import io
import json
import math
import os
import random

import numpy as np
from PIL import Image

ROOT = "/user/home/b.nabais/flow_grpo"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_spark_visual.html")
os.chdir(ROOT)


def jload(p):
    with open(p) as f:
        return json.load(f)


def pt(x, nd=3, sign=False):
    """Portuguese decimal comma."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


# ------------------------------------------------------------------ paired comparisons (as in build_report.py)
def per_prompt(x, pid):
    return np.array([np.nanmean(x[pid == p]) for p in np.unique(pid)])


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def metric_arrays(d, name):
    m = {"CLIP": d["clip"], "PickScore": d["pick"]}
    if name == "ocr":
        m["OCR"] = d["ocr"]
        m["Exact"] = (d["ocr"] >= 1.0).astype(float)
    else:
        ok = (d["clip"][:, None] > d["neg"]).astype(float)
        ok[np.isnan(d["neg"])] = np.nan
        with np.errstate(all="ignore"):
            m["Neg"] = np.nanmean(ok, 1)
    return m


def compare(ref_dir, run_dir):
    out = {}
    for name in ["align", "ocr"]:
        r = np.load(os.path.join(ref_dir, f"{name}.npz"))
        d = np.load(os.path.join(run_dir, f"{name}.npz"))
        rm, dm = metric_arrays(r, name), metric_arrays(d, name)
        for k in rm:
            with np.errstate(all="ignore"):
                diff = per_prompt(dm[k] - rm[k], r["prompt_ids"])
            diff = diff[~np.isnan(diff)]
            lo, hi = boot(diff)
            out[f"{name}.{k}"] = dict(val=float(np.nanmean(dm[k])), diff=float(diff.mean()), lo=lo, hi=hi,
                                      sig=bool(lo > 0 or hi < 0))
    return out


def ckpts(run):
    ds = []
    for d in glob.glob(f"logs/phase3_eval/mixed_{run}_checkpoint-*"):
        if os.path.exists(os.path.join(d, "summary.json")):
            ds.append((int(d.rsplit("-", 1)[1]) // 2, d))
    return sorted(ds)


P3 = {run: [(ep, compare("logs/phase3_eval/base_10steps", d)) for ep, d in ckpts(run)]
      for run in ["prm", "orm", "prm_wsched", "prm_ssched"]}

CONF = {}
for name in ["prm_checkpoint-120", "orm_checkpoint-120", "wsched_checkpoint-60", "wsched_checkpoint-200",
             "ssched_checkpoint-200"]:
    if os.path.exists(f"logs/phase3_confirm/{name}/summary.json"):
        CONF[name] = compare("logs/phase3_confirm/base_10steps", f"logs/phase3_confirm/{name}")
CONF_BASE = {n: jload(f"logs/phase3_confirm/base_{n}steps/summary.json") for n in [10, 20, 50]}


# ------------------------------------------------------------------ images
def jpeg_uri(im, q=82):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def grid_row(path, row, nrows, thumb_out=110):
    """One row of a grid image with square cells and nrows rows; returns the row's cells (last = final image)."""
    g = Image.open(path).convert("RGB")
    cell = g.height // nrows
    ncols = g.width // cell
    out = []
    for c in range(ncols):
        out.append(g.crop((c * cell, row * cell, (c + 1) * cell, (row + 1) * cell)).resize((thumb_out, thumb_out),
                                                                                             Image.LANCZOS))
    return out


def strip(imgs, labels, cls="strip"):
    cells = "".join(f"<figure class='cell'><img src='{jpeg_uri(im)}' alt='{html.escape(l)}'>"
                    f"<figcaption>{l}</figcaption></figure>" for im, l in zip(imgs, labels))
    return f"<div class='{cls}'>{cells}</div>"


def grid_cells(path, n, cell_h=None):
    g = Image.open(path).convert("RGB")
    h = g.height
    return [g.crop((i * h, 0, (i + 1) * h, h)).resize((150, 150), Image.LANCZOS) for i in range(n)]


# ------------------------------------------------------------------ SVG charts
W, H = 600, 260
ML, MR, MT, MB = 52, 16, 14, 44


def line_chart(series, xs, ymin, ymax, yticks, xlab, ylab, xpos=None, xticklabels=None, refs=(), fmt=None,
               zero=False):
    """series: list of dict(ys, cls, lo?, hi?, dash?, label?). xpos maps x to [0,1] (default linear)."""
    fmt = fmt or (lambda v: pt(v, 2))
    xmin, xmax = min(xs), max(xs)
    if xpos is None:
        xpos = lambda x: (x - xmin) / (xmax - xmin) if xmax > xmin else 0.5
    sx = lambda x: ML + xpos(x) * (W - ML - MR)
    sy = lambda y: MT + (1 - (y - ymin) / (ymax - ymin)) * (H - MT - MB)
    o = [f"<svg viewBox='0 0 {W} {H}' role='img' class='chart'>"]
    for t in yticks:
        o.append(f"<line class='grid{' zero' if (zero and abs(t) < 1e-12) else ''}' x1='{ML}' x2='{W - MR}' "
                 f"y1='{sy(t):.1f}' y2='{sy(t):.1f}'/><text class='tick' x='{ML - 6}' y='{sy(t) + 4:.1f}' "
                 f"text-anchor='end'>{fmt(t)}</text>")
    for x, lab in zip(xs, xticklabels or xs):
        o.append(f"<text class='tick' x='{sx(x):.1f}' y='{H - MB + 18}' text-anchor='middle'>{lab}</text>")
    o.append(f"<text class='axl' x='{(ML + W - MR) / 2:.0f}' y='{H - 6}' text-anchor='middle'>{xlab}</text>")
    o.append(f"<text class='axl' transform='translate(13 {(MT + H - MB) / 2:.0f}) rotate(-90)' "
             f"text-anchor='middle'>{ylab}</text>")
    for r in refs:
        y, lab = r
        o.append(f"<line class='ref' x1='{ML}' x2='{W - MR}' y1='{sy(y):.1f}' y2='{sy(y):.1f}'/>"
                 f"<text class='reflab' x='{W - MR - 4}' y='{sy(y) - 5:.1f}' text-anchor='end'>{lab}</text>")
    for s in series:
        pts = [(x, y) for x, y in zip(xs, s["ys"]) if y is not None and not (isinstance(y, float) and math.isnan(y))]
        if s.get("lo") is not None:
            band = [(x, lo, hi) for x, lo, hi in zip(xs, s["lo"], s["hi"]) if lo is not None and not math.isnan(lo)]
            if band:
                up = " ".join(f"{sx(x):.1f},{sy(hi):.1f}" for x, lo, hi in band)
                dn = " ".join(f"{sx(x):.1f},{sy(lo):.1f}" for x, lo, hi in reversed(band))
                o.append(f"<polygon class='band {s['cls']}' points='{up} {dn}'/>")
        if len(pts) > 1:
            o.append(f"<polyline class='ln {s['cls']}{' dash' if s.get('dash') else ''}' points='"
                     + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts) + "'/>")
        for x, y in pts:
            o.append(f"<circle class='dot {s['cls']}' cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3.2'/>")
        if s.get("label") and pts:
            x, y = pts[-1]
            o.append(f"<text class='endlab {s['cls']}' x='{sx(x) - 6:.1f}' y='{sy(y) - 8:.1f}' "
                     f"text-anchor='end'>{s['label']}</text>")
    o.append("</svg>")
    return "".join(o)


def hbars(items, vmax, unit_fmt, w=600):
    """items: (label, value, cls, note)."""
    rowh, lw = 34, 210
    h = rowh * len(items) + 8
    o = [f"<svg viewBox='0 0 {w} {h}' role='img' class='chart'>"]
    for i, (lab, v, cls, note) in enumerate(items):
        y = 4 + i * rowh
        bw = max(2, (w - lw - 90) * v / vmax)
        o.append(f"<text class='blab' x='{lw - 10}' y='{y + 20}' text-anchor='end'>{lab}</text>"
                 f"<rect class='bar {cls}' x='{lw}' y='{y + 6}' width='{bw:.1f}' height='20' rx='2'/>"
                 f"<text class='bval' x='{lw + bw + 8:.1f}' y='{y + 21}'>{unit_fmt(v)}{note}</text>")
    o.append("</svg>")
    return "".join(o)


def legend(items):
    return "<div class='legend'>" + "".join(f"<span><i class='sw {c}'></i>{l}</span>" for l, c in items) + "</div>"


def fig(body, title, text):
    return (f"<figure class='fig'>{body}<figcaption><strong>{title}</strong> {text}</figcaption></figure>")


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


# ================================================================== content
S = []  # sections


def section(num, title, lead, body):
    n = f"<span class='num'>{num}</span>" if num else ""
    S.append(f"<section id='p{num or title[:6]}'><h2>{n}{title}</h2><p class='lead'>{lead}</p>{body}</section>")


# ---- Ponto 1: timeline of the 10 steps
ts = [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]
cells = ""
for i, t in enumerate(ts, 1):
    c = "spark" if i <= 5 else ""
    q = " q" if i in (4, 6, 8, 10) else ""
    cut = " cut" if i == 6 else ""
    cells += f"<div class='st {c}{q}{cut}'><b>{i}</b><span>t={t}</span></div>"
timeline = (f"<div class='steps'>{cells}</div>"
            "<div class='stkey'><span><i class='k spark'></i>metade de muito ruído (a \"faísca latente\", t ∈ [1; 0,5])</span>"
            "<span><i class='k q'></i>passos onde o meu modelo dá nota na Fase 3</span>"
            "<span><i class='k cut'></i>ponto de poda da Fase 2 (depois do passo 6)</span></div>")
section(1, "Mudança de SD 3.5 para SD 1.5",
        "Passei o treino para Stable Diffusion 1.5, como pedido. A conversão linear do NoisyCLIP foi feita para latentes de "
        "4 canais (SD 1.5); o SD 3.5 tem 16. O SD 1.5 prevê ruído (não é flow matching), por isso reescrevi o gerador do "
        "GRPO: um passo de DDIM com ruído e a probabilidade exata de cada passo, LoRA de rank 32, 10 passos no treino.",
        fig(timeline, "Os 10 passos de geração usados em todo o trabalho.",
            "Cada caixa é um passo; t é o nível de ruído (901 = quase só ruído, 1 = imagem pronta). A metade de muito "
            "ruído do plano corresponde aos passos 1 a 5."))

# ---- Ponto 2: OCR GRPO table
rows2 = [
    ["Original: recompensa das pré-visualizações, frases longas", "~225", "14 h", "+0,003", "sem ganho"],
    ["Recompensa da imagem final, prompts curtos (1–2 palavras)", "~375 (8 prompts/época), depois ~79 (32)",
     "2,5 h + 2,5 h", "+0,011", "sem ganho"],
    ["Janela de treino nos passos onde o texto se forma", "~40", "7,6 h", "−0,015 a +0,005 (4 avaliações)", "sem ganho"],
    ["Janela de 4 passos", "~64", "15 h", "−0,002 (ép. 30) / +0,002 (ép. 60)", "sem ganho"],
    ["Clip do PPO maior (1e-4 → 1e-3)", "~32", "7,9 h", "−0,011", "sem ganho"],
    ["1 atualização por época", "~100", "6,8 h", "−0,021 (ép. 30) / −0,002 (ép. 90)", "sem ganho"],
    ["Learning rate maior (3e-4)", "~115", "3,8 h", "+0,0005", "sem ganho"],
]
section(2, "GRPO para texto: recompensa OCR + CLIP",
        "Recompensa = 80 × OCR + CLIPScore. O OCR (PaddleOCR) vale 1 se o texto está exato e 0 se não tem nada a ver. "
        "O CLIP varia só milésimas dentro de um grupo, por isso na prática quem manda é o OCR. "
        "A base é o SD 1.5 original, sem treino, com os mesmos passos e sementes.",
        fig(table(["O que mudei", "Épocas", "Tempo (1 GPU)", "OCR vs base", "Veredito"], rows2),
            "Oito configurações, nenhuma melhorou o texto.",
            "\"OCR vs base\" = OCR médio do modelo treinado menos o da base. Onde há dois valores, são dois checkpoints "
            "verificados de forma independente. Cada run parou quando estagnou, por isso a duração varia.")
        + "<ul class='take'><li><b>O modelo quase não mudava:</b> cada atualização mexia a política ~1e-7 (KL).</li>"
          "<li><b>Raramente havia uma imagem boa no grupo:</b> o SD 1.5 só acerta o texto exato ~2% das vezes.</li>"
          "<li><b>Avaliar mal engana:</b> dois \"ganhos\" durante o treino (z = 2,5 e 1,8) desapareceram na verificação.</li></ul>")

# ---- Ponto 3: per-step previews
s3 = jload("logs/intermediate_eval/win4_base/summary.json")
steps3 = [m["step"] for m in s3["per_step"]]
ocr3 = [m["mean_ocr"] for m in s3["per_step"]]
rank3 = [m["within_group_spearman"] for m in s3["per_step"]]
c3a = line_chart([{"ys": ocr3, "cls": "c2"}], steps3, 0, 0.14, [0, 0.04, 0.08, 0.12], "passo (de 10)",
                 "OCR da pré-visualização", refs=[(s3["final"]["mean_ocr"], "imagem final 0,12")])
c3b = line_chart([{"ys": rank3, "cls": "c1"}], steps3, 0, 1, [0, 0.25, 0.5, 0.75, 1], "passo (de 10)",
                 "acerto na ordenação do grupo")
row3 = grid_row("logs/intermediate_eval/win4_base/grids/prompt_000.png", 0, 8)
strip3 = strip(row3, [f"passo {i}" for i in range(1, 11)] + ["final"])
tab3 = table(["Passo (de 10)", "1–3", "4", "5", "6", "7", "8", "10"],
             [["OCR da pré-visualização", "0,00–0,05", "0,10", "0,12", "0,13", "0,13", "0,13", "0,12"],
              ["Acerto na ordenação do grupo", "0,02–0,14", "0,26", "0,45", "0,60", "0,70", "0,79", "0,93"]])
section(3, "Onde a imagem e o texto se formam",
        "Em cada passo, o modelo estima o ruído e, retirando-o, obtém uma previsão da imagem final (x̂₀). Passei essa "
        "previsão pelo VAE para ter uma pré-visualização e dei-lhe a nota do treino (OCR e CLIP). 128 prompts × 8 imagens.",
        fig(strip3, "A mesma imagem vista em cada passo.",
            "Prompt: <i>A book cover with the title \"GAME OVER\" in large serif font</i>. Nos passos 1–3 só há borrões; "
            "a partir do 4–5 a estrutura e as letras já estão no sítio; depois só se afinam detalhes.")
        + "<div class='two'>"
        + fig(c3a, "O texto fica legível cedo.", "O OCR da pré-visualização chega ao valor final no passo 5.")
        + fig(c3b, "Mas ordenar as imagens do grupo exige mais passos.",
              "Concordância (Spearman) entre a ordem das 8 imagens pela pré-visualização e pela imagem final. "
              "Com o VAE só passa de 0,6 no passo 6.")
        + "</div>"
        + fig(tab3, "Os mesmos números em tabela.",
              "A ordenação usa a nota total (80 × OCR + CLIP), que na prática segue o OCR.")
        + "<p class='note'>Cada pré-visualização com VAE + OCR custa ~1,4 passos da UNet: é este o custo que a Fase 1 "
          "quer eliminar.</p>")

# ---- Ponto 4: fewer steps
NS = [4, 6, 8, 10, 20, 50]
s4 = {n: jload(f"logs/intermediate_eval/h200_eta0_{n}steps/summary.json")["final"] for n in NS}
logpos = lambda x: (math.log(x) - math.log(4)) / (math.log(50) - math.log(4))
c4a = line_chart([{"ys": [s4[n]["mean_ocr"] for n in NS], "cls": "c2"}], NS, 0, 0.2, [0, 0.05, 0.1, 0.15, 0.2],
                 "número de passos (escala log)", "OCR", xpos=logpos)
c4b = line_chart([{"ys": [s4[n]["mean_clip"] for n in NS], "cls": "c1"}], NS, 0.8, 1.0, [0.8, 0.85, 0.9, 0.95, 1.0],
                 "número de passos (escala log)", "CLIP (escala do treino)", xpos=logpos)
imgs4 = [grid_row(f"logs/intermediate_eval/h200_eta0_{n}steps/grids/prompt_000.png", 0, 8)[-1] for n in NS]
strip4 = strip(imgs4, [f"{n} passos" for n in NS])
tab4 = table(["Passos", "4", "6", "8", "10", "20", "50"],
             [["OCR"] + [pt(s4[n]["mean_ocr"]) for n in NS],
              ["% do OCR a 50 passos"] + [f"{100 * s4[n]['mean_ocr'] / s4[50]['mean_ocr']:.0f}%" for n in NS],
              ["CLIP"] + [pt(s4[n]["mean_clip"]) for n in NS],
              ["Texto exato"] + [f"{100 * s4[n]['exact_ocr_frac']:.0f}%" for n in NS]])
section(4, "Quanto se perde com menos passos",
        "Para que serviu: medir quanto o modelo original perde com menos passos, antes de qualquer treino. É o ponto de "
        "partida e a régua do objetivo 40–50 → 10–15 passos. SD 1.5 base, 1024 imagens por linha, H200.",
        fig(strip4, "O mesmo prompt e a mesma semente, com 4 a 50 passos.",
            "Prompt: <i>A book cover with the title \"GAME OVER\" in large serif font</i>. Com poucos passos a imagem "
            "fica pobre; o texto é o que mais demora a ficar certo.")
        + "<div class='two'>"
        + fig(c4a, "O texto continua a melhorar até aos 50 passos.", "A 10 passos fica-se com 61% do OCR de 50.")
        + fig(c4b, "O alinhamento estabiliza cedo.", "O CLIP quase não muda a partir dos 10–20 passos.")
        + "</div>" + fig(tab4, "Tabela completa.", "")
        + "<p class='concl'><b>Conclusão:</b> com menos passos o SD 1.5 mantém quase todo o alinhamento, mas perde muito "
          "texto. É essa diferença que o treino tem de fechar.</p>")

# ---- Ponto 5: Phase 1 scorer
s5 = jload("logs/phase1_noisyclip/eval10_v2_lat_10steps_eta1/summary.json")
steps5 = [p["step"] for p in s5["per_step"]]
c5 = line_chart([{"ys": [p["trained_spearman"] for p in s5["per_step"]], "cls": "c1", "label": "o meu modelo"},
                 {"ys": [p["taesd_spearman"] for p in s5["per_step"]], "cls": "c2", "label": "TAESD + CLIP"},
                 {"ys": [p["proj_x0_spearman"] for p in s5["per_step"]], "cls": "c4", "label": "Passo A (Φ + CLIP)",
                  "dash": True}],
                steps5, -0.1, 1, [0, 0.25, 0.5, 0.75, 1], "passo (de 10)", "correlação com o CLIP final", zero=True)
cost5 = hbars([("Passo A (Φ + CLIP)", 0.10, "c4", ""), ("O meu modelo (v2)", 0.16, "c1", ""),
               ("TAESD + CLIP", 0.31, "c2", ""), ("VAE + CLIP", 1.78, "c3", "  (11× o meu)")], 1.78,
              lambda v: pt(v, 2))
row5 = grid_row("logs/phase1_noisyclip/eval10_v2_lat_10steps_eta1/grids/batch_000.png", 0, 8, 100)
strip5 = strip(row5, [f"passo {i}" for i in range(1, 11)] + ["final (VAE)"])
diagram5 = """<svg viewBox='0 0 640 256' role='img' class='chart diag'>
<text class='dh' x='10' y='18'>Treino</text>
<rect class='box' x='10' y='30' width='120' height='50' rx='6'/><text class='bt' x='70' y='52' text-anchor='middle'>imagem a meio</text><text class='bs' x='70' y='68' text-anchor='middle'>x̂₀ do passo t</text>
<rect class='box' x='165' y='30' width='130' height='50' rx='6'/><text class='bt' x='230' y='52' text-anchor='middle'>Φ + mini-rede</text><text class='bs' x='230' y='68' text-anchor='middle'>imagem barata</text>
<rect class='box me' x='330' y='30' width='130' height='50' rx='6'/><text class='bt' x='395' y='52' text-anchor='middle'>CLIP + LoRA</text><text class='bs' x='395' y='68' text-anchor='middle'>o aluno</text>
<rect class='box' x='495' y='30' width='135' height='50' rx='6'/><text class='bt' x='562' y='52' text-anchor='middle'>vetor previsto</text><text class='bs' x='562' y='68' text-anchor='middle'>da imagem final</text>
<rect class='box teach' x='495' y='120' width='135' height='50' rx='6'/><text class='bt' x='562' y='142' text-anchor='middle'>vetor do CLIP</text><text class='bs' x='562' y='158' text-anchor='middle'>da imagem final</text>
<rect class='box teach' x='330' y='120' width='130' height='50' rx='6'/><text class='bt' x='395' y='142' text-anchor='middle'>CLIP normal</text><text class='bs' x='395' y='158' text-anchor='middle'>o professor</text>
<rect class='box' x='165' y='120' width='130' height='50' rx='6'/><text class='bt' x='230' y='142' text-anchor='middle'>imagem final</text><text class='bs' x='230' y='158' text-anchor='middle'>descodificada (VAE)</text>
<path class='ar' d='M130 55 H163'/><path class='ar' d='M295 55 H328'/><path class='ar' d='M460 55 H493'/>
<path class='ar' d='M295 145 H328'/><path class='ar' d='M460 145 H493'/>
<path class='ar dsh' d='M562 82 V118'/><text class='bs' x='572' y='104'>aprende a aproximar</text>
<text class='dh' x='10' y='210'>Uso</text>
<text class='bs' x='10' y='230'>nota = semelhança entre o vetor previsto e o vetor do texto do prompt</text><text class='bs' x='10' y='246'>= um CLIPScore previsto da imagem final, sem VAE</text>
</svg>"""
tab5 = table(["Método", "Passo 4", "Passo 5", "Passo 6", "Custo (passos da UNet)"],
             [["<b>O meu modelo (v2)</b>"] + [f"<b>{pt(s5['per_step'][i]['trained_spearman'], 2)}</b>" for i in (3, 4, 5)] + ["<b>0,16</b>"],
              ["TAESD (VAE pequeno) + CLIP"] + [pt(s5['per_step'][i]['taesd_spearman'], 2) for i in (3, 4, 5)] + ["0,31"],
              ["Passo A (Φ + CLIP, sem treino)"] + [pt(s5['per_step'][i]['proj_x0_spearman'], 2) for i in (3, 4, 5)] + ["0,10"],
              ["VAE + CLIP", "—", "—", "—", "1,78"]])
tab5b = table(["Caminho", "Como obtém a imagem", "Custo (passos da UNet)"],
              [["VAE + CLIP", "Descodificador completo do SD 1.5 (512×512)", "1,78"],
               ["TAESD + CLIP", "Descodificador pequeno, treinado para imitar o VAE", "0,31"],
               ["O meu modelo (v2)", "Φ + mini-rede própria (256×256), sem descodificador", "0,16"]])
section(5, "Fase 1: um avaliador que dá nota a meio da geração",
        "Objetivo: saber a meio do caminho se a imagem vai ficar alinhada com o prompt, sem VAE. O \"NoisyCLIP\" do "
        "repositório era um modelo aleatório, por isso construí o meu em dois passos. <b>Passo A:</b> converter x̂₀ em "
        "cores com Φ (15 números fixos, ajustados por regressão) e passar pelo CLIP. <b>Passo B:</b> treinar o CLIP com "
        "uma LoRA para, a partir da imagem a meio, prever a nota que o CLIP dará à imagem final. A versão final (v2) lê "
        "também o latente com uma pequena rede e foi retreinada no cenário do GRPO (10 passos com ruído).",
        fig(strip5, "O que o meu modelo \"vê\": a imagem barata Φ(x̂₀) em cada passo.",
            "Prompt: <i>Two joggers running along a misty riverbank at dawn</i>. É pequena e grosseira, mas a cena "
            "aparece logo no passo 3. A última coluna é a imagem final descodificada pelo VAE, para comparar.")
        + fig(diagram5, "Professor e aluno.",
              "O CLIP normal é o professor: dá nota à imagem final. O meu modelo é o aluno: vê a imagem a meio e aprende "
              "a adivinhar essa nota. Só a LoRA, a rede do passo e a mini-rede são treinadas.")
        + fig(legend([("o meu modelo (v2)", "c1"), ("TAESD + CLIP", "c2"), ("Passo A, sem treino", "c4")]) + c5,
              "O meu modelo prevê melhor o resultado na primeira metade.",
              "Correlação (Spearman) entre a nota em cada passo e o CLIP da imagem final. 128 prompts nunca vistos × 4 "
              "imagens, 10 passos com ruído. No fim o TAESD alcança-o, porque a imagem já está nítida.")
        + fig(cost5, "E custa muito menos que descodificar.", "Custo de uma nota, em passos da UNet (1 = um passo de geração).")
        + fig(tab5, "Resumo em tabela.", "Correlação com o CLIP da imagem final em cada passo.")
        + fig(tab5b, "Como cada caminho obtém a imagem.",
              "O VAE só é usado uma vez, no fim, para a imagem que se entrega. O TAESD entra apenas como comparação.")
        + "<p class='note'><b>Limite:</b> com erros subtis (contagem, cor, relações) o próprio CLIP na imagem final só "
          "acerta 81%, e o meu modelo chega a 78%. O teto passa a ser o professor.</p>")

# ---- Ponto 6: pruning
traj = []
import random as _r
rng = _r.Random(3)
finals = [0.9, 0.82, 0.78, 0.55, 0.45, 0.35, 0.3, 0.2]
pruned = [False, False, False, True, True, True, True, True]
paths = ""
for k, (fv, pr) in enumerate(zip(finals, pruned)):
    ys = []
    for s in range(11):
        f = s / 10
        ys.append(0.5 + (fv - 0.5) * (f ** 0.8) + (rng.random() - 0.5) * 0.06 * (1 - f))
    last = 6 if pr else 10
    pts_ = " ".join(f"{50 + s * 52:.0f},{200 - ys[s] * 170:.1f}" for s in range(last + 1))
    paths += f"<polyline class='ln {'c3 faint' if pr else 'c1'}' points='{pts_}'/>"
    xe, ye = 50 + last * 52, 200 - ys[last] * 170
    paths += (f"<text class='x' x='{xe:.0f}' y='{ye + 4:.1f}' text-anchor='middle'>×</text>" if pr else
              f"<circle class='dot c1' cx='{xe:.0f}' cy='{ye:.1f}' r='3.5'/>")
ticks6 = "".join(f"<text class='tick' x='{50 + s * 52}' y='222' text-anchor='middle'>{s}</text>" for s in range(0, 11, 2))
diagram6 = (f"<svg viewBox='0 0 620 240' role='img' class='chart'>"
            f"<line class='ref' x1='{50 + 6 * 52}' x2='{50 + 6 * 52}' y1='10' y2='205'/>"
            f"<text class='reflab' x='{50 + 6 * 52 + 6}' y='20'>verificação no passo 6</text>{paths}{ticks6}"
            f"<text class='axl' x='310' y='238' text-anchor='middle'>passo</text>"
            f"<text class='axl' transform='translate(16 110) rotate(-90)' text-anchor='middle'>nota do meu modelo</text></svg>")
save6 = hbars([("O meu modelo", 22, "c1", ""), ("TAESD + CLIP", 14, "c2", "  (10–18% conforme o caso)"),
               ("VAE + CLIP", 3, "c3", "  (≤ 5%, por vezes negativo)")], 25, lambda v: f"{v:.0f}%")
tab6 = table(["Risco controlado (α = 0,1)", "Grupo de 4", "Grupo de 8"],
             [["Nunca perder a melhor imagem", "14% poupado", "17% poupado"],
              ["Manter uma imagem a < 0,005 da melhor", "—", "<b>22% poupado</b> (58% paradas no passo 6)"],
              ["Manter uma imagem a < 0,01 da melhor", "—", "24% poupado"]])
section(6, "Fase 2: parar cedo as trajetórias más",
        "No GRPO geram-se 8 imagens por prompt. Se a meio o meu modelo já sabe que uma vai ficar claramente atrás da "
        "melhor do grupo, paro-a e poupo os passos que faltam. O limiar é escolhido com Conformal Risk Control, para "
        "o risco de perder a boa imagem ficar ≤ 10%.",
        fig(diagram6, "Como funciona (esquema ilustrativo).",
            "Linhas = 8 imagens do mesmo prompt. No passo 6, as que estão demasiado atrás da líder são paradas (×); "
            "só as outras chegam ao passo 10.")
        + fig(save6, "Computação poupada (grupos de 8, risco 10%).",
              "Com o meu modelo poupa-se ~22%; com descodificadores a nota custa tanto que a poupança desaparece.")
        + fig(tab6, "Resultados por tipo de risco.",
              "128 prompts × 8 imagens, 10 passos; calibrado em metade dos prompts e testado na outra, 40 vezes. O risco "
              "medido no teste ficou em 0,09–0,10, ou seja, a garantia funciona.")
        + "<p class='note'>O melhor ponto de poda é o passo 6 de 10. Podar em vários pontos não ajudou.</p>")

# ---- Ponto 7: Phase 3
def series(run, key):
    xs = [ep for ep, _ in P3[run]]
    return xs, [r[key]["diff"] for _, r in P3[run]], [r[key]["lo"] for _, r in P3[run]], [r[key]["hi"] for _, r in P3[run]]


def p3_chart(runs, key, ymin, ymax, yt, ylab, nd=2):
    xs_all = sorted(set().union(*[[ep for ep, _ in P3[r]] for r, _, _ in runs]))
    ss = []
    for r, cls, band in runs:
        m = {ep: v for ep, v in zip(*series(r, key)[:2])}
        lo = {ep: v for ep, v in zip(series(r, key)[0], series(r, key)[2])}
        hi = {ep: v for ep, v in zip(series(r, key)[0], series(r, key)[3])}
        s = {"ys": [m.get(x, float("nan")) for x in xs_all], "cls": cls}
        if band:
            s["lo"] = [lo.get(x, float("nan")) for x in xs_all]
            s["hi"] = [hi.get(x, float("nan")) for x in xs_all]
        ss.append(s)
    return line_chart(ss, xs_all, ymin, ymax, yt, "época de treino", ylab, zero=True,
                      fmt=lambda v: pt(v, nd, sign=True) if v else "0")


c7a = p3_chart([("orm", "c3", True), ("prm", "c1", True)], "ocr.OCR", -0.03, 0.05, [-0.02, 0, 0.02, 0.04],
               "OCR vs base")
c7b = p3_chart([("orm", "c3", True), ("prm", "c1", True)], "align.PickScore", -0.005, 0.004,
               [-0.004, -0.002, 0, 0.002, 0.004], "PickScore vs base", nd=3)
rows7 = []
for name, lab in [("prm_checkpoint-120", "PRM (o meu modelo), época 60"), ("orm_checkpoint-120", "ORM (CLIP), época 60")]:
    r = CONF[name]
    rows7.append([lab, pt(r["ocr.OCR"]["val"]), pt(r["ocr.OCR"]["diff"], 3, True) + (" *" if r["ocr.OCR"]["sig"] else " (n.s.)"),
                  pt(r["align.PickScore"]["diff"], 4, True) + (" *" if r["align.PickScore"]["sig"] else " (n.s.)")])
b10 = CONF_BASE[10]
for n in [10, 20, 50]:
    b = CONF_BASE[n]
    rows7.append([f"Base, {n} passos", pt(b["ocr"]["ocr"]),
                  "—" if n == 10 else pt(b["ocr"]["ocr"] - b10["ocr"]["ocr"], 3, True) + " *",
                  "—" if n == 10 else pt(b["align"]["pickscore"] - b10["align"]["pickscore"], 4, True) + " *"])
tab7 = table(["Modelo (10 passos, salvo indicação)", "OCR", "OCR vs base 10", "PickScore vs base 10"], rows7)
imgs7 = (grid_cells("logs/phase3_eval/base_10steps/grid_ocr.png", 5) + grid_cells("logs/phase3_eval/mixed_prm_checkpoint-120/grid_ocr.png", 5)
         + grid_cells("logs/phase3_eval/base_50steps/grid_ocr.png", 5))
g7 = ("<div class='imgrid'>" + "".join(
    f"<div class='rowlab'>{lab}</div>" + "".join(f"<img src='{jpeg_uri(im)}' alt='{lab}'>" for im in imgs7[k * 5:(k + 1) * 5])
    for k, lab in enumerate(["Base, 10 passos", "PRM, época 60", "Base, 50 passos"])) + "</div>")
section(7, "Fase 3: GRPO com o meu avaliador vs com CLIP",
        "Foco no alinhamento sem perder texto: 1440 prompts do GenAI-Bench + 1088 de texto. Recompensa = 100 × alinhamento "
        "+ 80 × OCR. Duas runs iguais exceto no alinhamento: <b>PRM</b> (o meu avaliador nos passos 4, 6, 8, 10, sem VAE) e "
        "<b>ORM</b> (CLIP na imagem final). 10 passos, 32 prompts × 8 imagens por época, 100 épocas (~4,5 h).",
        fig(legend([("PRM (o meu avaliador)", "c1"), ("ORM (CLIP na imagem final)", "c3"), ("sombra = intervalo de 95%", "c4")])
            + "<div class='two'>" + c7a + c7b + "</div>",
            "Durante o treino: diferença para a base em cada checkpoint.",
            "512 + 512 imagens, sempre as mesmas sementes. O PRM sobe no OCR em alguns checkpoints; o ORM fica no zero e "
            "desce no PickScore. Com tantos checkpoints, um pico isolado pode ser sorte, por isso há a confirmação abaixo.")
        + fig(tab7, "Confirmação com sementes novas (1024 + 1024 imagens).",
              "* = significativo (intervalo de 95% não inclui zero); n.s. = não significativo. O ganho de +0,026 visto "
              "durante o treino não se confirmou: foi o melhor de 12 checkpoints, um efeito de seleção.")
        + fig(g7, "Exemplos: mesmo prompt e mesma semente em cada coluna.",
              "Prompts (texto pedido): TICKETS (placa de rua), TICKETS (bolo), TOP SECRET (placa de madeira), SALE NOW "
              "(faixa), KEEP OUT (quadro de giz). Diferenças pequenas entre base e PRM; os 50 passos ajudam mais o texto.")
        + "<p class='concl'><b>Conclusão:</b> o treino funciona e a política já se mexe (KL ~2e-4), mas ainda não há "
          "ganho confirmado. O alinhamento não piorou.</p>")

# ---- Ponto 8: curricula
eps = list(range(0, 101, 5))
w_al = [100 * (1 - 0.5 * min(1, e / 60)) for e in eps]
w_ocr = [80 * min(1, e / 60) for e in eps]
c8w = line_chart([{"ys": w_al, "cls": "c1", "label": "peso do alinhamento"},
                  {"ys": w_ocr, "cls": "c2", "label": "peso do OCR"}], eps, 0, 100, [0, 25, 50, 75, 100],
                 "época", "peso", fmt=lambda v: f"{v:.0f}")
st = [4 if e < 20 else 6 if e < 40 else 8 if e < 60 else 10 for e in eps]
c8s = line_chart([{"ys": st, "cls": "c4", "label": "passos de geração"}], eps, 0, 12, [0, 4, 6, 8, 10],
                 "época", "passos", fmt=lambda v: f"{v:.0f}")
runs8 = [("prm", "c1", False), ("prm_wsched", "c2", True)] + ([("prm_ssched", "c4", False)] if P3["prm_ssched"] else [])
c8 = p3_chart(runs8, "ocr.OCR", -0.02, 0.05, [-0.02, 0, 0.02, 0.04], "OCR vs base")
c8p = p3_chart(runs8, "align.PickScore", -0.004, 0.003, [-0.004, -0.002, 0, 0.002], "PickScore vs base", nd=3)
tab8 = table(["Run", "Pesos (alinhamento / OCR)", "Passos no treino", "Duração", "Estado"],
             [["Curriculum de pesos", "rampa em 60 épocas: 100 → 50 / 0 → 80", "10", "100 épocas, 4,5 h", "terminou"],
              ["Curriculum de passos", "100 / 80 fixos", "4 → 6 → 8 → 10 (20 épocas cada)", "100 épocas, 3,5 h", "terminou"],
              ["Os dois juntos", "rampa", "4 → 10", "100 épocas", "a correr"]])
w_conf = [n for n in CONF if n.startswith(("wsched", "ssched"))]
conf8 = ("<p class='note'>Confirmação com sementes novas (curriculum de pesos épocas 30 e 100, de passos época 100): "
         "na fila.</p>" if not w_conf else
         fig(table(["Modelo", "OCR vs base 10", "PickScore vs base 10"],
                   [[n, pt(CONF[n]["ocr.OCR"]["diff"], 3, True) + (" *" if CONF[n]["ocr.OCR"]["sig"] else " (n.s.)"),
                     pt(CONF[n]["align.PickScore"]["diff"], 4, True) + (" *" if CONF[n]["align.PickScore"]["sig"] else " (n.s.)")]
                    for n in w_conf]), "Confirmação dos curricula com sementes novas.", ""))
leg8 = [("PRM sem curriculum", "c1"), ("curriculum de pesos (com intervalo de 95%)", "c2")] + \
       ([("curriculum de passos", "c4")] if P3["prm_ssched"] else [])
section(8, "Os curricula sugeridos pelo orientador",
        "(1) Mudar os pesos do alinhamento e do OCR ao longo do treino. (2) Aumentar o número de passos ao longo do treino. "
        "Três runs iguais à PRM exceto no curriculum, uma de cada vez. Os passos começam em 4 e não em 2: com 2 passos o "
        "SD 1.5 não gera nada utilizável.",
        "<div class='two'>"
        + fig(c8w, "Curriculum de pesos.", "Primeiro o alinhamento, depois o texto entra aos poucos; o alinhamento nunca vai a zero.")
        + fig(c8s, "Curriculum de passos.", "4 → 6 → 8 → 10 passos, 20 épocas em cada patamar.")
        + "</div>"
        + fig(tab8, "As três runs.", "Todas avaliadas a 10 passos, com as mesmas sementes da base.")
        + fig(legend(leg8) + "<div class='two'>" + c8 + c8p + "</div>", "Resultados durante o treino.",
              "O curriculum de pesos fica acima de zero no OCR em todos os checkpoints (significativo nas épocas 30, 40 e "
              "90), com o PickScore a voltar a ~0 no fim. "
              + ("" if P3["prm_ssched"] else "Os checkpoints do curriculum de passos ainda estão a ser avaliados."))
        + conf8
        + "<p class='concl'><b>Conclusão:</b> o curriculum de pesos é o resultado mais promissor, mas só conta depois da "
          "confirmação com sementes novas.</p>")

# ---- verification
section("", "Como verifico se um resultado é real",
        "Só digo que algo melhorou depois de uma verificação independente. Aprendi isto depois de três falsos ganhos.",
        "<ol class='steps-list'><li><b>Emparelhado:</b> modelo e base geram com as mesmas sementes.</li>"
        "<li><b>Sementes novas</b>, nunca usadas para escolher checkpoints.</li>"
        "<li><b>Mais imagens:</b> 1024 por conjunto (128 prompts × 8).</li>"
        "<li><b>A mesma GPU:</b> H200 e A100 geram imagens diferentes com a mesma semente.</li>"
        "<li><b>Intervalos de confiança de 95%</b> por bootstrap; só conta se não incluir o zero.</li></ol>")

# ---- glossary
gl = [("GRPO", "Gero um grupo de imagens por prompt; o modelo aprende a fazer mais como as melhores e menos como as piores."),
      ("PRM / ORM", "Avaliador de processo (dá nota a meio) / de resultado (só à imagem final)."),
      ("x̂₀", "A previsão da imagem final que o modelo faz em cada passo."),
      ("VAE / TAESD", "Descodificador do latente para imagem (caro) / versão pequena e rápida."),
      ("Φ (Phi)", "Conversão linear fixa dos 4 canais do latente para RGB (15 números)."),
      ("LoRA", "Pequenas matrizes treináveis por cima de um modelo congelado."),
      ("CLIPScore / PickScore", "Nota de correspondência imagem–texto / outro juiz, independente do treino."),
      ("Spearman", "Correlação de ordenações: 1 = ordena exatamente como o resultado final."),
      ("Conformal Risk Control", "Escolhe um limiar com garantia estatística de risco ≤ α."),
      ("KL", "Quanto o modelo mudou numa atualização.")]
section("", "Glossário", "Os termos em uma frase.", table(["Termo", "Significado"], [[a, b] for a, b in gl]))

# ================================================================== page
CSS = """
/* Layout: one reading column; each point is a section with lead text, then captioned figures. */
:root{--bg:#f6f7f8;--panel:#ffffff;--fg:#1c2329;--mut:#5b6872;--line:#d9dee2;--c1:#0f6e78;--c2:#b7791f;--c3:#b4442f;--c4:#7a8791;--band:.16;
--display:"Space Grotesk",system-ui,sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12171b;--panel:#1a2026;--fg:#e4e9ec;--mut:#9aa7b0;--line:#2e373e;--c1:#4fb3bd;--c2:#e0a84a;--c3:#e8826d;--c4:#97a3ac;--band:.22;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#12171b;--panel:#1a2026;--fg:#e4e9ec;--mut:#9aa7b0;--line:#2e373e;--c1:#4fb3bd;--c2:#e0a84a;--c3:#e8826d;--c4:#97a3ac;--band:.22;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--body)}
.wrap{max-width:900px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
header h1{font:600 2rem/1.15 var(--display);margin:0 0 6px;text-wrap:balance}
header p{color:var(--mut);margin:0;max-width:65ch}
nav{display:flex;flex-wrap:wrap;gap:6px 14px;margin:18px 0 8px;font-size:.85rem}
nav a{color:var(--c1);text-decoration:none}nav a:hover{text-decoration:underline}
section{margin-top:44px;padding-top:20px;border-top:1px solid var(--line)}
h2{font:600 1.35rem/1.25 var(--display);margin:0 0 8px;display:flex;gap:10px;align-items:baseline;text-wrap:balance}
.num{font:500 .8rem var(--mono);color:var(--c1);border:1px solid var(--c1);border-radius:4px;padding:1px 6px;flex:none}
.lead{max-width:70ch;margin:0 0 14px}
.fig{margin:18px 0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;min-width:0}
.fig figcaption{font-size:.86rem;color:var(--mut);margin-top:8px;max-width:75ch}
.fig figcaption strong{color:var(--fg)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.two>.fig{margin:0}
.chart{width:100%;height:auto;display:block}
.chart text{fill:var(--mut);font:11px var(--body)}
.chart .tick{font:10.5px var(--mono)}
.chart .axl{font-size:11px}
.grid{stroke:var(--line);stroke-width:1}.grid.zero{stroke:var(--mut);stroke-width:1.2}
.ref{stroke:var(--mut);stroke-dasharray:4 4}.chart .reflab{fill:var(--mut);font-size:10.5px}
.ln{fill:none;stroke-width:2.2}.ln.dash{stroke-dasharray:5 4}.ln.faint{opacity:.55}
.c1.ln{stroke:var(--c1)}.c2.ln{stroke:var(--c2)}.c3.ln{stroke:var(--c3)}.c4.ln{stroke:var(--c4)}
.c1.dot{fill:var(--c1)}.c2.dot{fill:var(--c2)}.c3.dot{fill:var(--c3)}.c4.dot{fill:var(--c4)}
.band{opacity:var(--band);stroke:none}.c1.band{fill:var(--c1)}.c2.band{fill:var(--c2)}.c3.band{fill:var(--c3)}.c4.band{fill:var(--c4)}
.chart .endlab{font-weight:600}.chart .endlab.c1{fill:var(--c1)}.chart .endlab.c2{fill:var(--c2)}.chart .endlab.c3{fill:var(--c3)}.chart .endlab.c4{fill:var(--c4)}
.bar.c1{fill:var(--c1)}.bar.c2{fill:var(--c2)}.bar.c3{fill:var(--c3)}.bar.c4{fill:var(--c4)}
.chart .blab{fill:var(--fg);font-size:12px}.chart .bval{fill:var(--fg);font:11.5px var(--mono)}
.chart .x{fill:var(--c3);font-size:15px;font-weight:700}
.diag .box{fill:var(--panel);stroke:var(--line);stroke-width:1.4}.diag .box.me{stroke:var(--c1);stroke-width:2}.diag .box.teach{stroke:var(--c2);stroke-width:2}
.diag .bt{fill:var(--fg);font-size:12.5px;font-weight:600}.diag .bs{fill:var(--mut);font-size:11px}.diag .dh{fill:var(--fg);font:600 12px var(--display);letter-spacing:.06em;text-transform:uppercase}
.ar{stroke:var(--mut);stroke-width:1.5;fill:none;marker-end:url(#a)}.ar.dsh{stroke-dasharray:4 3;stroke:var(--c1)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.82rem;color:var(--mut);margin-bottom:6px}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:14px;height:4px;border-radius:2px;display:inline-block}
.sw.c1{background:var(--c1)}.sw.c2{background:var(--c2)}.sw.c3{background:var(--c3)}.sw.c4{background:var(--c4)}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;font-size:.78rem;letter-spacing:.03em;color:var(--mut);text-transform:uppercase}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(72px,1fr));gap:4px}
.strip .cell{margin:0}.strip img{width:100%;display:block;border-radius:3px}
.strip figcaption{font:10.5px var(--mono);color:var(--mut);text-align:center;margin-top:2px}
.imgrid{display:grid;grid-template-columns:110px repeat(5,1fr);gap:4px;align-items:center;overflow-x:auto}
.imgrid img{width:100%;display:block;border-radius:3px;min-width:56px}.rowlab{font-size:.8rem;color:var(--mut)}
.steps{display:grid;grid-template-columns:repeat(10,1fr);gap:4px}
.st{border:1px solid var(--line);border-radius:5px;padding:8px 2px;text-align:center;background:var(--panel);display:flex;flex-direction:column}
.st b{font:600 1rem var(--display)}.st span{font:10px var(--mono);color:var(--mut)}
.st.spark{background:color-mix(in srgb,var(--c2) 18%,var(--panel))}
.st.q{outline:2px solid var(--c1);outline-offset:-2px}.st.cut{border-right:4px solid var(--c3)}
.stkey{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.8rem;color:var(--mut);margin-top:10px}
.stkey span{display:inline-flex;gap:6px;align-items:center}.k{width:14px;height:14px;border-radius:3px;display:inline-block;border:1px solid var(--line)}
.k.spark{background:color-mix(in srgb,var(--c2) 30%,var(--panel))}.k.q{border:2px solid var(--c1)}.k.cut{border-right:4px solid var(--c3)}
.take{margin:10px 0 0;padding-left:20px;max-width:75ch}.take li{margin:4px 0}
.note{font-size:.9rem;color:var(--mut);max-width:75ch}
.concl{border-left:3px solid var(--c1);padding:6px 12px;background:var(--panel);max-width:75ch}
.steps-list li{margin:5px 0}
@media (max-width:520px){.steps{grid-template-columns:repeat(5,1fr)}.imgrid{grid-template-columns:70px repeat(5,1fr)}}
"""

MARKER = ("<svg width='0' height='0' style='position:absolute'><defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' "
          "markerWidth='7' markerHeight='7' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' style='fill:var(--mut)'/>"
          "</marker></defs></svg>")
nav = "".join(f"<a href='#p{i}'>{i}. {t}</a>" for i, t in [(1, "SD 1.5"), (2, "OCR"), (3, "Passos"), (4, "Menos passos"),
                                                          (5, "Fase 1"), (6, "Fase 2"), (7, "Fase 3"), (8, "Curricula")])
page = f"""<title>Latent Spark em imagens</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>{CSS}</style>
{MARKER}
<div class="wrap">
<header><h1>Latent Spark em imagens</h1>
<p>Os mesmos pontos do guião da reunião, com as tabelas, gráficos e imagens de cada um. SD 1.5, 6–11 de novembro de 2026.
Todos os números vêm dos ficheiros de resultados.</p></header>
<nav>{nav}</nav>
{''.join(S)}
</div>"""
with open(OUT, "w") as f:
    f.write(page)
print(f"wrote {OUT} {os.path.getsize(OUT) // 1024} KB; prm {[e for e, _ in P3['prm']]} wsched {[e for e, _ in P3['prm_wsched']]} "
      f"ssched {[e for e, _ in P3['prm_ssched']]} conf {list(CONF)}")
