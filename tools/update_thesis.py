"""Inserta los Capítulos 3 y 4 en el documento de tesis.

Uso: python tools/update_thesis.py <tesis.docx> [--out salida.docx]
- Crea una copia de respaldo junto al original (<nombre>.backup-<fecha>.docx).
- Reemplaza el contenido entre los títulos «Desarrollo del Trabajo» y «CAPITULO 4»,
  y entre «Pruebas de Concepto» y «CAPITULO 5».
- Agrega la figura cobertura-riesgo y marca el documento para actualizar los índices
  al abrirlo en Word.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import chapter3  # noqa: E402
import chapter4  # noqa: E402
import docx_writer as W  # noqa: E402


def _txt(p: str) -> str:
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))


def _paras(xml: str):
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"<w:p[ >].*?</w:p>|<w:tbl>.*?</w:tbl>", xml, re.S)]


def _find(paras, pred, start=0):
    for i in range(start, len(paras)):
        if pred(paras[i][2]):
            return i
    raise ValueError("párrafo no encontrado")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    src = args.docx
    out = args.out or src
    if out == src:
        backup = src.with_name(f"{src.stem}.backup-{dt.datetime.now():%Y%m%d-%H%M}{src.suffix}")
        shutil.copy2(src, backup)
        print("Respaldo:", backup)

    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)
    doc = (tmp / "word/document.xml").read_text(encoding="utf-8")

    # Figuras: registrar imágenes y relaciones
    rels_p = tmp / "word/_rels/document.xml.rels"
    ct_p = tmp / "[Content_Types].xml"
    ct = ct_p.read_text(encoding="utf-8")
    if 'Extension="png"' not in ct:
        ct_p.write_text(ct.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>'), encoding="utf-8")

    def add_image(fig: Path, rid: str, pid: int, name: str):
        if not fig.exists():
            return None
        rels = rels_p.read_text(encoding="utf-8")
        target = f"media/vgate_{fig.name}"
        if rid not in rels:
            rels = rels.replace(
                "</Relationships>",
                f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/></Relationships>',
            )
            rels_p.write_text(rels, encoding="utf-8")
        shutil.copy2(fig, tmp / "word" / target)
        w, h = png_size(fig)
        cx = 5_900_000
        return (rid, cx, int(cx * h / w), pid, name)

    ro = __import__("json").loads((ROOT / "results/retrieval_only.json").read_text())
    chapter4.retrieval_figure(ROOT, ro)
    img_retr = add_image(ROOT / "results/retrieval_diagnostics.png", "rIdVgateFig1", 9101, "Figura diagnostico recuperacion")
    img_risk = None
    if (ROOT / "results/evaluation.json").exists():
        img_risk = add_image(ROOT / "results/risk_coverage.png", "rIdVgateFig2", 9102, "Figura cobertura-riesgo")

    ch3 = W.render(chapter3.blocks(ROOT))
    ch4a, ch4b = chapter4.blocks(ROOT, img_risk, img_retr)
    ch4a, ch4b = W.render(ch4a), W.render(ch4b)

    paras = _paras(doc)
    i31 = _find(paras, lambda p: "Heading3" in p and "Desarrollo del Trabajo" in _txt(p))
    i4 = _find(paras, lambda p: "Heading1" in p and _txt(p).strip().startswith("CAPITULO 4"), i31)
    i41 = _find(paras, lambda p: "Heading3" in p and "Pruebas de Concepto" in _txt(p), i4)
    i42 = _find(paras, lambda p: "Heading3" in p and "Análisis de Resultados" in _txt(p), i41)
    i5 = _find(paras, lambda p: "Heading1" in p and _txt(p).strip().startswith("CAPITULO 5"), i42)

    # Reemplazos de atrás hacia adelante para no invalidar posiciones.
    edits = [
        (paras[i42][1], paras[i5][0], ch4b),   # después de 4.2 hasta Capítulo 5
        (paras[i41][1], paras[i42][0], ch4a),  # después de 4.1 hasta 4.2
        (paras[i31][1], paras[i4][0], ch3),    # después de 3.1 hasta Capítulo 4
    ]
    for s, e, new in edits:
        doc = doc[:s] + new + doc[e:]
    (tmp / "word/document.xml").write_text(doc, encoding="utf-8")

    # Pedir a Word que actualice índices/campos al abrir
    st_p = tmp / "word/settings.xml"
    st = st_p.read_text(encoding="utf-8")
    if "w:updateFields" not in st:
        # El esquema exige que updateFields preceda a hdrShapeDefaults/footnotePr/endnotePr/compat.
        m = re.search(r"<w:(hdrShapeDefaults|footnotePr|endnotePr|compat)[ >/]", st)
        st = st[: m.start()] + '<w:updateFields w:val="true"/>' + st[m.start():]
        st_p.write_text(st, encoding="utf-8")

    tmp_out = out.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tmp.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp).as_posix())
    tmp_out.replace(out)
    print("Documento actualizado:", out)


if __name__ == "__main__":
    main()
