"""Utilidades para insertar contenido en el .docx de la tesis respetando sus estilos.

Trabaja directamente sobre word/document.xml (no reformatea el resto del documento).
Bloques soportados:
  ("h3", texto)            subtítulo numerado (Heading3, lista numId 9 nivel 1)
  ("h4", texto)            subtítulo de cuarto nivel (párrafo en negrita)
  ("p", texto)             párrafo; **negrita** y *cursiva* en línea
  ("bullet", texto)        viñeta (ListBullet)
  ("caption", texto)       título de tabla/figura (centrado, cursiva)
  ("table", filas)         tabla TableGrid; la primera fila es encabezado
  ("image", (rid, cx, cy)) figura en línea (EMU)
  ("code", texto)          línea monoespaciada
"""
from __future__ import annotations

import re
from xml.sax.saxutils import escape

RPR_BODY = '<w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:lang w:eastAsia="es-ES" w:bidi="es-ES"/></w:rPr>'
H3_PPR = (
    '<w:pPr><w:pStyle w:val="Heading3"/><w:numPr><w:ilvl w:val="1"/><w:numId w:val="9"/></w:numPr>'
    '<w:rPr><w:rFonts w:ascii="Aptos" w:eastAsia="Times New Roman" w:hAnsi="Aptos" w:cs="Times New Roman"/>'
    '<w:color w:val="auto"/><w:kern w:val="0"/><w:lang w:eastAsia="es-ES" w:bidi="es-ES"/></w:rPr></w:pPr>'
)
H3_RPR = (
    '<w:rPr><w:rFonts w:ascii="Aptos" w:eastAsia="Times New Roman" w:hAnsi="Aptos" w:cs="Times New Roman"/>'
    '<w:color w:val="auto"/><w:kern w:val="0"/><w:lang w:eastAsia="es-ES" w:bidi="es-ES"/></w:rPr>'
)


def _runs(text: str, base_rpr_inner: str = "") -> str:
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        extra = ""
        if part.startswith("**") and part.endswith("**"):
            part, extra = part[2:-2], "<w:b/><w:bCs/>"
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            part, extra = part[1:-1], "<w:i/><w:iCs/>"
        out.append(
            f'<w:r><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/>{extra}{base_rpr_inner}'
            f'<w:lang w:eastAsia="es-ES" w:bidi="es-ES"/></w:rPr><w:t xml:space="preserve">{escape(part)}</w:t></w:r>'
        )
    return "".join(out)


def p(text: str) -> str:
    return f'<w:p><w:pPr><w:jc w:val="both"/>{RPR_BODY}</w:pPr>{_runs(text)}</w:p>'


def h3(text: str, bookmark: tuple[int, str] | None = None) -> str:
    bm_s = bm_e = ""
    if bookmark:
        bm_s = f'<w:bookmarkStart w:id="{bookmark[0]}" w:name="{bookmark[1]}"/>'
        bm_e = f'<w:bookmarkEnd w:id="{bookmark[0]}"/>'
    return f'<w:p>{H3_PPR}{bm_s}<w:r>{H3_RPR}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>{bm_e}</w:p>'


def h4(text: str) -> str:
    return (
        f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="80"/>{RPR_BODY}</w:pPr>'
        f'{_runs(text, "<w:b/><w:bCs/>")}</w:p>'
    )


def bullet(text: str) -> str:
    return (
        '<w:p><w:pPr><w:pStyle w:val="ListBullet"/><w:jc w:val="both"/>'
        '<w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/></w:rPr></w:pPr>'
        f"{_runs(text)}</w:p>"
    )


_CAPTION_RPR = '<w:i/><w:iCs/><w:sz w:val="20"/><w:szCs w:val="20"/>'


def caption(text: str) -> str:
    return (
        '<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/></w:pPr>'
        f"{_runs(text, _CAPTION_RPR)}</w:p>"
    )


def code(text: str) -> str:
    return (
        '<w:p><w:pPr><w:spacing w:after="0"/><w:ind w:left="567"/></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:sz w:val="18"/></w:rPr>'
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    )


def table(rows: list[list[str]], widths: list[int] | None = None, total: int = 9000) -> str:
    ncol = len(rows[0])
    widths = widths or [total // ncol] * ncol
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    trs = []
    for i, row in enumerate(rows):
        tcs = []
        for j, cell in enumerate(row):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>' if i == 0 else ""
            bold = "<w:b/><w:bCs/>" if i == 0 else ""
            tcs.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{shade}</w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>'
                f'<w:r><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/>{bold}<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
                f'<w:t xml:space="preserve">{escape(str(cell))}</w:t></w:r></w:p></w:tc>'
            )
        hdr = "<w:trPr><w:tblHeader/></w:trPr>" if i == 0 else ""
        trs.append(f"<w:tr>{hdr}{''.join(tcs)}</w:tr>")
    return (
        f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="{sum(widths)}" w:type="dxa"/>'
        f'<w:jc w:val="center"/><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
        f'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(trs)}</w:tbl>'
        + '<w:p><w:pPr><w:spacing w:after="0"/></w:pPr></w:p>'
    )


def image(rid: str, cx: int, cy: int, pid: int, name: str) -> str:
    return (
        '<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{pid}" name="{name}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{pid}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        "</wp:inline></w:drawing></w:r></w:p>"
    )


def render(blocks: list[tuple]) -> str:
    out = []
    for kind, val in blocks:
        out.append(
            {
                "h3": h3,
                "h4": h4,
                "p": p,
                "bullet": bullet,
                "caption": caption,
                "code": code,
            }[kind](val)
            if kind not in ("table", "image")
            else (table(*val) if kind == "table" else image(*val))
        )
    return "".join(out)
