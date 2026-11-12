#!/usr/bin/env python3
"""Build Methodology_revised.docx from Methodology_revised.md, reusing the styles of the original Word file.

    python3 docs/methodology/build_docx.py <original Methodology.docx> docs/methodology/Methodology_revised.md <output.docx>

The Markdown source is the version-controlled text of the Methodology. Supported syntax, deliberately small:
'# ', '## ', '### ' headings; paragraphs; '- ' bullets; pipe tables (first row = header); a line starting
with '$$' is a display equation (written in Word's linear equation format, so Word can build it up with
Equation > Professional); **bold** and *italic* inline. Everything else in the original file (styles, fonts,
numbering, page set-up) is kept; only the document body is replaced.
"""
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def runs(text):
    """Inline **bold** and *italic* to Word runs."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        properties = ""
        if part.startswith("**"):
            part, properties = part[2:-2], "<w:rPr><w:b/></w:rPr>"
        elif part.startswith("*"):
            part, properties = part[1:-1], "<w:rPr><w:i/></w:rPr>"
        out.append(f'<w:r>{properties}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(out)


def paragraph(text, style=None, extra=""):
    properties = (f'<w:pStyle w:val="{style}"/>' if style else "") + extra
    return f"<w:p>{'<w:pPr>' + properties + '</w:pPr>' if properties else ''}{runs(text)}</w:p>"


def equation(text):
    return ('<w:p><w:pPr><w:jc w:val="center"/></w:pPr><m:oMathPara><m:oMath><m:r><m:t xml:space="preserve">'
            f"{escape(text)}</m:t></m:r></m:oMath></m:oMathPara></w:p>")


def table(rows):
    border = "".join(f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="808080"/>' for side in ("top", "left", "bottom", "right", "insideH", "insideV"))
    columns = max(len(r) for r in rows)
    width = 9000 // columns
    xml = [f'<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblBorders>{border}</w:tblBorders>'
           '<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar></w:tblPr>',
           "<w:tblGrid>" + f'<w:gridCol w:w="{width}"/>' * columns + "</w:tblGrid>"]
    for index, row in enumerate(rows):
        cells = []
        for cell in row + [""] * (columns - len(row)):
            text = f"**{cell}**" if index == 0 and cell else cell
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if index == 0 else ""
            cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>'
                         f'<w:p><w:pPr><w:spacing w:after="0"/></w:pPr>{smallRuns(text)}</w:p></w:tc>')
        xml.append("<w:tr>" + "".join(cells) + "</w:tr>")
    xml.append("</w:tbl><w:p/>")
    return "".join(xml)


def smallRuns(text):
    """Table text at 9 pt."""
    xml = runs(text)
    xml = xml.replace("<w:r><w:rPr>", '<w:r><w:rPr><w:sz w:val="18"/>')
    return xml.replace("<w:r><w:t", '<w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:t')


def body(markdown):
    out, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
        elif line.startswith("### "):
            out.append(paragraph(line[4:], "Heading3")); i += 1
        elif line.startswith("## "):
            out.append(paragraph(line[3:], "Heading2")); i += 1
        elif line.startswith("# "):
            out.append(paragraph(line[2:], "Heading2")); i += 1
        elif line.startswith("$$"):
            out.append(equation(line.strip("$ "))); i += 1
        elif line.startswith("- "):
            out.append(paragraph(line[2:], "ListParagraph", '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>')); i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(table(rows))
        else:
            out.append(paragraph(line, extra='<w:spacing w:after="140"/>')); i += 1
    return "".join(out)


def main():
    original, source, target = (Path(p) for p in sys.argv[1:4])
    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(original) as archive:
        archive.extractall(work)
    document = (work / "word" / "document.xml").read_text(encoding="utf8")
    start = document.index("<w:body>") + len("<w:body>")
    section = re.search(r"<w:sectPr[ >].*?</w:sectPr>(?=\s*</w:body>)", document, flags=re.S)
    end = section.start() if section else document.index("</w:body>")
    document = document[:start] + body(source.read_text(encoding="utf8")) + document[end:]
    (work / "word" / "document.xml").write_text(document, encoding="utf8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(work.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(work).as_posix())
    shutil.rmtree(work)
    print(target)


if __name__ == "__main__":
    main()
