from __future__ import annotations

import math
import re
import zipfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape, quoteattr


_INVALID_XML_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_INVALID_SHEET_NAME_CHARACTERS = re.compile(r"[\\/:?*\[\]]")


@dataclass(frozen=True)
class TableSheet:
    name: str
    fieldnames: Sequence[str]
    group_key_columns: Sequence[str] = ()
    merge_columns: Sequence[str] = ()
    rows: Sequence[Mapping[str, Any]] | None = None
    wrap_columns: Sequence[str] = ()
    freeze_columns: int = 0


def _column_name(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_text(value: Any) -> str:
    return escape(_INVALID_XML_CHARACTERS.sub("", str(value)))


def _cell_xml(reference: str, value: Any, style: int) -> str:
    style_attribute = f' s="{style}"' if style else ""
    if isinstance(value, bool):
        return f'<c r="{reference}"{style_attribute} t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            value = ""
        else:
            return f'<c r="{reference}"{style_attribute}><v>{value}</v></c>'
    text = _xml_text(value)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return (
        f'<c r="{reference}"{style_attribute} t="inlineStr">'
        f'<is><t{preserve}>{text}</t></is></c>'
    )


def _column_width(fieldname: str, rows: Sequence[Mapping[str, Any]]) -> float:
    longest = len(fieldname)
    for row in rows:
        value = str(row.get(fieldname, ""))
        for part in value.splitlines() or [""]:
            longest = max(longest, len(part))
    if fieldname in {"FUSED_OP_NPU_TYPE", "FUSED_OP_GPU_TYPE"}:
        return min(max(longest + 2, 20), 42)
    if fieldname == "Mapping Mode Counts":
        return min(max(longest + 2, 24), 60)
    if fieldname == "排查结论":
        return 60
    if fieldname == "排查建议":
        return 76
    if fieldname == "OP Type":
        return min(max(longest + 2, 16), 36)
    if fieldname == "NPU Types":
        return min(max(longest + 2, 24), 48)
    if fieldname == "Scope":
        return min(max(longest * 2 + 2, 24), 50)
    if fieldname == "NPU Path":
        return 64
    if fieldname in {"Candidate Cause", "Comparison Status", "GPU Cast Evidence", "Pattern Issues"}:
        return 34
    if fieldname in {"NPU Input Shapes", "NPU Input DTypes", "NPU Kernel Indices", "GPU Kernel Indices"}:
        return 28
    return min(max(longest + 2, 10), 24)


def _worksheet_xml(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    group_key_columns: Sequence[str],
    merge_columns: Sequence[str],
    wrap_columns: Sequence[str] = (),
    freeze_columns: int = 0,
) -> str:
    merge_column_indices = {
        fieldnames.index(fieldname)
        for fieldname in merge_columns
        if fieldname in fieldnames
    }

    merged_ranges: list[str] = []
    merged_body_cells: set[tuple[int, int]] = set()
    start = 0
    while start < len(rows):
        key = tuple(str(rows[start].get(column, "")) for column in group_key_columns)
        end = start + 1
        while end < len(rows) and tuple(
            str(rows[end].get(column, "")) for column in group_key_columns
        ) == key:
            end += 1
        if end - start > 1:
            first_excel_row = start + 2
            last_excel_row = end + 1
            for column_index in merge_column_indices:
                column = _column_name(column_index)
                merged_ranges.append(
                    f"{column}{first_excel_row}:{column}{last_excel_row}"
                )
                for body_row in range(start + 1, end):
                    merged_body_cells.add((body_row, column_index))
        start = end

    last_column = _column_name(max(len(fieldnames) - 1, 0))
    last_row = max(len(rows) + 1, 1)
    column_xml = "".join(
        f'<col min="{index + 1}" max="{index + 1}" '
        f'width="{_column_width(fieldname, rows):.2f}" customWidth="1"/>'
        for index, fieldname in enumerate(fieldnames)
    )
    header_cells = "".join(
        _cell_xml(f"{_column_name(index)}1", fieldname, 1)
        for index, fieldname in enumerate(fieldnames)
    )
    row_xml = [f'<row r="1" ht="30" customHeight="1">{header_cells}</row>']
    for body_index, row in enumerate(rows):
        excel_row = body_index + 2
        cells = []
        for column_index, fieldname in enumerate(fieldnames):
            if (body_index, column_index) in merged_body_cells:
                continue
            value = row.get(fieldname, "")
            if fieldname == "Mapping Mode Counts" or fieldname in wrap_columns:
                style = 3
            elif column_index in merge_column_indices:
                style = 2
            elif fieldname.endswith("(%)"):
                style = 5
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value /= 100
            elif fieldname.endswith("(us)"):
                style = 6
            elif fieldname.endswith("Ratio(GPU/NPU)"):
                style = 7
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if value >= 2.0:
                        style = 8
                    elif value >= 1.0:
                        style = 9
                    elif value < 1.0:
                        style = 10
            elif fieldname.endswith("Count") or fieldname == "Mapped CPU OP Calls":
                style = 4
            else:
                style = 0
            if row.get("Row Kind") == "total":
                style = {0: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17}.get(style, style)
            cells.append(
                _cell_xml(
                    f"{_column_name(column_index)}{excel_row}",
                    value,
                    style,
                )
            )
        mode_counts = str(row.get("Mapping Mode Counts", "")) if "Mapping Mode Counts" in fieldnames else ""
        row_height = min(90, 15 * max(1, math.ceil(len(mode_counts) / 58)))
        if wrap_columns:
            row_height = max(row_height, 15 * max(
                1, *(sum(max(1, math.ceil(sum(2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1 for c in line)
                                          / max(1, _column_width(field, rows) - 4)))
                         for line in str(row.get(field, "")).splitlines() or [""])
                     for field in wrap_columns),
            ))
        height_attribute = f' ht="{row_height}" customHeight="1"' if row_height > 15 else ""
        row_xml.append(f'<row r="{excel_row}"{height_attribute}>{"".join(cells)}</row>')
    merge_xml = (
        f'<mergeCells count="{len(merged_ranges)}">'
        + "".join(f'<mergeCell ref="{value}"/>' for value in merged_ranges)
        + "</mergeCells>"
        if merged_ranges
        else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{last_column}{last_row}"/>'
        '<sheetViews><sheetView workbookViewId="0">'
        + (f'<pane xSplit="{freeze_columns}" ySplit="1" '
           f'topLeftCell="{_column_name(freeze_columns)}2" activePane="bottomRight" state="frozen"/>'
           if freeze_columns else
           '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>') +
        '</sheetView></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<cols>{column_xml}</cols>'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        f'<autoFilter ref="A1:{last_column}{last_row}"/>'
        # OOXML requires autoFilter before mergeCells.  Browsers are often
        # tolerant of the reverse order, but desktop Excel rejects or repairs it.
        f'{merge_xml}'
        '</worksheet>'
    )


def write_grouped_workbook_xlsx(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    sheets: Sequence[TableSheet],
) -> None:
    """Write a dependency-free multi-sheet XLSX with vertical group merges."""
    if not sheets:
        raise ValueError("at least one worksheet is required")
    names = [sheet.name for sheet in sheets]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("worksheet names must be unique")
    for sheet in sheets:
        if not sheet.name or len(sheet.name) > 31 or _INVALID_SHEET_NAME_CHARACTERS.search(sheet.name):
            raise ValueError(f"invalid worksheet name: {sheet.name!r}")
        if not sheet.fieldnames:
            raise ValueError(f"worksheet {sheet.name!r} needs at least one column")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<numFmts count="3">'
        '<numFmt numFmtId="164" formatCode="0.00%"/>'
        '<numFmt numFmtId="165" formatCode="#,##0.000"/>'
        '<numFmt numFmtId="166" formatCode="0.000"/>'
        '</numFmts>'
        '<fonts count="3">'
        '<font><sz val="10"/><name val="Arial"/></font>'
        '<font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Arial"/></font>'
        '<font><b/><sz val="10"/><name val="Arial"/></font>'
        '</fonts>'
        '<fills count="7">'
        '<fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/>'
        '<bgColor indexed="64"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FF1B5E20"/>'
        '<bgColor indexed="64"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFC6E0B4"/>'
        '<bgColor indexed="64"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/>'
        '<bgColor indexed="64"/></patternFill></fill>'
        '<fill><patternFill patternType="solid"><fgColor rgb="FFDCE6F1"/>'
        '<bgColor indexed="64"/></patternFill></fill>'
        '</fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/>'
        '</border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" '
        'borderId="0"/></cellStyleXfs>'
        '<cellXfs count="18">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" '
        'applyFont="1" applyFill="1" applyAlignment="1">'
        '<alignment horizontal="center" vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" '
        'applyAlignment="1"><alignment horizontal="center" vertical="center" '
        'wrapText="1"/></xf>'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" '
        'applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>'
        '<xf numFmtId="3" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
        '<xf numFmtId="166" fontId="1" fillId="3" borderId="0" xfId="0" '
        'applyNumberFormat="1" applyFont="1" applyFill="1"/>'
        '<xf numFmtId="166" fontId="0" fillId="4" borderId="0" xfId="0" '
        'applyNumberFormat="1" applyFill="1"/>'
        '<xf numFmtId="166" fontId="0" fillId="5" borderId="0" xfId="0" '
        'applyNumberFormat="1" applyFill="1"/>'
        + ''.join(f'<xf numFmtId="{fmt}" fontId="2" fillId="6" borderId="0" xfId="0" '
                  'applyNumberFormat="1" applyFont="1" applyFill="1" applyAlignment="1">'
                  f'<alignment vertical="center"{alignment}/></xf>'
                  for fmt, alignment in ((0, ''), (0, ' horizontal="center" wrapText="1"'),
                                         (0, ' wrapText="1"'), (3, ''), (164, ''), (165, ''), (166, '')))
        +
        '</cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/>'
        '</cellStyles>'
        '</styleSheet>'
    )
    sheet_entries = "".join(
        f'<sheet name={quoteattr(sheet.name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, sheet in enumerate(sheets, 1)
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{sheet_entries}</sheets></workbook>'
    )
    sheet_relationships = "".join(
        f'<Relationship Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{sheet_relationships}'
        f'<Relationship Id="rId{len(sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/></Relationships>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    sheet_content_types = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(sheets) + 1)
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f'{sheet_content_types}'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        for index, sheet in enumerate(sheets, 1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                _worksheet_xml(
                    sheet.rows if sheet.rows is not None else rows,
                    sheet.fieldnames,
                    sheet.group_key_columns,
                    sheet.merge_columns,
                    sheet.wrap_columns,
                    sheet.freeze_columns,
                ),
            )
        archive.writestr("xl/styles.xml", styles_xml)


def write_merged_table_xlsx(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    sheet_name: str,
    group_key_columns: Sequence[str],
    merge_columns: Sequence[str],
) -> None:
    """Backward-compatible single-sheet writer."""
    write_grouped_workbook_xlsx(
        path,
        rows,
        [TableSheet(sheet_name, fieldnames, group_key_columns, merge_columns)],
    )
