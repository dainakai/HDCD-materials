"""Place computed table values in the paper's LaTeX layout."""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def metric(value, *, signed=False):
    rounded = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return format(rounded, "+.3f" if signed else ".3f")


def export_table(frame, number, output):
    template = (ROOT / f"reference/paper_tables/table{number}.tex").read_text()
    rows = frame.set_index("method")
    cells = []
    if number == 3:
        generated = template
        for label, column in [("Average Precision", "event_ap"), (r"Maximum $F_1$", "maximum_f1"),
                              (r"Precision at recall $=0.8$", "precision_at_recall_0p8"),
                              (r"Precision at recall $=0.9$", "precision_at_recall_0p9"),
                              ("Candidate-pool recall", "candidate_pool_recall")]:
            ours = metric(rows.loc["3d", column])
            values = [ours]
            cells.append(ours)
            for method in ["2d", "minip2d"]:
                value = metric(rows.loc[method, column])
                delta = metric(Decimal(value) - Decimal(ours), signed=True)
                values.append(value + " ($" + delta + "$)")
                cells.extend([value, delta])
            replacement = "    " + label + " & " + " & ".join(values) + " " + r"\\"
            generated, count = re.subn(r"^\s*" + re.escape(label) + r" &[^\n]*",
                                       lambda _: replacement, generated, flags=re.M)
            if count != 1:
                raise AssertionError(f"Table 3: missing metric row {label}")
        if generated != template:
            raise AssertionError("Table 3: computed typeset values differ from the reference table")
        (output / "table3.tex").write_text(generated)
        return {"paper_rounded_cells_match": True, "numeric_cells": len(cells), "latex": "table3.tex",
                "rounding": "three decimal places, half up",
                "parenthetical_differences": "method minus Ours, using displayed values"}
    elif number == 4:
        for name in ["trajectory", "image", "relation", "suppression"]:
            row = rows.loc[name]
            cells.extend([metric(row.event_ap), metric(row.precision_at_recall_0p8),
                          f"{int(row.false_positives_at_recall_0p8):,}",
                          metric(row.precision_at_recall_0p9),
                          f"{int(row.false_positives_at_recall_0p9):,}"])
    else:
        raise ValueError(number)
    pattern = re.compile(r"(?<=&)(\s*)(\\textbf\{)?([+-]?\d[\d,.]*)(\})?(?=\s*(?:&|\\\\))")
    matches = list(pattern.finditer(template))
    if len(matches) != len(cells):
        raise AssertionError(f"Table {number}: layout has {len(matches)} cells, expected {len(cells)}")
    actual = iter(cells)

    def replace(match):
        return match[1] + (match[2] or "") + next(actual) + (match[4] or "")

    generated = pattern.sub(replace, template)
    if generated != template:
        raise AssertionError(f"Table {number}: computed typeset values differ from the paper's table")
    (output / f"table{number}.tex").write_text(generated)
    return {"paper_rounded_cells_match": True, "numeric_cells": len(cells),
            "latex": f"table{number}.tex", "rounding": "three decimal places, half up",
            "difference_column": "difference of displayed values" if number == 3 else None}
