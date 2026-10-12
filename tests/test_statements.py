"""Statement import across formats and bank layouts (CSV, legacy XLS, HTML-as-XLS, XLSX, PDF)."""

from __future__ import annotations

import pytest

from finpilot.services.statements import StatementError, parse_statement
from finpilot.services.statements.columns import classify
from finpilot.services.statements.values import DateParser, parse_money
from tests import statement_fixtures as fx


def _check(content: bytes, filename: str, exp: dict, **kw):
    res = parse_statement(content, filename, use_jev=False, **kw)
    got = [(r.date.isoformat(), r.amount) for r in res.rows]
    assert got == exp["rows"], (res.mapping, res.errors, res.warnings)
    return res


@pytest.mark.parametrize(
    "builder",
    [fx.icici_csv, fx.sbi_xls, fx.hdfc_html_xls, fx.kotak_xlsx, fx.headerless_csv, fx.crdr_suffix_csv],
    ids=["icici-csv", "sbi-xls", "hdfc-html-xls", "kotak-xlsx", "headerless-csv", "crdr-suffix-txt"],
)
def test_bank_layouts_parse_exactly(builder):
    content, name, exp = builder()
    res = _check(content, name, exp)
    assert not res.errors
    closing = [r.balance for r in res.rows if r.balance is not None]
    assert exp["closing"] in closing


def test_icici_details_preserved():
    content, name, exp = fx.icici_csv()
    res = _check(content, name, exp)
    assert res.format == "csv"
    assert res.mapping["debit"] == "Withdrawal Amount(INR)" and res.mapping["date"] == "Transaction Date"
    # the remark wrapped mid-token onto a second physical row is glued back together
    assert res.rows[3].description == fx.LEDGER[3][1]
    assert res.checks["balance_reconciled_pct"] == 100.0
    assert res.summary()["date_range"] == ["2026-07-13", "2026-09-29"]


def test_pdf_ruled_table_across_pages_with_opening_balance():
    content, name, exp = fx.axis_pdf()
    res = _check(content, name, exp)
    assert res.format == "pdf" and res.opening_balance == fx.OPENING
    assert res.checks["balance_reconciled_pct"] == 100.0  # opening balance lets the first row reconcile too


def test_pdf_text_layout_infers_signs_from_balance():
    content, name, exp = fx.text_pdf()
    res = parse_statement(content, name, use_jev=False)
    got = [(r.date.isoformat(), r.amount) for r in res.rows]
    # newest first; every sign but the oldest row's comes from the running balance
    assert got[:-1] == exp["rows"][:-1]
    assert any("running balance" in w for w in res.warnings)


def test_password_protected_pdf():
    content, name, exp = fx.axis_pdf(password="01011990")
    with pytest.raises(StatementError) as e:
        parse_statement(content, name, use_jev=False)
    assert e.value.code == "password_required"
    with pytest.raises(StatementError) as e:
        parse_statement(content, name, password="wrong", use_jev=False)
    assert e.value.code == "password_incorrect"
    _check(content, name, exp, password="01011990")


def test_unreadable_files_explain_themselves():
    with pytest.raises(StatementError) as e:
        parse_statement(fx.image_only_pdf(), "scan.pdf", use_jev=False)
    assert e.value.code == "scanned_pdf"
    with pytest.raises(StatementError) as e:
        parse_statement(b"not really a pdf", "statement.pdf", use_jev=False)
    assert e.value.code == "corrupt"
    with pytest.raises(StatementError):
        parse_statement(b"   ", "empty.csv", use_jev=False)
    res = parse_statement(b"foo,bar\n1,2\n", "x.csv", use_jev=False)
    assert not res.rows and res.errors


def test_values_and_dates():
    assert parse_money("1,23,456.78") == 123456.78
    assert parse_money("1,234.00 Dr") == -1234.0 and parse_money("99 CR") == 99.0
    assert parse_money("(450.00)") == -450.0 and parse_money("250.00-") == -250.0
    assert parse_money("−75") == -75.0 and parse_money("₹ 1,000") == 1000.0
    assert parse_money("-") is None and parse_money("abc") is None
    p = DateParser(["13,07,2026", "01,08,2026"])
    assert p.parse("01,08,2026")[0].isoformat() == "2026-08-01"  # ICICI's comma dates, day first
    p = DateParser(["05/07/2026", "06/07/2026"])
    assert p.parse("05/07/2026")[0].month == 7  # ambiguous → day-first (India)
    p = DateParser(["07/13/2026", "07/14/2026"])
    assert p.parse("07/13/2026")[0].day == 13  # only month-first parses → US
    d, t = DateParser(["13 Jul 2026 21:45"]).parse("13 Jul 2026 21:45")
    assert d.isoformat() == "2026-07-13" and t.hour == 21


@pytest.mark.parametrize(
    ("header", "role"),
    [
        ("Withdrawal Amount(INR)", "debit"),
        ("Deposit Amt.", "credit"),
        ("Dr/Cr Amount", "amount"),
        ("Debit/Credit", "type"),
        ("Txn Posted On", "date"),
        ("Value Dt", "value_date"),
        ("Narrative", "description"),
        ("Closing Balance", "balance"),
        ("Chq./Ref.No.", "reference"),
        ("S No.", "serial"),
    ],
)
def test_header_vocabulary(header, role):
    assert classify(header) == role


def test_upload_endpoint_handles_excel_and_pdf_password(seeded):
    from fastapi.testclient import TestClient

    from finpilot.main import create_app

    with TestClient(create_app()) as c:
        content, name, exp = fx.kotak_xlsx()
        r = c.post("/api/transactions/upload", files={"file": (name, content)}, params={"run": "false"})
        assert r.status_code == 200, r.text
        batch = r.json()["batch"]
        assert batch["imported"] == len(exp["rows"]) and batch["format"] == "xlsx"
        assert batch["summary"]["checks"]["balance_reconciled_pct"] == 100.0
        pdf, pname, _ = fx.axis_pdf(password="secret")
        r = c.post("/api/transactions/upload", files={"file": (pname, pdf)}, params={"run": "false"})
        assert r.status_code == 422 and r.json()["detail"]["code"] == "password_required"
        r = c.post(
            "/api/transactions/upload",
            files={"file": (pname, pdf)},
            data={"password": "secret"},
            params={"run": "false"},
        )
        assert r.status_code == 200
        b = r.json()["batch"]
        assert b["format"] == "pdf" and b["imported"] + b["duplicates"] == len(exp["rows"])
        # the identical PDF again adds nothing (row fingerprints)
        r = c.post(
            "/api/transactions/upload",
            files={"file": (pname, pdf)},
            data={"password": "secret"},
            params={"run": "false"},
        )
        assert r.json()["batch"]["imported"] == 0 and r.json()["batch"]["duplicates"] == len(exp["rows"])
        r = c.post("/api/transactions/upload", files={"file": ("scan.pdf", fx.image_only_pdf())})
        assert r.status_code == 422 and r.json()["detail"]["code"] == "scanned_pdf"
