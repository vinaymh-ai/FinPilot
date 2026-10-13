"""Synthetic bank statements in the layouts real Indian banks export (all names/accounts fictional).

Each builder returns (bytes, filename, expected) where `expected` lists (iso date, signed amount)
in file order plus the closing balance, so tests can assert exact parsing.
"""

from __future__ import annotations

import datetime as dt
import io

# A small ledger used by every fixture: (date, description, signed amount); balances are derived.
LEDGER = [
    (dt.date(2026, 7, 13), "MMT/IMPS/619415828955/Transfer to Fam/RAMESH K/State Bank of I", 1.00),
    (
        dt.date(2026, 7, 14),
        "UPI/Suresh Kum/9000000001@ibl/Payment fr/State Bank/711927613493/AXL1bf0644c938",
        -30.00,
    ),
    (dt.date(2026, 8, 7), "NEFT-HDFCH01180747885-ACME CONSULTING PVT LTD-0001NEFT ACME6197", 94571.00),
    (
        dt.date(2026, 8, 7),
        "UPI/CRED Club/cred.club@axis/payment on/AXIS BANK/658528409634/ACDe319cadca",
        -20960.92,
    ),
    (dt.date(2026, 8, 10), "BIL/Personal Loan XX85585 EMI Demo", -38449.00),
    (
        dt.date(2026, 8, 10),
        "UPI/ICCL Mutua/mfautopay.elem/Indian Cle/HDFC BANK/103830699305/HDF58A6",
        -2000.00,
    ),
    (
        dt.date(2026, 8, 11),
        "UPI/Indian Rai/bdpg2.ir@sbi/Payment fr/State Bank/466838525397/AXLb747f5b",
        -522.70,
    ),
    (dt.date(2026, 9, 29), "035001000000:Int.Pd:30-06-2026 to 29-09-2026", 46.00),
]
OPENING = 5148.49


def balances(opening: float = OPENING) -> list[float]:
    out, bal = [], opening
    for _, _, amt in LEDGER:
        bal = round(bal + amt, 2)
        out.append(bal)
    return out


def expected(order: str = "asc") -> dict:
    rows = [(d.isoformat(), a) for d, _, a in LEDGER]
    return {"rows": rows if order == "asc" else rows[::-1], "closing": balances()[-1]}


def icici_csv() -> tuple[bytes, str, dict]:
    """ICICI 'Detailed statement' CSV: leading blank column, preamble, quoted dd,mm,yyyy dates,
    a remark wrapped onto a continuation row, and a legends footer."""
    lines = [
        ",,,,,,,,,",
        ",DETAILED STATEMENT,,,,,,,,",
        ",Account Number,,000000000001 ( INR )  - DEMO USER,,,,,,",
        ',Transaction Date from,,"13,07,2026",to,"13,10,2026",,,,',
        ",Transactions List - DEMO USER - 000000000001,,,,,,,,",
        ",S No.,Value Date,Transaction Date,Cheque Number,Transaction Remarks,Withdrawal Amount(INR),"
        "Deposit Amount(INR),Balance(INR),",
    ]
    for i, ((d, desc, amt), bal) in enumerate(zip(LEDGER, balances(), strict=True), start=1):
        ds = f'"{d:%d,%m,%Y}"'
        wd, dep = (f"{-amt:.2f}", "0.00") if amt < 0 else ("0.00", f"{amt:.2f}")
        if i == 4:  # remark wrapped onto the next physical row, mid-token
            lines.append(f",{i},{ds},{ds},,{desc[:27]},{wd},{dep},{bal:.2f},")
            lines.append(f",,,,,{desc[27:]},,,,")
        else:
            lines.append(f",{i},{ds},{ds},,{desc},{wd},{dep},{bal:.2f},")
    lines += [
        ",Legends Used in Account Statement,,,,,,,,",
        ",1. INFT - Internal Fund Transfer (Within ICICI Bank),,,,,,,,",
        ",26. MMT - Mobile Money Transfer (Insta FT - IMPS),,,,,,,,",
    ]
    return ("\n".join(lines) + "\n").encode(), "OpTransactionHistory.csv", expected()


def sbi_xls() -> tuple[bytes, str, dict]:
    """SBI-style legacy .xls: preamble rows, text dates '7 Aug 2026', Debit/Credit/Balance columns."""
    import xlwt

    wb = xlwt.Workbook()
    ws = wb.add_sheet("Statement")
    ws.write(0, 0, "Account Name : DEMO USER")
    ws.write(1, 0, "Account Number : 00000000001")
    ws.write(2, 0, "Balance as on 13 Jul 2026 : 5,148.49")
    header = ["Txn Date", "Value Date", "Description", "Ref No./Cheque No.", "Debit", "Credit", "Balance"]
    for j, h in enumerate(header):
        ws.write(4, j, h)
    for i, ((d, desc, amt), bal) in enumerate(zip(LEDGER, balances(), strict=True), start=5):
        ws.write(i, 0, f"{d.day} {d:%b %Y}")
        ws.write(i, 1, f"{d.day} {d:%b %Y}")
        ws.write(i, 2, desc)
        ws.write(i, 3, "")
        ws.write(i, 4, f"{-amt:,.2f}" if amt < 0 else "")
        ws.write(i, 5, f"{amt:,.2f}" if amt > 0 else "")
        ws.write(i, 6, f"{bal:,.2f}")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue(), "sbi_statement.xls", expected()


def hdfc_html_xls() -> tuple[bytes, str, dict]:
    """HDFC-style 'XLS' that is really an HTML table, newest first, dd/mm/yy dates, Cr/Dr balances."""
    rows = ""
    for (d, desc, amt), bal in reversed(list(zip(LEDGER, balances(), strict=True))):
        wd = f"{-amt:,.2f}" if amt < 0 else ""
        dep = f"{amt:,.2f}" if amt > 0 else ""
        rows += f"<tr><td>{d:%d/%m/%y}</td><td>{desc}</td><td>0000</td><td>{d:%d/%m/%y}</td><td>{wd}</td><td>{dep}</td><td>{bal:,.2f} Cr</td></tr>\n"
    html = f"""<html><body><table><tr><td>HDFC BANK Ltd. Statement of accounts</td></tr></table>
<table border=1><tr><th>Date</th><th>Narration</th><th>Chq./Ref.No.</th><th>Value Dt</th>
<th>Withdrawal Amt.</th><th>Deposit Amt.</th><th>Closing Balance</th></tr>
{rows}</table></body></html>"""
    return html.encode(), "Acct_Statement_XX1234.xls", expected("desc")


def kotak_xlsx() -> tuple[bytes, str, dict]:
    """Kotak-style .xlsx: real datetime cells, one Amount column plus a Dr/Cr column."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Transactions"
    ws.append(["Kotak Mahindra Bank — Account Statement"])
    ws.append([])
    ws.append(["Sl. No.", "Transaction Date", "Description", "Chq / Ref No.", "Amount", "Dr / Cr", "Balance"])
    for i, ((d, desc, amt), bal) in enumerate(zip(LEDGER, balances(), strict=True), start=1):
        ws.append(
            [
                i,
                dt.datetime(d.year, d.month, d.day, 10, 30),
                desc,
                "",
                abs(amt),
                "CR" if amt > 0 else "DR",
                bal,
            ]
        )
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue(), "kotak.xlsx", expected()


def axis_pdf(password: str | None = None) -> tuple[bytes, str, dict]:
    """Axis-style PDF with a ruled table split across two pages (header repeated on each page)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

    buf = io.BytesIO()
    kwargs = {"encrypt": password} if password else {}
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), **kwargs)
    header = ["Tran Date", "Chq No", "Particulars", "Debit", "Credit", "Balance"]
    data = [
        [
            f"{d:%d-%m-%Y}",
            "",
            desc[:60],
            f"{-amt:.2f}" if amt < 0 else "",
            f"{amt:.2f}" if amt > 0 else "",
            f"{bal:.2f}",
        ]
        for (d, desc, amt), bal in zip(LEDGER, balances(), strict=True)
    ]
    style = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("FONTSIZE", (0, 0), (-1, -1), 7)])
    story = [Paragraph("AXIS BANK — Statement of Account No: 000000000001", getSampleStyleSheet()["Normal"])]
    story.append(
        Table([header] + [["OPENING BALANCE", "", "", "", "", f"{OPENING:.2f}"]] + data[:4], style=style)
    )
    story.append(PageBreak())
    story.append(Table([header] + data[4:], style=style))
    doc.build(story)
    return buf.getvalue(), "axis_statement.pdf", expected()


def text_pdf() -> tuple[bytes, str, dict]:
    """A PDF with no table grid: transaction lines drawn as text, newest first, one amount column."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    c.setFont("Helvetica", 9)
    c.drawString(40, y, "Statement of Account  —  DEMO USER")
    y -= 20
    c.drawString(40, y, "Date        Description                                   Amount        Balance")
    y -= 14
    for (d, desc, amt), bal in reversed(list(zip(LEDGER, balances(), strict=True))):
        c.drawString(40, y, f"{d:%d %b %Y}   {desc[:48]:<48}   {abs(amt):>12,.2f}   {bal:>12,.2f}")
        y -= 14
    c.save()
    return buf.getvalue(), "statement_text.pdf", expected("desc")


def image_only_pdf() -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.rect(50, 50, 200, 200, fill=1)  # no text at all — like a scan
    c.save()
    return buf.getvalue()


def headerless_csv() -> tuple[bytes, str, dict]:
    lines = [
        f"{d:%d/%m/%Y},{desc},{amt:.2f},{bal:.2f}"
        for (d, desc, amt), bal in zip(LEDGER, balances(), strict=True)
    ]
    return ("\n".join(lines) + "\n").encode(), "export.csv", expected()


def crdr_suffix_csv() -> tuple[bytes, str, dict]:
    lines = ["Txn Date;Remarks;Amount;Balance"]
    for (d, desc, amt), bal in zip(LEDGER, balances(), strict=True):
        lines.append(f"{d:%d.%m.%Y};{desc};{abs(amt):,.2f} {'Cr' if amt > 0 else 'Dr'};{bal:,.2f} Cr")
    return ("\n".join(lines) + "\n").encode("cp1252"), "statement.txt", expected()
