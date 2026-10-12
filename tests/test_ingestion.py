from finpilot.services import repository as repo
from finpilot.services.ingestion import import_rows, parse_csv, parse_money


def test_parse_money_variants():
    assert parse_money("1,20,000.50") == 120000.5
    assert parse_money("(450.00)") == -450.0
    assert parse_money("2,000 DR") == -2000.0
    assert parse_money("₹ 999 CR") == 999.0
    assert parse_money("") is None


def test_hdfc_style_debit_credit_columns_with_preamble():
    csv_text = """HDFC BANK Statement
Account No: XXXX1234

Date,Narration,Chq./Ref.No.,Value Dt,Withdrawal Amt.,Deposit Amt.,Closing Balance
01/10/26,NEFT CR-HDFC0000123-ACME TECH SALARY,000,01/10/26,,"1,15,000.00","1,50,000.00"
03/10/26,UPI-SWIGGY-swiggy@ybl-YESB0000001-412345678901-UPI,000,03/10/26,540.00,,"1,49,460.00"
13/10/26,POS 1234XXXXXX5678 TATA STARBUCKS,000,13/10/26,320.00,,"1,49,140.00"
"""
    res = parse_csv(csv_text)
    assert not res.errors, res.errors
    assert [r.amount for r in res.rows] == [115000.0, -540.0, -320.0]
    assert res.rows[2].date.day == 13 and res.rows[2].date.month == 10  # dd/mm detected from "13"
    assert res.rows[-1].balance == 149140.0


def test_us_style_with_type_column():
    csv_text = "Posting Date,Description,Amount,Type\n10/13/2026,COFFEE SHOP,4.50,DEBIT\n10/14/2026,PAYROLL,2500,CREDIT\n"
    res = parse_csv(csv_text)
    assert [r.amount for r in res.rows] == [-4.5, 2500.0]
    assert res.rows[0].date.month == 10 and res.rows[0].date.day == 13


def test_reimport_is_idempotent_and_sets_balance(db):
    csv_text = (
        "date,description,amount,balance\n2026-10-01,SALARY ACME,100000,150000\n"
        "2026-10-02,SWIGGY,-500,149500\n2026-10-02,SWIGGY,-500,149000\n"
    )
    user = repo.get_or_create_user(db)
    parsed = parse_csv(csv_text)
    b1 = import_rows(db, user, parsed.rows, source="csv")
    b2 = import_rows(db, user, parse_csv(csv_text).rows, source="csv")
    assert (b1.rows_imported, b2.rows_imported, b2.rows_duplicate) == (
        3,
        0,
        3,
    )  # identical rows kept once each
    bal, as_of = repo.current_balance(db, user)
    assert bal == 149000.0 and as_of.isoformat() == "2026-10-02"


def test_rejects_unusable_file():
    res = parse_csv("foo,bar\n1,2\n")
    assert res.errors and not res.rows
