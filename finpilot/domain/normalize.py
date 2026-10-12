"""Bank-statement description normalisation.

Raw descriptions look like:
    "UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/Payment"
    "POS 445566XXXXXX1234 AMZN Mktp IN*2K3L19X BANGALORE"
    "NETFLIX.COM 866-579-7172 CA"
    "ACH D- BAJAJ FINANCE LTD-EMI 1234"
We reduce them to (a) a normalised upper-case text for pattern matching and (b) a short,
stable merchant key used for grouping (recurring detection, user corrections).
"""

from __future__ import annotations

import re

from finpilot.domain.lexicon import match_merchant

_CHANNEL_PREFIX = re.compile(
    r"^(?:(?:POS|ECOM|PUR|PURCHASE|DEBIT CARD|DEBIT|DC|CC|VISA|MC|NFC|ACH ?[DC]?|NACH|ECS|BIL|BILLPAY|IMPS|NEFT|RTGS|"
    r"UPI|INB|MB|IB|TPT|TO|BY|TRF|TRANSFER)[\s/:\-*]+)+"
)
_UPI_NOISE = {
    "UPI",
    "DR",
    "CR",
    "P2M",
    "P2A",
    "PAYMENT",
    "PAY",
    "ORDER",
    "COLLECT",
    "SENT",
    "RECEIVED",
    "NA",
    "UPIINTENT",
    "PAYMENT FROM PHONE",
    "PAID",
    "REQUEST",
    "MANDATE",
    "AUTOPAY",
    "IMPS",
    "NEFT",
}
_IFSC = re.compile(r"[A-Z]{4}0[A-Z0-9]{6}")
_BANK_CODES = {
    "YESB",
    "HDFC",
    "ICIC",
    "SBIN",
    "UTIB",
    "KKBK",
    "PYTM",
    "AIRP",
    "IBKL",
    "BARB",
    "PUNB",
    "CNRB",
    "IDFB",
    "INDB",
    "FDRL",
    "AXIS",
    "ICICI",
    "SBI",
    "YBL",
    "OKAXIS",
    "OKICICI",
    "OKSBI",
    "OKHDFCBANK",
    "PAYTM",
}
_STOPWORDS = {
    "PVT",
    "PRIVATE",
    "LTD",
    "LIMITED",
    "LLP",
    "INC",
    "CORP",
    "CO",
    "COM",
    "WWW",
    "IN",
    "INDIA",
    "ONLINE",
    "PAYMENT",
    "PAYMENTS",
    "TECHNOLOGIES",
    "SERVICES",
    "SERVICE",
    "THE",
    "AND",
    "MKTP",
    "IND",
    "INR",
    "TXN",
    "REF",
    "NO",
    "ID",
    "BANGALORE",
    "BENGALURU",
    "MUMBAI",
    "DELHI",
    "NEW",
    "GURGAON",
    "GURUGRAM",
    "NOIDA",
    "PUNE",
    "HYDERABAD",
    "CHENNAI",
    "KOLKATA",
    "AHMEDABAD",
    "JAIPUR",
    "KOCHI",
    "BLR",
    "BOM",
    "DEL",
    "MAA",
    "HYD",
    "CA",
    "NY",
    "US",
    "USA",
    "KA",
    "MH",
    "TN",
    "DL",
    "TS",
    "WB",
    "GJ",
    "XX",
    "UPI",
    "POS",
    "ECOM",
}
_PERSON_HINT = re.compile(r"^[A-Z]{2,15}( [A-Z]{1,15}){1,2}$")
_BUSINESS_HINT = re.compile(
    r"STORE|MART|SHOP|CAFE|RESTAURANT|HOTEL|MEDIC|PHARMA|TRADERS|ENTERPRISE|AGENC|CENTRE|CENTER|BAKERY|SWEETS|"
    r"FOODS|KITCHEN|SALON|CLINIC|HOSPITAL|PETROL|FUEL|MOTORS|TEXTILE|GARMENT|JEWEL|ELECTRONIC|MOBILE|STATION|"
    r"SUPER|BAZAAR|BHAVAN|BHAWAN|DHABA|LTD|PVT|LLP|SOLUTIONS|SERVICES|ACADEMY|SCHOOL|COLLEGE|INSTITUTE|"
    r"STUDIO|BOUTIQUE|DAIRY|FRESH|GYM|FITNESS|BOOK|PHOTO|TRAVEL|TOURS|RENT|AUTO|CABS|LAUNDRY|DRY CLEAN|"
    r"CHARGE|FEE|INTEREST|SALARY|REFUND|ATM|EMI|LOAN|INSURANCE|POLICY|BILL|RECHARGE|TAX|GOVT|MUNICIPAL"
)


def normalize_text(description: str) -> str:
    s = (description or "").upper()
    s = re.sub(r"[^A-Z0-9@&' ]+", " ", s.replace("'", ""))
    s = s.replace("&", " ")
    return re.sub(r"\s+", " ", s).strip()


def _upi_payee(description: str) -> tuple[str | None, str | None]:
    """Return (payee_name, vpa_handle) for UPI-style descriptions."""
    up = description.upper()
    if "UPI" not in up and "@" not in up:
        return None, None
    parts = [p.strip() for p in re.split(r"[/\-|]", description) if p.strip()]
    vpa = next((p for p in parts if "@" in p), None)
    handle = vpa.split("@")[0].upper() if vpa else None
    for p in parts:
        u = p.upper().strip()
        if u in _UPI_NOISE or "@" in u or re.fullmatch(r"[\dX*]+", u) or _IFSC.fullmatch(u):
            continue
        if len(u) <= 2 or u in _BANK_CODES:
            continue
        return u, handle
    return None, handle


def clean_merchant(description: str) -> str:
    """Human-friendly merchant name from a raw description (no lexicon lookup)."""
    payee, handle = _upi_payee(description)
    base = payee or description.upper()
    base = _CHANNEL_PREFIX.sub("", base.strip())
    base = re.sub(r"\*\s*[A-Z0-9]{4,}", " ", base)  # e-commerce order codes ("AMZN MKTP*2K3L19X")
    base = re.sub(r"\b[\dX*]{4,}\b", " ", base)  # card numbers, refs
    base = re.sub(r"[^A-Z0-9& ]+", " ", base)
    tokens = [
        t
        for t in base.split()
        if t not in _STOPWORDS
        and t not in ("CR", "DR")
        and not t.isdigit()
        and len(t) > 1
        and not _IFSC.fullmatch(t)
        and sum(c.isdigit() for c in t) < 3  # drop IFSC codes / reference ids
        and not (len(t) >= 5 and any(c.isdigit() for c in t))  # random alphanumeric order codes
    ]
    if not tokens and handle:
        tokens = [handle]
    name = " ".join(tokens[:3]) if tokens else (description.strip()[:40] or "Unknown")
    return name.title()


def merchant_key(description: str) -> tuple[str, str]:
    """Return (merchant_display_name, normalised_text)."""
    norm = normalize_text(description)
    payee, handle = _upi_payee(description)
    probe = " ".join(x for x in (norm, handle or "") if x)
    rule = match_merchant(probe)
    if rule is not None:
        return rule.name, norm
    return clean_merchant(description), norm


def looks_like_person(description: str) -> bool:
    """Heuristic: a UPI payee that looks like a personal name (P2P transfer), not a business."""
    payee, _ = _upi_payee(description)
    if not payee:
        return False
    return bool(_PERSON_HINT.fullmatch(payee)) and not _BUSINESS_HINT.search(payee)
