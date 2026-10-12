import datetime as dt

from finpilot.domain.normalize import looks_like_person, merchant_key
from finpilot.domain.recurring import TxnLite, detect_recurring
from finpilot.llm.client import extract_json
from finpilot.llm.grounding import check_grounding, extract_numbers


def test_merchant_key_handles_bank_formats():
    assert merchant_key("UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/Payment")[0] == "Swiggy"
    assert merchant_key("POS 445566XXXXXX1234 AMZN Mktp IN*2K3L19X BANGALORE")[0] == "Amazon"
    assert merchant_key("NETFLIX.COM 866-579-7172 CA")[0] == "Netflix"
    assert merchant_key("SWIGGY INSTAMART BLR")[0] == "Swiggy Instamart"  # specific rule beats brand rule
    assert merchant_key("SWIGGY ONE MEMBERSHIP")[0] == "Swiggy One"
    assert merchant_key("ECOM WEWORK INDIA*Y4YYKMA")[0] == "Wework"  # order codes stripped
    assert merchant_key("NEFT CR-HDFC0000123-DPS SCHOOL FEES")[0] == "Dps School Fees"  # IFSC stripped


def test_p2p_detection():
    assert looks_like_person("UPI/412345678901/RAHUL SHARMA/rahul@okicici")
    assert not looks_like_person("UPI/412345678901/SHRI GANESH MEDICALS/ganesh@ybl")


def _monthly(merchant, amount, n, start=dt.date(2026, 1, 5), cat="subscriptions", id0=0):
    from finpilot.domain.stats import add_months

    return [TxnLite(id0 + i, add_months(start, i), amount, merchant, cat) for i in range(n)]


def test_detects_monthly_subscription_and_price_hike():
    txns = _monthly("Netflix", -649, 4) + _monthly("Netflix", -799, 2, start=dt.date(2026, 5, 5), id0=10)
    series = detect_recurring(txns, as_of=dt.date(2026, 6, 20))
    assert len(series) == 1
    s = series[0]
    assert s.cadence == "monthly" and s.kind == "subscription"
    assert s.price_change_pct and s.price_change_pct > 20
    assert s.next_expected == dt.date(2026, 7, 5)


def test_busy_merchant_coincidences_are_not_recurring():
    # many food orders with similar amounts must not produce a "subscription"
    txns = [
        TxnLite(i, dt.date(2026, 1, 1) + dt.timedelta(days=3 * i), -450 - (i % 3), "Swiggy", "food_delivery")
        for i in range(40)
    ]
    assert detect_recurring(txns) == []


def test_fixed_fee_inside_busy_merchant_is_found():
    orders = [
        TxnLite(
            100 + i, dt.date(2026, 1, 2) + dt.timedelta(days=4 * i), -(300 + 37 * i), "Amazon", "shopping"
        )
        for i in range(40)
    ]
    prime = _monthly("Amazon", -299.0, 6, cat="shopping")
    series = detect_recurring(orders + prime, as_of=dt.date(2026, 6, 10))
    assert [s.typical_amount for s in series] == [-299.0]


def test_lapsed_status():
    txns = _monthly("Canva", -499, 4)
    series = detect_recurring(txns, as_of=dt.date(2026, 9, 1))
    assert series[0].status == "lapsed"


def test_grounding_accepts_supplied_and_rejects_invented_numbers():
    facts = {"saving": 598.0, "items": [{"monthly_cost": 299.0}], "change_pct": 47.2}
    ok, bad = check_grounding(
        "Cancel two services to save ₹598/month (₹7,176 a year), spending is up 47%.", facts
    )
    assert ok, bad
    ok, bad = check_grounding("This saves ₹2,500 every month.", facts)
    assert not ok and bad == [2500.0]


def test_number_extraction_formats():
    vals = [v for v, _ in extract_numbers("₹1,20,000 and 1.5L and 12k and 40%")]
    assert vals == [120000.0, 150000.0, 12000.0, 40.0]


def test_extract_json_tolerates_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"b": [1,2]} thanks') == {"b": [1, 2]}


def test_grounding_does_not_mistake_money_for_a_year():
    facts = {"low_point": 8200, "deadline": "2026-12-20"}
    assert not check_grounding("Your balance could fall to ₹2,000 in November.", facts)[0]
    assert check_grounding("By December 2026 you could be at ₹8,200.", facts)[0]
