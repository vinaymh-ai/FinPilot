"""Synthetic bank-statement generator with ground truth.

Used for (a) the demo account and (b) self-evaluation. Every generated row carries its true
category, whether it belongs to a recurring series, and whether it is an injected anomaly, so we
can *measure* the categorizer, the recurring/anomaly detectors and the forecaster.

Descriptions are deliberately messy (UPI refs, POS card masks, city suffixes) and each persona
includes long-tail local merchants that are NOT in the built-in merchant lexicon — that is where
rules alone fail and an LLM (or user feedback) has to earn its keep.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from finpilot.domain.stats import add_months, days_in_month

CITIES = ["BANGALORE", "BENGALURU", "BLR", "MUMBAI", "PUNE", ""]
BANKS = ["YESB", "HDFC", "ICIC", "SBIN", "UTIB", "KKBK"]


@dataclass
class SynthTxn:
    date: dt.date
    time: dt.time | None
    description: str
    amount: float
    true_category: str
    true_merchant: str
    recurring_key: str | None = None
    anomaly: str | None = None
    impulse: bool = False


@dataclass
class SynthDataset:
    persona: str
    display_name: str
    opening_balance: float
    transactions: list[SynthTxn]
    goals: list[dict]
    recurring_truth: dict[str, dict] = field(default_factory=dict)

    @property
    def closing_balance(self) -> float:
        return round(self.opening_balance + sum(t.amount for t in self.transactions), 2)

    @property
    def end_date(self) -> dt.date:
        return max(t.date for t in self.transactions)


# --------------------------------------------------------------------------------- merchants
@dataclass(frozen=True)
class Merchant:
    name: str  # ground-truth merchant name
    category: str
    raw: str  # how it appears on statements
    channel: str = "upi"  # upi | pos | ecom | nach | neft
    vpa: str | None = None


def M(
    name: str, category: str, raw: str | None = None, channel: str = "upi", vpa: str | None = None
) -> Merchant:
    return Merchant(name, category, raw or name.upper(), channel, vpa)


def _ref(rng: random.Random, n: int = 12) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def describe(m: Merchant, rng: random.Random, note: str = "") -> str:
    city = rng.choice(CITIES)
    if m.channel == "upi":
        vpa = m.vpa or (m.raw.split()[0].lower() + rng.choice(["@ybl", "@okaxis", "@paytm", "@icici"]))
        style = rng.random() * (0.8 if note else 1.0)  # keep the note (e.g. "RENT") when one is given
        if style < 0.5:
            return f"UPI/DR/{_ref(rng)}/{m.raw}/{rng.choice(BANKS)}/{vpa}/{note or 'Payment'}"
        if style < 0.8:
            return f"UPI-{m.raw}-{vpa}-{rng.choice(BANKS)}0000{_ref(rng, 3)}-{_ref(rng)}-{note or 'UPI'}"
        return f"UPI/{_ref(rng)}/{m.raw}/{vpa}"
    if m.channel == "pos":
        return f"POS {_ref(rng, 6)}XXXXXX{_ref(rng, 4)} {m.raw} {city}".strip()
    if m.channel == "ecom":
        return f"ECOM {m.raw}*{''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(7))}"
    if m.channel == "nach":
        return f"NACH DR {m.raw} {_ref(rng, 8)}"
    if m.channel == "neft":
        return f"NEFT CR-{rng.choice(BANKS)}0000{_ref(rng, 3)}-{m.raw}"
    return m.raw


# Spending pools: (merchant, relative weight). Long-tail merchants are intentionally absent
# from the lexicon; some contain a helpful keyword ("MEDICALS"), some do not ("TRUFFLES").
POOLS: dict[str, list[tuple[Merchant, float]]] = {
    "food_delivery": [
        (M("Swiggy", "food_delivery", "SWIGGY", vpa="swiggy@ybl"), 6),
        (M("Zomato", "food_delivery", "ZOMATO", vpa="zomato@hdfcbank"), 3),
        (M("EatSure", "food_delivery", "EATSURE", "ecom"), 0.6),
        (M("Domino's", "food_delivery", "DOMINOS PIZZA", "ecom"), 0.6),
    ],
    "groceries": [
        (M("Blinkit", "groceries", "BLINKIT", vpa="blinkit@icici"), 3),
        (M("Zepto", "groceries", "ZEPTO", vpa="zepto@ybl"), 2.5),
        (M("BigBasket", "groceries", "BIGBASKET", "ecom"), 1.5),
        (M("Swiggy Instamart", "groceries", "SWIGGY INSTAMART", vpa="instamart@ybl"), 1.2),
        (M("Sri Lakshmi Provision Stores", "groceries", "SRI LAKSHMI PROVISION STORES"), 1.2),
        (M("Annapurna Super Market", "groceries", "ANNAPURNA SUPER MARKET", "pos"), 0.8),
        (M("Fresh Farm Vegetables", "groceries", "FRESH FARM VEGETABLES"), 0.8),
        (M("Nandini Milk Parlour", "groceries", "NANDINI MILK PARLOUR"), 0.7),
        (M("Raju Kirana", "groceries", "RAJU KIRANA"), 0.6),
    ],
    "dining": [
        (M("Starbucks", "dining", "TATA STARBUCKS", "pos"), 1.5),
        (M("Third Wave Coffee", "dining", "THIRD WAVE COFFEE", "pos"), 1.2),
        (M("Truffles", "dining", "TRUFFLES", "pos"), 1.0),
        (M("MTR 1924", "dining", "MTR 1924", "pos"), 0.8),
        (M("Hotel Annapoorna", "dining", "HOTEL ANNAPOORNA"), 0.9),
        (M("Vidyarthi Bhavan", "dining", "VIDYARTHI BHAVAN"), 0.6),
        (M("Biryani Zone", "dining", "BIRYANI ZONE"), 0.7),
        (M("The Blue Door Cafe", "dining", "THE BLUE DOOR CAFE", "pos"), 0.6),
        (M("Corner House", "dining", "CORNER HOUSE ICE CREAMS", "pos"), 0.5),
        (M("McDonald's", "dining", "MCDONALDS", "pos"), 0.7),
    ],
    "transport": [
        (M("Uber", "transport", "UBER INDIA", vpa="uber@axisbank"), 3),
        (M("Ola", "transport", "OLA CABS", vpa="olacabs@ybl"), 1.5),
        (M("Rapido", "transport", "RAPIDO", vpa="rapido@ybl"), 2),
        (M("Namma Metro", "transport", "BMRCL NAMMA METRO"), 1.5),
        (M("Shell", "transport", "SHELL INDIA MARKETS", "pos"), 0.7),
        (M("Sri Durga Fuels", "transport", "SRI DURGA FUELS", "pos"), 0.5),
        (M("Kumar Auto Works", "transport", "KUMAR AUTO WORKS"), 0.2),
    ],
    "shopping": [
        (M("Amazon", "shopping", "AMAZON PAY INDIA", "ecom"), 3),
        (M("Flipkart", "shopping", "FLIPKART INTERNET", "ecom"), 2),
        (M("Myntra", "shopping", "MYNTRA DESIGNS", "ecom"), 1.5),
        (M("Chumbak", "shopping", "CHUMBAK DESIGN", "pos"), 0.5),
        (M("Fabindia", "shopping", "FABINDIA OVERSEAS", "pos"), 0.5),
        (M("Poorvika Mobiles", "shopping", "POORVIKA MOBILES", "pos"), 0.3),
        (M("Bata", "shopping", "BATA INDIA FOOTWEAR", "pos"), 0.4),
    ],
    "entertainment": [
        (M("BookMyShow", "entertainment", "BOOKMYSHOW", "ecom"), 2),
        (M("PVR", "entertainment", "PVR INOX", "pos"), 1),
        (M("Steam", "entertainment", "STEAM PURCHASE", "ecom"), 0.6),
        (M("Wonderla", "entertainment", "WONDERLA HOLIDAYS"), 0.3),
        (M("Smaaash", "entertainment", "SMAAASH ENTERTAINMENT"), 0.3),
    ],
    "health": [
        (M("Apollo Pharmacy", "health", "APOLLO PHARMACY", "pos"), 2),
        (M("Shri Ganesh Medicals", "health", "SHRI GANESH MEDICALS"), 1.2),
        (M("City Dental Clinic", "health", "CITY DENTAL CLINIC"), 0.3),
        (M("Sai Diagnostics", "health", "SAI DIAGNOSTICS"), 0.3),
        (M("Tata 1mg", "health", "TATA 1MG", "ecom"), 0.8),
    ],
    "personal_care": [
        (M("Urban Company", "personal_care", "URBAN COMPANY", "ecom"), 1.5),
        (M("Green Trends", "personal_care", "GREEN TRENDS"), 1.0),
        (M("Style Unisex Salon", "personal_care", "STYLE UNISEX SALON"), 0.8),
    ],
    "education": [
        (M("Udemy", "education", "UDEMY ONLINE COURSES", "ecom"), 1),
        (M("Sapna Book House", "education", "SAPNA BOOK HOUSE", "pos"), 0.6),
    ],
    "gifts_donations": [
        (M("Ferns N Petals", "gifts_donations", "FERNS N PETALS", "ecom"), 1),
        (M("ISKCON Temple", "gifts_donations", "ISKCON TEMPLE DONATION"), 0.5),
        (M("Akshaya Patra", "gifts_donations", "AKSHAYA PATRA FOUNDATION"), 0.5),
    ],
    "transfers": [
        (M("Rahul Sharma", "transfers", "RAHUL SHARMA", vpa="rahul.s@okicici"), 1),
        (M("Priya Nair", "transfers", "PRIYA NAIR", vpa="priyanair@oksbi"), 1),
        (M("Karthik Rao", "transfers", "KARTHIK RAO", vpa="karthik@ybl"), 0.7),
    ],
    "cash": [(M("ATM", "cash", "ATM WDL SBI ATM", "pos"), 1)],
}


@dataclass
class VariableSpec:
    category: str
    per_week: float
    mean: float
    sigma: float = 0.55
    weekend_boost: float = 1.0
    night_prob: float = 0.05
    exclude: tuple[str, ...] = ()
    round_to: int = 1


@dataclass
class RecurringSpec:
    key: str
    merchant: Merchant
    amount: float
    day: int
    cadence: str = "monthly"  # monthly | quarterly | yearly
    start_offset: int = 0  # months after history start
    end_offset: int | None = None  # stop after this month index (exclusive)
    jitter_days: int = 1
    amount_sd: float = 0.0  # relative variation (bills)
    hike: tuple[int, float] | None = None  # (month index, new amount)


@dataclass
class PersonaSpec:
    key: str
    display_name: str
    opening_balance: float
    salary: float
    salary_day: int
    employer: str
    recurring: list[RecurringSpec]
    variable: list[VariableSpec]
    side_income: tuple[float, float, float] | None = None  # (monthly prob, low, high)
    salary_sd: float = 0.0
    goals: list[dict] = field(default_factory=list)
    last_month_trend: dict[str, float] = field(
        default_factory=dict
    )  # category -> multiplier in final 5 weeks
    stop_merchants_months_before_end: dict[str, int] = field(default_factory=dict)


def _young_professional() -> PersonaSpec:
    nach = "nach"
    return PersonaSpec(
        key="young_professional",
        display_name="Arjun Mehta",
        opening_balance=85_000,
        salary=115_000,
        salary_day=1,
        employer="ACME TECHNOLOGIES PVT LTD SALARY",
        recurring=[
            RecurringSpec("rent", M("Rent", "rent", "RAMESH KUMAR", vpa="rameshk@oksbi"), 28_000, 3),
            RecurringSpec(
                "electricity",
                M("BESCOM", "utilities", "BESCOM ELECTRICITY BILL", "ecom"),
                1_450,
                11,
                amount_sd=0.18,
            ),
            RecurringSpec("broadband", M("ACT Fibernet", "utilities", "ACT FIBERNET", "ecom"), 1_180, 5),
            RecurringSpec("mobile", M("Airtel", "utilities", "AIRTEL POSTPAID", "ecom"), 599, 15),
            RecurringSpec("sip", M("Groww SIP", "investments", "BSE STAR MF GROWW SIP", nach), 10_000, 7),
            RecurringSpec(
                "netflix", M("Netflix", "subscriptions", "NETFLIX COM", "ecom"), 649, 12, hike=(5, 799)
            ),
            RecurringSpec(
                "hotstar", M("Disney+ Hotstar", "subscriptions", "DISNEY HOTSTAR", "ecom"), 299, 20
            ),
            RecurringSpec("primevideo", M("Prime Video", "subscriptions", "PRIME VIDEO IN", "ecom"), 299, 18),
            RecurringSpec("spotify", M("Spotify", "subscriptions", "SPOTIFY INDIA", "ecom"), 119, 9),
            RecurringSpec(
                "swiggyone", M("Swiggy One", "subscriptions", "SWIGGY ONE MEMBERSHIP", "ecom"), 299, 2
            ),
            RecurringSpec(
                "zomatogold", M("Zomato Gold", "subscriptions", "ZOMATO GOLD MEMBERSHIP", "ecom"), 349, 24
            ),
            RecurringSpec(
                "chatgpt",
                M("ChatGPT Plus", "subscriptions", "OPENAI CHATGPT SUBSCR", "ecom"),
                1_999,
                14,
                start_offset=2,
            ),
            RecurringSpec(
                "cultfit", M("Cult.fit", "health", "CULTFIT CUREFIT HEALTHCARE", "ecom"), 1_499, 25
            ),
            RecurringSpec(
                "health_ins",
                M("Star Health", "insurance", "STAR HEALTH INSURANCE", "ecom"),
                14_500,
                16,
                cadence="yearly",
                start_offset=1,
            ),
        ],
        variable=[
            VariableSpec("food_delivery", 3.6, 430, 0.45, weekend_boost=1.4, night_prob=0.2),
            VariableSpec("groceries", 2.6, 640, 0.6),
            VariableSpec("dining", 1.4, 850, 0.6, weekend_boost=2.0),
            VariableSpec("transport", 4.2, 240, 0.7),
            VariableSpec("shopping", 1.0, 1_600, 0.85, night_prob=0.25),
            VariableSpec("entertainment", 0.45, 750, 0.5, weekend_boost=2.0),
            VariableSpec("health", 0.35, 650, 0.7),
            VariableSpec("personal_care", 0.25, 900, 0.5),
            VariableSpec("transfers", 0.5, 1_200, 0.7, round_to=100),
            VariableSpec("cash", 0.25, 3_000, 0.3, round_to=500),
            VariableSpec("gifts_donations", 0.12, 1_200, 0.5),
            VariableSpec("education", 0.08, 1_100, 0.4),
        ],
        side_income=(0.25, 4_000, 15_000),
        goals=[
            {
                "name": "Goa trip with friends",
                "target_amount": 60_000,
                "saved_amount": 12_000,
                "months_ahead": 2.3,
                "priority": 1,
            },
            {
                "name": "Emergency fund (6 months)",
                "target_amount": 400_000,
                "saved_amount": 110_000,
                "months_ahead": 5.6,
                "priority": 2,
            },
        ],
        last_month_trend={"food_delivery": 1.5},
        stop_merchants_months_before_end={"Zomato": 3},
    )


def _family() -> PersonaSpec:
    return PersonaSpec(
        key="family",
        display_name="Sharma family",
        opening_balance=140_000,
        salary=185_000,
        salary_day=1,
        employer="GLOBEX SOFTWARE INDIA PVT LTD PAYROLL",
        recurring=[
            RecurringSpec("home_loan", M("HDFC Home Loan", "loans", "HDFC LTD LOAN EMI", "nach"), 42_000, 5),
            RecurringSpec(
                "car_loan", M("Bajaj Finance", "loans", "BAJAJ FINANCE LTD EMI", "nach"), 11_500, 10
            ),
            RecurringSpec(
                "electricity",
                M("MSEDCL", "utilities", "MSEDCL ELECTRICITY", "ecom"),
                2_600,
                12,
                amount_sd=0.2,
            ),
            RecurringSpec(
                "gas", M("Mahanagar Gas", "utilities", "MAHANAGAR GAS LTD", "ecom"), 900, 18, amount_sd=0.15
            ),
            RecurringSpec("broadband", M("Jio Fiber", "utilities", "JIO FIBER BROADBAND", "ecom"), 999, 6),
            RecurringSpec("mobile", M("Airtel", "utilities", "AIRTEL POSTPAID FAMILY", "ecom"), 1_349, 15),
            RecurringSpec(
                "school",
                M("School fees", "education", "DPS SCHOOL FEES", "neft"),
                48_000,
                10,
                cadence="quarterly",
            ),
            RecurringSpec("netflix", M("Netflix", "subscriptions", "NETFLIX COM", "ecom"), 649, 3),
            RecurringSpec("hotstar", M("Disney+ Hotstar", "subscriptions", "HOTSTAR", "ecom"), 299, 21),
            RecurringSpec(
                "prime",
                M("Amazon Prime", "subscriptions", "AMAZON PRIME MEMBERSHIP", "ecom"),
                1_499,
                8,
                cadence="yearly",
            ),
            RecurringSpec(
                "life_ins",
                M("LIC", "insurance", "LIC OF INDIA PREMIUM", "ecom"),
                6_200,
                20,
                cadence="quarterly",
            ),
            RecurringSpec("sip", M("Zerodha Coin SIP", "investments", "ZERODHA COIN SIP", "nach"), 15_000, 7),
            RecurringSpec(
                "maid", M("House help", "other", "LAKSHMI DEVI HOUSEHELP", vpa="lakshmi@ybl"), 6_000, 1
            ),
        ],
        variable=[
            VariableSpec("groceries", 4.5, 900, 0.6),
            VariableSpec("food_delivery", 1.5, 650, 0.45, weekend_boost=1.6),
            VariableSpec("dining", 0.9, 1_900, 0.5, weekend_boost=2.5),
            VariableSpec("transport", 2.5, 600, 0.7),
            VariableSpec("shopping", 1.6, 2_100, 0.8, night_prob=0.1),
            VariableSpec("health", 0.7, 900, 0.8),
            VariableSpec("entertainment", 0.35, 1_400, 0.4, weekend_boost=2.0),
            VariableSpec("personal_care", 0.3, 1_200, 0.5),
            VariableSpec("education", 0.25, 1_500, 0.6),
            VariableSpec("cash", 0.4, 4_000, 0.3, round_to=500),
            VariableSpec("gifts_donations", 0.2, 2_000, 0.5),
            VariableSpec("transfers", 0.3, 3_000, 0.6, round_to=500),
        ],
        goals=[
            {
                "name": "Car down-payment",
                "target_amount": 250_000,
                "saved_amount": 60_000,
                "months_ahead": 8,
                "priority": 1,
            }
        ],
        last_month_trend={"shopping": 1.6},
    )


def _freelancer() -> PersonaSpec:
    return PersonaSpec(
        key="freelancer",
        display_name="Neha Iyer",
        opening_balance=95_000,
        salary=70_000,
        salary_day=8,
        employer="UPWORK ESCROW INC PAYOUT",
        salary_sd=0.35,
        recurring=[
            RecurringSpec("rent", M("Rent", "rent", "NOBROKER RENT PAYMENT", "ecom"), 22_000, 2),
            RecurringSpec("coworking", M("WeWork", "other", "WEWORK INDIA", "ecom"), 7_500, 4),
            RecurringSpec("adobe", M("Adobe", "subscriptions", "ADOBE SYSTEMS", "ecom"), 4_230, 9),
            RecurringSpec("notion", M("Notion", "subscriptions", "NOTION LABS", "ecom"), 830, 11),
            RecurringSpec("canva", M("Canva", "subscriptions", "CANVA PTY", "ecom"), 499, 13, end_offset=4),
            RecurringSpec("spotify", M("Spotify", "subscriptions", "SPOTIFY", "ecom"), 119, 17),
            RecurringSpec(
                "youtube", M("YouTube Premium", "subscriptions", "GOOGLE YOUTUBE PREMIUM", "ecom"), 129, 22
            ),
            RecurringSpec("mobile", M("Jio", "utilities", "RELIANCE JIO PREPAID", "ecom"), 349, 26),
            RecurringSpec(
                "electricity", M("BESCOM", "utilities", "BESCOM BILL", "ecom"), 900, 14, amount_sd=0.2
            ),
        ],
        variable=[
            VariableSpec("food_delivery", 4.5, 380, 0.5, night_prob=0.3),
            VariableSpec("groceries", 1.8, 550, 0.6),
            VariableSpec("dining", 1.8, 700, 0.6, weekend_boost=1.6),
            VariableSpec("transport", 3.0, 220, 0.6),
            VariableSpec("shopping", 1.2, 1_400, 0.9, night_prob=0.35),
            VariableSpec("entertainment", 0.5, 600, 0.5),
            VariableSpec("health", 0.3, 700, 0.7),
            VariableSpec("travel", 0.15, 6_000, 0.6),
            VariableSpec("cash", 0.2, 2_000, 0.3, round_to=500),
            VariableSpec("transfers", 0.4, 1_500, 0.7, round_to=100),
        ],
        side_income=(0.6, 15_000, 60_000),
        goals=[
            {
                "name": "New laptop",
                "target_amount": 150_000,
                "saved_amount": 20_000,
                "months_ahead": 4,
                "priority": 1,
            }
        ],
        last_month_trend={"dining": 1.5},
    )


PERSONAS = {"young_professional": _young_professional, "family": _family, "freelancer": _freelancer}

TRAVEL_POOL = [
    (M("IndiGo", "travel", "INDIGO INTERGLOBE AVIATION", "ecom"), 2),
    (M("MakeMyTrip", "travel", "MAKEMYTRIP INDIA", "ecom"), 2),
    (M("OYO", "travel", "OYO ROOMS", "ecom"), 1),
    (M("IRCTC", "travel", "IRCTC E TICKETING", "ecom"), 1.5),
    (M("Zostel", "travel", "ZOSTEL HOSTELS", "ecom"), 0.7),
]
POOLS["travel"] = TRAVEL_POOL


def _pick(pool: list[tuple[Merchant, float]], rng: random.Random, exclude: set[str]) -> Merchant | None:
    items = [(m, w) for m, w in pool if m.name not in exclude]
    if not items:
        return None
    total = sum(w for _, w in items)
    r = rng.random() * total
    for m, w in items:
        r -= w
        if r <= 0:
            return m
    return items[-1][0]


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _lognormal_amount(rng: random.Random, mean: float, sigma: float, round_to: int) -> float:
    mu = math.log(mean) - sigma**2 / 2
    value = rng.lognormvariate(mu, sigma)
    if round_to > 1:
        return max(round_to, round(value / round_to) * round_to)
    return round(value, 0) if value > 200 else round(value, 2)


def _time(rng: random.Random, night: bool) -> dt.time:
    if night:
        h = rng.choice([22, 23, 23, 0, 1, 1, 2])
    else:
        h = int(min(21, max(7, rng.gauss(14, 3.5))))
    return dt.time(h, rng.randrange(60))


def generate(
    persona: str = "young_professional",
    months: int = 8,
    end: dt.date | None = None,
    seed: int = 7,
    anomalies: bool = True,
) -> SynthDataset:
    spec = PERSONAS[persona]()
    rng = random.Random(f"{persona}-{seed}")
    end = end or dt.date.today()
    start = add_months(end.replace(day=1), -(months - 1))
    if months > 1:
        start = start.replace(day=1)
    txns: list[SynthTxn] = []
    trend_from = end - dt.timedelta(days=35)

    # ---- salary + side income
    for mi in range(months):
        mstart = add_months(start, mi)
        pay_day = min(spec.salary_day, days_in_month(mstart.year, mstart.month))
        d = mstart.replace(day=pay_day)
        if d.weekday() == 6:
            d -= dt.timedelta(days=2)
        elif d.weekday() == 5:
            d -= dt.timedelta(days=1)
        if start <= d <= end:
            amt = spec.salary * (1 + rng.gauss(0, spec.salary_sd)) if spec.salary_sd else spec.salary
            amt = round(max(spec.salary * 0.3, amt), 0)
            emp = M("Salary", "income", spec.employer, "neft")
            txns.append(SynthTxn(d, dt.time(9, 5), describe(emp, rng), amt, "income", "Salary", "salary"))
        if spec.side_income and rng.random() < spec.side_income[0]:
            sd = mstart + dt.timedelta(days=rng.randrange(5, 25))
            if sd <= end:
                client = rng.choice(["FIVERR INTL", "TOPTAL LLC", "DEEL INC CONTRACTOR", "UPWORK GLOBAL"])
                amt = round(rng.uniform(spec.side_income[1], spec.side_income[2]), 0)
                txns.append(
                    SynthTxn(
                        sd,
                        dt.time(11, 0),
                        describe(M(client, "income", client, "neft"), rng),
                        amt,
                        "income",
                        "Freelance income",
                    )
                )

    # ---- recurring commitments
    for r in spec.recurring:
        step = {"monthly": 1, "quarterly": 3, "yearly": 12}[r.cadence]
        mi = r.start_offset
        while mi < months and (r.end_offset is None or mi < r.end_offset):
            mstart = add_months(start, mi)
            day = min(r.day, days_in_month(mstart.year, mstart.month))
            d = mstart.replace(day=day) + dt.timedelta(days=rng.randint(-r.jitter_days, r.jitter_days))
            if start <= d <= end:
                amount = r.hike[1] if r.hike and mi >= r.hike[0] else r.amount
                if r.amount_sd:
                    amount = round(amount * max(0.4, rng.gauss(1, r.amount_sd)), 0)
                note = "RENT" if r.key == "rent" else ""
                txns.append(
                    SynthTxn(
                        d,
                        _time(rng, False),
                        describe(r.merchant, rng, note),
                        -float(amount),
                        r.merchant.category,
                        r.merchant.name,
                        recurring_key=r.key,
                    )
                )
            mi += step

    # ---- variable spending
    day = start
    stop_dates = {m: add_months(end, -k) for m, k in spec.stop_merchants_months_before_end.items()}
    while day <= end:
        weekend = day.weekday() >= 5
        for v in spec.variable:
            lam = v.per_week / 7 * (v.weekend_boost if weekend else 1.0)
            if day >= trend_from:
                lam *= spec.last_month_trend.get(v.category, 1.0)
            for _ in range(_poisson(rng, lam)):
                exclude = {m for m, sd in stop_dates.items() if day >= sd}
                m = _pick(POOLS[v.category], rng, exclude)
                if m is None:
                    continue
                night = rng.random() < v.night_prob
                amount = _lognormal_amount(rng, v.mean, v.sigma, v.round_to)
                impulse = night and v.category in ("shopping", "food_delivery") and amount > v.mean * 0.8
                txns.append(
                    SynthTxn(
                        day, _time(rng, night), describe(m, rng), -amount, m.category, m.name, impulse=impulse
                    )
                )
        # occasional incoming transfer from friends
        if rng.random() < 0.04:
            m = _pick(POOLS["transfers"], rng, set())
            assert m is not None
            amt = round(rng.uniform(300, 2500) / 50) * 50
            desc = f"UPI/CR/{_ref(rng)}/{m.raw}/{rng.choice(BANKS)}/{m.vpa}/Split"
            txns.append(SynthTxn(day, _time(rng, False), desc, float(amt), "transfers", m.name))
        day += dt.timedelta(days=1)

    # ---- one trip somewhere in the middle of history
    trip_day = start + dt.timedelta(days=int((end - start).days * rng.uniform(0.3, 0.6)))
    for m, base in [(TRAVEL_POOL[0][0], 7_800), (TRAVEL_POOL[2][0], 4_600)]:
        txns.append(
            SynthTxn(
                trip_day,
                _time(rng, False),
                describe(m, rng),
                -float(round(base * rng.uniform(0.9, 1.2))),
                "travel",
                m.name,
            )
        )

    # ---- injected anomalies (ground truth for the detector evaluation)
    if anomalies:
        a_day = end - dt.timedelta(days=rng.randint(6, 20))
        croma = M("Croma", "shopping", "CROMA INFINITI RETAIL", "pos")
        txns.append(
            SynthTxn(
                a_day,
                dt.time(23, 41),
                describe(croma, rng),
                -54_990.0,
                "shopping",
                "Croma",
                anomaly="large_purchase",
                impulse=True,
            )
        )
        d_day = end - dt.timedelta(days=rng.randint(2, 12))
        swg = POOLS["food_delivery"][0][0]
        amt = -float(round(rng.uniform(450, 700)))
        txns.append(SynthTxn(d_day, dt.time(20, 14), describe(swg, rng), amt, "food_delivery", "Swiggy"))
        txns.append(
            SynthTxn(
                d_day,
                dt.time(20, 15),
                describe(swg, rng),
                amt,
                "food_delivery",
                "Swiggy",
                anomaly="duplicate_charge",
            )
        )
        f_day = end - dt.timedelta(days=rng.randint(15, 40))
        fee = M("Late fee", "fees", "LATE PAYMENT FEE CREDIT CARD", "pos")
        txns.append(
            SynthTxn(f_day, dt.time(4, 0), f"{fee.raw} GST INCL", -1_180.0, "fees", "Bank fee", anomaly="fee")
        )
        u_day = end - dt.timedelta(days=rng.randint(25, 60))
        unk = M("Vivid Interiors", "shopping", "VIVID INTERIORS AND DECOR")
        txns.append(
            SynthTxn(
                u_day,
                dt.time(16, 30),
                describe(unk, rng),
                -18_500.0,
                "shopping",
                "Vivid Interiors",
                anomaly="large_new_merchant",
            )
        )

    txns = [t for t in txns if start <= t.date <= end]
    txns.sort(key=lambda t: (t.date, t.time or dt.time(0, 0)))

    truth = {
        r.key: {
            "merchant": r.merchant.name,
            "cadence": r.cadence,
            "amount": r.amount,
            "category": r.merchant.category,
        }
        for r in spec.recurring
    }
    truth["salary"] = {
        "merchant": "Salary",
        "cadence": "monthly",
        "amount": spec.salary,
        "category": "income",
    }
    goals = []
    for g in spec.goals:
        deadline = end + dt.timedelta(days=int(g["months_ahead"] * 30.4))
        goals.append(
            {
                "name": g["name"],
                "target_amount": g["target_amount"],
                "saved_amount": g["saved_amount"],
                "deadline": deadline.isoformat(),
                "priority": g["priority"],
            }
        )
    return SynthDataset(spec.key, spec.display_name, spec.opening_balance, txns, goals, truth)


def write_csv(ds: SynthDataset, path: str | Path, with_truth: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    balance = ds.opening_balance
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["date", "time", "description", "amount", "balance"]
        if with_truth:
            header += ["true_category", "true_merchant", "recurring_key", "anomaly", "impulse"]
        w.writerow(header)
        for t in ds.transactions:
            balance += t.amount
            row = [
                t.date.isoformat(),
                t.time.strftime("%H:%M") if t.time else "",
                t.description,
                f"{t.amount:.2f}",
                f"{balance:.2f}",
            ]
            if with_truth:
                row += [
                    t.true_category,
                    t.true_merchant,
                    t.recurring_key or "",
                    t.anomaly or "",
                    int(t.impulse),
                ]
            w.writerow(row)
    return path
