"""Hand-labelled bank narrations in the formats Indian banks actually print.

Each case is (narration, signed amount, expected category, expected payment channel). The
narrations imitate HDFC, ICICI, SBI, Axis and Kotak statement styles (identifiers changed) and mix
well-known brands with local merchants, person-to-person payments, intermediaries and bank
entries. Labels were written before the set was first measured and are not tuned afterwards;
`run_eval` reports the deterministic categorizer's accuracy on them as a held-out check that is
independent of the synthetic generator.

Labelling conventions (the taxonomy's, not the rules'):
* a payment to an individual with no stated purpose is `transfers` (P2P); a stated purpose wins
  ("RENT" → rent);
* a payment through an intermediary (CRED, Razorpay, PayU) is labelled by its stated purpose, or
  `other` when the narration gives none;
* an explicit card-bill payment is `credit_card_payment`.
"""

from __future__ import annotations

NARRATION_CASES: list[tuple[str, float, str, str]] = [
    # ---------------------------------------------------------------- HDFC Bank style
    ("UPI-SWIGGY-swiggy@icici-ICIC0DC0099-412345678901-UPI", -486.0, "food_delivery", "upi"),
    (
        "UPI-ZOMATO LTD-zomato.payu@axisbank-UTIB0000100-412345678902-ZOMATO ORDER",
        -612.5,
        "food_delivery",
        "upi",
    ),
    ("POS 512345XXXXXX1234 DMART AVENUE SUPERMA BANGALORE", -2314.0, "groceries", "card"),
    ("POS 512345XXXXXX1234 SHELL INDIA MARKETS PUNE", -1500.0, "transport", "card"),
    ("NEFT CR-CITI0000001-GLOBEX SOFTWARE PVT LTD-SALARY SEP 2026", 142000.0, "income", "neft"),
    ("ACH D- BAJAJ FINANCE LTD-P400ABC1234567", -6230.0, "loans", "auto_debit"),
    ("ATW-512345XXXXXX1234-S1AWBL01-BANGALORE", -5000.0, "cash", "atm"),
    ("IMPS-412345678903-SURESH REDDY-HDFC-XXXXXXXX1234-HOUSE RENT OCT", -22000.0, "rent", "imps"),
    (".INT.PD:01-07-2026 TO 30-09-2026", 1834.0, "income", "bank"),
    ("UPI-BLINKIT-blinkit.payu@hdfcbank-HDFC0MERUPI-412345678904-UPI", -742.0, "groceries", "upi"),
    (
        "UPI-ANJALI MEHTA-anjalimehta@okhdfcbank-HDFC0000123-412345678905-DINNER SPLIT",
        850.0,
        "transfers",
        "upi",
    ),
    ("POS 512345XXXXXX1234 MCDONALDS HARDCASTLE MUMBAI", -389.0, "dining", "card"),
    (
        "UPI-UBER INDIA SYSTEMS PVT-uber.payu@axisbank-UTIB0000100-412345678906-UPI",
        -264.0,
        "transport",
        "upi",
    ),
    ("ECOM PUR 512345XXXXXX1234 NETFLIX COM 08-10-2026", -649.0, "subscriptions", "card"),
    ("NACH DR ICICI PRUDENTIAL LIFE INSURANCE 4123456", -3500.0, "insurance", "auto_debit"),
    (
        "UPI-ACT FIBERNET-actfibernet@icici-ICIC0DC0099-412345678907-BROADBAND BILL",
        -1180.0,
        "utilities",
        "upi",
    ),
    ("POS 512345XXXXXX1234 CROMA RETAIL BANGALORE", -45990.0, "shopping", "card"),
    ("UPI-PVR INOX LIMITED-pvrinox@kotak-KKBK0000958-412345678908-UPI", -780.0, "entertainment", "upi"),
    ("UPI-APOLLO PHARMACY-apollopharmacy@hdfcbank-HDFC0MERUPI-412345678909-UPI", -436.0, "health", "upi"),
    ("FT - DR - 50100123456789 - VIKRAM SINGH - TRANSFER", -10000.0, "transfers", "other"),
    # ---------------------------------------------------------------- ICICI Bank style
    ("UPI/412345678910/Payment from Ph/swiggy@icici/ICICI Bank", -355.0, "food_delivery", "upi"),
    ("MMT/IMPS/412345678911/Rent Oct/KAVITA RAO/HDFC Bank", -18500.0, "rent", "imps"),
    ("BIL/INFT/001234567890/ICICI CREDIT CARD/CC XX1234", -24567.0, "credit_card_payment", "netbanking"),
    ("VPS/ZOMATO MEDIA PVT/202610121234/ZOMATO GOLD", -149.0, "subscriptions", "card"),
    ("NWD-512345XXXXXX1234-SPCND123-BANGALORE", -2000.0, "cash", "atm"),
    ("CMS/ ACH/ZERODHA BROKING LTD/ZER123456", -15000.0, "investments", "auto_debit"),
    ("UPI/412345678912/IRCTC TICKET/irctc@sbi/State Bank of India", -1265.0, "travel", "upi"),
    ("UPI/412345678913/RELIANCE JIO/jio@axisbank/Axis Bank/RECHARGE", -299.0, "utilities", "upi"),
    (
        "UPI/412345678914/Payment from Ph/cred.club@axisb/Axis Bank/CREDIT CARD BILL",
        -12000.0,
        "credit_card_payment",
        "upi",
    ),
    ("UPI/412345678915/Payment from Ph/razorpay@icici/ICICI Bank", -2999.0, "other", "upi"),
    ("IPS/AMAZON SELLER SERVICES/202610111234/", -1899.0, "shopping", "card"),
    ("UPI/412345678916/NAMMA METRO/bmrcl@ybl/Yes Bank", -60.0, "transport", "upi"),
    ("UPI/412345678917/RAHUL VERMA/rahulv@okicici/ICICI Bank", -1500.0, "transfers", "upi"),
    ("INF/INFT/001234567891/SELF TRANSFER/HDFC BANK", -50000.0, "transfers", "netbanking"),
    ("UPI/412345678918/BOOKMYSHOW/bookmyshow@hdfcbank/HDFC Bank", -540.0, "entertainment", "upi"),
    ("UPI/412345678919/LIC OF INDIA/licpremium@sbi/State Bank of India/PREMIUM", -8400.0, "insurance", "upi"),
    ("REV/UPI/412345678920/AMAZON REFUND/amazon@apl", 1299.0, "shopping", "upi"),
    ("ATD/AUTO DEBIT CC0XX1234/ICICI BANK CREDIT CARD", -9870.0, "credit_card_payment", "auto_debit"),
    ("BIL/ONL/000123456789/BESCOM/ELECTRICITY BILL", -1874.0, "utilities", "netbanking"),
    ("UPI/412345678921/CULT FIT/cultfit@icici/ICICI Bank", -1999.0, "health", "upi"),
    # ---------------------------------------------------------------- SBI style
    ("TO TRANSFER-UPI/DR/412345678922/RAMESH K/SBIN/rameshk@oksbi/rent", -15000.0, "rent", "upi"),
    ("BY TRANSFER-NEFT*HDFC0000001*N123456789*INITECH SOLUTIONS*", 98000.0, "income", "neft"),
    ("ATM WDL-ATM CASH 1234 SBI KORAMANGALA", -3000.0, "cash", "atm"),
    ("BY CASH DEPOSIT-CDM BRANCH 0123", 20000.0, "transfers", "cash_deposit"),
    ("CREDIT INTEREST", 412.0, "income", "bank"),
    ("TO TRANSFER-UPI/DR/412345678923/ZEPTO/YESB/zepto@ybl/Payment", -567.0, "groceries", "upi"),
    ("TO TRANSFER-UPI/DR/412345678924/MEESHO/ICIC/meesho@icici/Payment", -449.0, "shopping", "upi"),
    ("TO TRANSFER-UPI/DR/412345678925/BIGBASKET/ICIC/bigbasket@icici/Payment", -1645.0, "groceries", "upi"),
    ("DEBIT-ATMCard AMC 412345*1234", -236.0, "fees", "bank"),
    (
        "TO TRANSFER-UPI/DR/412345678926/SRI VENKATESHWARA MEDICALS/SBIN/9876543210@ybl/Payment",
        -310.0,
        "health",
        "upi",
    ),
    ("TO TRANSFER-UPI/DR/412345678927/ANAND BHAVAN/SBIN/anandbhavan@ybl/Payment", -220.0, "dining", "upi"),
    ("TO TRANSFER-UPI/DR/412345678928/INDIAN OIL CORP/SBIN/iocl@sbi/Payment", -2000.0, "transport", "upi"),
    ("BY TRANSFER-UPI/CR/412345678929/PRIYA SHARMA/SBIN/priya@oksbi/Payment", 2500.0, "transfers", "upi"),
    ("TO TRANSFER-UPI/DR/412345678930/KSRTC/SBIN/ksrtc@sbi/Payment", -845.0, "travel", "upi"),
    ("SMS CHARGES FOR QTR ENDED SEP 2026", -17.7, "fees", "bank"),
    ("TO TRANSFER-UPI/DR/412345678931/HP GAS/SBIN/hpgas@sbi/Payment", -905.0, "utilities", "upi"),
    (
        "TO TRANSFER-UPI/DR/412345678932/GIVEINDIA/SBIN/giveindia@icici/Payment",
        -1000.0,
        "gifts_donations",
        "upi",
    ),
    ("MIN BAL CHARGES SEP 2026", -295.0, "fees", "bank"),
    ("TO TRANSFER-UPI/DR/412345678933/LENSKART/SBIN/lenskart@icici/Payment", -2400.0, "shopping", "upi"),
    ("TO TRANSFER-UPI/DR/412345678934/UDEMY/SBIN/udemy@icici/Payment", -499.0, "education", "upi"),
    # ---------------------------------------------------------------- Axis Bank style
    ("UPI/P2M/412345678935/Blinkit/Paid via/AXIS BANK", -312.0, "groceries", "upi"),
    ("POS/RAPIDO BIKE TAXI/BANGALORE/412345", -78.0, "transport", "card"),
    ("NEFT/AXISN123456789/SALARY/UMBRELLA CORP", 87500.0, "income", "neft"),
    ("ECOM PUR/SPOTIFY INDIA/412345XXXXXX1234", -119.0, "subscriptions", "card"),
    ("UPI/P2A/412345678936/ARJUN NAIR/Paid via/HDFC BANK", -700.0, "transfers", "upi"),
    ("UPI/P2M/412345678937/STARBUCKS COFFEE/Paid via/ICICI BANK", -455.0, "dining", "upi"),
    ("UPI/P2M/412345678938/URBAN COMPANY/Paid via/AXIS BANK", -899.0, "personal_care", "upi"),
    ("UPI/P2M/412345678939/MAKEMYTRIP/Paid via/AXIS BANK", -12450.0, "travel", "upi"),
    ("UPI/P2M/412345678940/NYKAA/Paid via/AXIS BANK", -1340.0, "personal_care", "upi"),
    ("UPI/P2M/412345678941/GROWW/Paid via/AXIS BANK/SIP", -5000.0, "investments", "upi"),
    ("POS/DECATHLON SPORTS INDIA/BANGALORE/412345", -2799.0, "shopping", "card"),
    ("UPI/P2M/412345678942/FERNS N PETALS/Paid via/AXIS BANK", -1199.0, "gifts_donations", "upi"),
    ("UPI/P2M/412345678943/AIRTEL PAYMENTS/Paid via/AXIS BANK/POSTPAID", -599.0, "utilities", "upi"),
    ("UPI/P2M/412345678944/OLA CABS/Paid via/AXIS BANK", -312.0, "transport", "upi"),
    ("UPI/P2M/412345678945/INDIGO AIRLINES/Paid via/AXIS BANK", -6780.0, "travel", "upi"),
    ("ECOM PUR/YOUTUBE PREMIUM/412345XXXXXX1234", -149.0, "subscriptions", "card"),
    ("UPI/P2M/412345678946/DOMINOS PIZZA/Paid via/AXIS BANK", -599.0, "food_delivery", "upi"),
    ("UPI/P2M/412345678947/CHAAYOS/Paid via/AXIS BANK", -260.0, "dining", "upi"),
    ("UPI/P2M/412345678948/THYROCARE/Paid via/AXIS BANK", -1100.0, "health", "upi"),
    ("UPI/P2M/412345678949/PAYU/Paid via/AXIS BANK", -1450.0, "other", "upi"),
    # ---------------------------------------------------------------- Kotak style
    ("UPI/ZOMATO LTD/412345678950/Payment", -432.0, "food_delivery", "upi"),
    ("PCD/1234/APOLLO PHARMACY/BANGALORE", -712.0, "health", "card"),
    ("NACH-DR-ICCL-MF SIP-412345678", -3000.0, "investments", "auto_debit"),
    ("IMPS/P2A/412345678951/KARTHIK/HDFC/Lunch", -350.0, "transfers", "imps"),
    ("UPI/MYNTRA DESIGNS/412345678952/Payment", -1599.0, "shopping", "upi"),
    ("UPI/TATA PLAY/412345678953/DTH RECHARGE", -450.0, "utilities", "upi"),
    ("UPI/SWIGGY INSTAMART/412345678954/Payment", -678.0, "groceries", "upi"),
    ("UPI/OYO ROOMS/412345678955/Payment", -2340.0, "travel", "upi"),
    ("UPI/NOBROKER/412345678956/RENT PAYMENT", -25000.0, "rent", "upi"),
    ("UPI/STAR HEALTH INSURANCE/412345678957/RENEWAL", -11800.0, "insurance", "upi"),
    ("UPI/DREAM11/412345678958/Payment", -500.0, "entertainment", "upi"),
    ("UPI/CASHBACK/412345678959/GPAY REWARD", 51.0, "income", "upi"),
    ("UPI/FLIPKART/412345678960/Payment", -3499.0, "shopping", "upi"),
    ("UPI/PHARMEASY/412345678961/Payment", -890.0, "health", "upi"),
    ("UPI/RAPIDO/412345678962/Payment", -95.0, "transport", "upi"),
    ("UPI/SUNITA DEVI/9876501234@ybl/Payment", -1200.0, "transfers", "upi"),
    ("UPI/BURGER KING/412345678963/Payment", -349.0, "dining", "upi"),
    ("UPI/UPWORK ESCROW/412345678964/PAYOUT", 45000.0, "income", "upi"),
    ("LATE PAYMENT FEE CC XX1234", -750.0, "fees", "other"),
    ("UPI/GOKIWI/412345678965/ICICI CRED CARD", -8000.0, "credit_card_payment", "upi"),
]

# The first measurement, taken before any rule was changed in response to this set. Fixes made
# afterwards were generic parsing bugs it exposed (bank notes such as "Payment from Ph" / "BY
# TRANSFER" read as payees, IMPS "P2A" person-to-account payments, HDFC "ATW" ATM withdrawals, card
# auto-debits, company NEFT credits, "INF"/"CREDIT INTEREST" channels). `run_eval` reports both the
# first-run figures and the current ones; only the first run is a held-out estimate.
FIRST_RUN = {
    "date": "2026-10-14",
    "category_accuracy": 0.91,
    "channel_accuracy": 0.97,
    "confident_share": 0.84,
    "confident_precision": 0.988,
}
