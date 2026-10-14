"""Second real-world narration benchmark: breadth across banks and transaction *types*.

The first set (`narration_cases.py`) checked common merchants. This one asks "can it classify any
kind of bank transaction?": salary and dividends, interest, tax payments and refunds, bank charges
and GST on charges, reversals and refunds, self-transfers and wallet loads, SIPs, PPF/NPS, digital
gold, EMIs, insurance, card bills, rent platforms, FASTag, metro, school fees, donations — in the
narration styles of SBI, HDFC, ICICI, Axis, Kotak, PNB, Bank of Baroda, Canara, Union, IDFC First,
Yes, IndusInd and Federal Bank.

Written before running the categorizer on it; the first-run score is recorded in `FIRST_RUN` and
reported next to the current one. Channel is graded only where it is unambiguous (None = not graded).
"""

from __future__ import annotations

# (narration, signed amount, expected category, expected channel | None)
NARRATION_CASES_2: list[tuple[str, float, str, str | None]] = [
    # ---- income --------------------------------------------------------------------------------
    ("NEFT CR-ICIC0000001-INFOSYS LIMITED-SALARY SEP 2026", 142000, "income", "neft"),
    ("BY TRANSFER-NEFT*ICIC0000104*CMS1234567*ACME TECHNOLOGIES PVT LTD SALARY--", 98000, "income", "neft"),
    ("CMS TRANSACTION/WIPRO LIMITED/SALARY FOR SEPTEMBER", 86500, "income", None),
    ("ACH C- INFOSYS LTD DIVIDEND-123456", 1840, "income", "auto_debit"),
    ("ACH CR TCS LTD FINAL DIVIDEND 2026", 720, "income", "auto_debit"),
    ("CREDIT INTEREST CAPITALISED", 412.55, "income", "bank"),
    ("Int.Pd:0123:01-07-2026 to 30-09-2026", 1290.4, "income", None),
    ("INT.CREDIT", 233.1, "income", None),
    ("NEFT-IN-CBDT INCOME TAX REFUND AY 2026-27", 18450, "income", "neft"),
    ("ITD REFUND CREDIT 2026", 9600, "income", None),
    ("UPI/CR/412345678940/UPWORK ESCROW/HDFC/upwork@hdfcbank/Payment", 36000, "income", "upi"),
    ("PAYTM CASHBACK RECEIVED", 50, "income", None),
    # ---- transfers -----------------------------------------------------------------------------
    ("FT - DR - 50100123456789 - SELF", -20000, "transfers", None),
    ("INF/INFT/123456789/SELF TRANSFER", -15000, "transfers", "netbanking"),
    ("TO OWN ACCOUNT 1234567890", -10000, "transfers", None),
    ("IMPS-412345678901-PRIYA SHARMA-SBIN-XXXX1234-GIFT", -2000, "gifts_donations", "imps"),
    ("UPI/DR/412345678911/RAMESH KUMAR/SBIN/9845011111@ybl/Payment", -700, "transfers", "upi"),
    ("UPI/CR/412345678912/ANITHA RAO/ICIC/anitha.r@okicici/Payment", 1500, "transfers", "upi"),
    ("PAYTM WALLET TOPUP", -2000, "transfers", None),
    ("AMAZON PAY BALANCE LOAD", -1000, "transfers", None),
    ("BY CASH DEPOSIT CDM 1234 BANGALORE", 25000, "transfers", "cash_deposit"),
    # ---- investments ---------------------------------------------------------------------------
    (".ACH D- ICICI PRUDENTIAL MUTUAL FUND-SIP123", -5000, "investments", "auto_debit"),
    ("NACH-DR-BSE STAR MF-GROWW-SIP", -3000, "investments", "auto_debit"),
    ("NACH/ZERODHA BROKING LTD/FUNDS ADDED", -25000, "investments", "auto_debit"),
    ("UPI/DR/412345678913/ZERODHA/HDFC/zerodha@hdfcbank/Add funds", -10000, "investments", "upi"),
    ("PPF DEPOSIT A/C 1234567", -12500, "investments", None),
    ("NPS CONTRIBUTION NSDL CRA", -5000, "investments", None),
    ("Sweep Trf To: 1234567890 (FD)", -50000, "investments", None),
    ("RD INSTALLMENT 12345", -2000, "investments", None),
    ("UPI/DR/412345678914/SAFEGOLD/YESB/safegold@ybl/Digital gold", -1000, "investments", "upi"),
    ("SOVEREIGN GOLD BOND SUBSCRIPTION", -6300, "investments", None),
    # ---- loans ---------------------------------------------------------------------------------
    ("ACH D- BAJAJ FINANCE LTD-P400ABCD1234", -4560, "loans", "auto_debit"),
    ("DEBIT-ACHDr HDFC0000001 HDFC BANK HOME LOAN EMI", -38500, "loans", "auto_debit"),
    ("NACH-DR-TPCODE-TATA CAPITAL CAR LOAN", -12750, "loans", "auto_debit"),
    ("UPI/DR/412345678915/KREDITBEE/YESB/kreditbee@ybl/EMI", -2300, "loans", "upi"),
    ("LOAN EMI RECOVERY 123456789", -9100, "loans", None),
    # ---- insurance -----------------------------------------------------------------------------
    ("NACH-DR-TPCODE-LIC OF INDIA", -3480, "insurance", "auto_debit"),
    ("BIL/ONL/000123456/HDFC LIFE INSURANCE/PREMIUM", -10500, "insurance", "netbanking"),
    ("UPI/DR/412345678916/STAR HEALTH/ICIC/starhealth@icici/Renewal", -18200, "insurance", "upi"),
    ("POS 416021XXXXXX1234 ACKO GENERAL INSURANCE", -4200, "insurance", "card"),
    ("UPI/DR/412345678917/POLICYBAZAAR/YESB/pbazaar@ybl/Policy", -7400, "insurance", "upi"),
    # ---- credit-card bills ---------------------------------------------------------------------
    ("UPI/DR/412345678918/CRED CLUB/AXIS/cred.club@axisb/HDFC CC BILL", -15600, "credit_card_payment", "upi"),
    ("IB BILLPAY DR-HDFCQH-123456XXXXXX1234", -21000, "credit_card_payment", "netbanking"),
    ("ATD/Auto Debit CC0xx1234", -8900, "credit_card_payment", "auto_debit"),
    ("BIL/ONL/000123456/SBI CARD PAYMENT", -6700, "credit_card_payment", "netbanking"),
    # ---- rent ----------------------------------------------------------------------------------
    ("UPI/DR/412345678919/NOBROKER/YESB/nobroker@ybl/Rent Oct", -22000, "rent", "upi"),
    ("MMT/IMPS/412345678920/RENT OCT/RAMESH KUMAR/HDFC0001234", -18000, "rent", "imps"),
    (
        "UPI/DR/412345678921/PRESTIGE APARTMENT OWNERS/HDFC/prestige@hdfcbank/Maintenance",
        -3500,
        "rent",
        "upi",
    ),
    # ---- utilities -----------------------------------------------------------------------------
    ("BIL/ONL/001234567890/BESCOM/BES1234", -1430, "utilities", "netbanking"),
    ("BBPS/TATA POWER DDL/ELECTRICITY BILL", -2210, "utilities", None),
    ("UPI/DR/412345678922/MSEDCL/SBIN/msedcl@sbi/Electricity", -1640, "utilities", "upi"),
    ("BBPS-BWSSB WATER BILL 123", -420, "utilities", None),
    ("UPI/DR/412345678923/MAHANAGAR GAS/ICIC/mgl@icici/Gas bill", -760, "utilities", "upi"),
    ("POS 416021XXXXXX1234 ACT FIBERNET", -1179, "utilities", "card"),
    ("UPI/DR/412345678924/JIO RECHARGE/YESB/jio@ybl/Recharge", -349, "utilities", "upi"),
    ("SI-AIRTEL POSTPAID 9845012345", -599, "utilities", None),
    ("UPI/DR/412345678925/TATA PLAY/YESB/tataplay@ybl/DTH", -420, "utilities", "upi"),
    # ---- taxes ---------------------------------------------------------------------------------
    ("TAX PAYMENT-CBDT-ITNS280-ADVANCE TAX", -25000, "taxes", None),
    ("OLTAS CHALLAN 280 SELF ASSESSMENT TAX", -8200, "taxes", None),
    ("GST PAYMENT CPIN 26091234567", -14560, "taxes", None),
    ("UPI/DR/412345678926/BBMP PROPERTY TAX/SBIN/bbmp@sbi/Property tax", -6400, "taxes", "upi"),
    ("BIL/ONL/000123456/TRAFFIC CHALLAN/KARNATAKA POLICE", -500, "taxes", "netbanking"),
    # ---- bank fees -----------------------------------------------------------------------------
    ("SMS Alert Charges Jul-Sep 2026", -17.7, "fees", None),
    ("ATM WDL CHARGES 3 TXNS", -63, "fees", None),
    ("MIN BAL CHGS SEP 2026", -590, "fees", None),
    ("Debit Card Annual Fee", -499, "fees", None),
    ("GST @18% on Charge", -89.82, "fees", None),
    ("CHQ RETURN CHARGES 000123", -354, "fees", None),
    ("IMPS CHARGES", -5.9, "fees", None),
    # ---- cash ----------------------------------------------------------------------------------
    ("ATM WDL-ATM CASH 1234 MG ROAD BANGALORE", -5000, "cash", "atm"),
    ("NWD-416021XXXXXX1234-S1ANBL12-BANGALORE", -2000, "cash", "atm"),
    ("EAW-416021XXXXXX1234-S1AWBL01-BANGALORE", -3000, "cash", "atm"),
    ("CASH WITHDRAWAL SELF CHQ 000045", -10000, "cash", None),
    # ---- refunds and reversals -----------------------------------------------------------------
    ("UPI/CR/412345678927/AMAZON PAY/YESB/amazonpay@ybl/Refund", 1299, "shopping", "upi"),
    ("REV-UPI/DR/412345678928/SWIGGY/YESB/swiggy@ybl", 286, "food_delivery", None),
    ("UPI REVERSAL 412345678929 ZOMATO", 412, "food_delivery", None),
    ("REFUND FLIPKART ORDER OD12345", 2499, "shopping", None),
    ("IRCTC REFUND PNR 1234567890", 1180, "travel", None),
    # ---- groceries -----------------------------------------------------------------------------
    ("UPI-ZEPTO-ZEPTO.PAYU@AXISBANK-UTIB0000100-412345678930-PAYMENT FROM PHONE", -642, "groceries", "upi"),
    ("UPI/P2M/412345678931/BLINKIT/Payment/YES BANK", -388, "groceries", "upi"),
    ("POS/DMART AVENUE/BANGALORE", -2140, "groceries", "card"),
    ("UPI/DR/412345678932/SRI LAKSHMI PROVISIONS/SBIN/lakshmiprov@okaxis/Payment", -460, "groceries", "upi"),
    ("POS 416021XXXXXX1234 MORE SUPERMARKET", -980, "groceries", "card"),
    # ---- food ----------------------------------------------------------------------------------
    ("UPI/ZOMATO LTD/412345678933/UPI", -510, "food_delivery", "upi"),
    ("UPI-SWIGGY-SWIGGY8@YBL-YESB0YBLUPI-412345678934-PAYMENT FROM PHONE", -364, "food_delivery", "upi"),
    ("POS 416021XXXXXX1234 BARBEQUE NATION", -3250, "dining", "card"),
    ("PCD/1234/RAMESHWARAM CAFE/BANGALORE", -420, "dining", "card"),
    # ---- transport & travel --------------------------------------------------------------------
    ("UPI/DR/412345678935/IDFC FASTAG/IDFB/fastag@idfcbank/Recharge", -1000, "transport", "upi"),
    ("UPI/DR/412345678936/BMRCL/YESB/bmrcl@ybl/Metro card", -500, "transport", "upi"),
    ("POS 416021XXXXXX1234 HPCL PETROL PUMP", -2500, "transport", "card"),
    ("POS 416021XXXXXX1234 INDIAN OIL CORP", -1800, "transport", "card"),
    ("UPI/DR/412345678937/NAMMA YATRI/YESB/nammayatri@ybl/Ride", -186, "transport", "upi"),
    ("UPI/DR/412345678938/IRCTC/SBIN/irctc@sbi/Train", -1420, "travel", "upi"),
    ("ECOM PUR/MAKEMYTRIP/GURGAON", -8600, "travel", "card"),
    ("POS 416021XXXXXX1234 INDIGO AIRLINES", -5400, "travel", "card"),
    ("UPI/DR/412345678939/REDBUS/YESB/redbus@ybl/Bus", -1250, "travel", "upi"),
    ("ECOM PUR/AIRBNB/DUBLIN", -9800, "travel", "card"),
    # ---- shopping ------------------------------------------------------------------------------
    ("ECOM PUR/AMAZON/BANGALORE", -1899, "shopping", "card"),
    ("POS 416021XXXXXX1234 AMAZON PAY INDIA PRIVA", -749, "shopping", "card"),
    ("UPI/DR/412345678941/FLIPKART/YESB/flipkart@ybl/Order", -2299, "shopping", "upi"),
    ("POS 416021XXXXXX1234 RELIANCE DIGITAL", -15990, "shopping", "card"),
    ("POS 416021XXXXXX1234 IKEA BENGALURU", -6500, "shopping", "card"),
    ("POS 416021XXXXXX1234 TANISHQ JEWELLERS", -42000, "shopping", "card"),
    # ---- health, education, personal care, entertainment, subscriptions, giving --------------------
    ("UPI/DR/412345678942/PRACTO/YESB/practo@ybl/Consultation", -600, "health", "upi"),
    ("UPI/DR/412345678943/TATA 1MG/YESB/tata1mg@ybl/Medicines", -845, "health", "upi"),
    ("POS 416021XXXXXX1234 MANIPAL HOSPITAL", -4200, "health", "card"),
    ("UPI/DR/412345678944/DR LAL PATHLABS/YESB/lalpath@ybl/Tests", -1350, "health", "upi"),
    ("NEFT DR-SBIN0001234-DELHI PUBLIC SCHOOL-SCHOOL FEE TERM 2", -45000, "education", "neft"),
    ("UPI/DR/412345678945/UNACADEMY/YESB/unacademy@ybl/Subscription", -2400, "education", "upi"),
    ("ECOM PUR/COURSERA/MOUNTAIN VIEW", -3900, "education", "card"),
    ("UPI/DR/412345678946/URBAN COMPANY/YESB/urbanclap@ybl/Payment", -799, "personal_care", "upi"),
    ("POS 416021XXXXXX1234 NATURALS SALON", -650, "personal_care", "card"),
    ("UPI/DR/412345678947/BOOKMYSHOW/YESB/bookmyshow@ybl/Tickets", -620, "entertainment", "upi"),
    ("PCD/1234/PVR INOX/BANGALORE", -880, "entertainment", "card"),
    ("UPI/DR/412345678948/DREAM11/YESB/dream11@ybl/Add cash", -500, "entertainment", "upi"),
    ("CARD TXN AT NETFLIX.COM", -649, "subscriptions", "card"),
    ("SI-SPOTIFY INDIA 1234", -119, "subscriptions", None),
    ("ECOM PUR/GOOGLE *YOUTUBE PREMIUM", -129, "subscriptions", "card"),
    ("UPI/DR/412345678949/ISKCON TEMPLE/SBIN/iskcon@sbi/Donation", -1001, "gifts_donations", "upi"),
    ("UPI/DR/412345678950/GIVEINDIA/YESB/giveindia@ybl/Donation", -2000, "gifts_donations", "upi"),
]

# Recorded the first time this set ran (rules only, no LLM), before any fix it motivated.
FIRST_RUN = {
    "date": "2026-10-14",
    "category_accuracy": 0.85,
    "channel_accuracy": 0.9651,
    "confident_share": 0.808,
    "confident_precision": 0.9691,
    "misses": "no taxes category; bank-charge narrations; self-transfers; wallet loads; NPS/FD/RD/digital gold; "
    "EAW ATM code; BILLPAY to a masked card number",
}
