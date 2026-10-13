"""Merchant knowledge base and generic keyword signals used by the categorizer.

Rules are matched against the *normalised* description (upper-case, punctuation → spaces).
Order matters: more specific rules (e.g. "SWIGGY ONE", "SWIGGY INSTAMART") precede the
generic brand rule ("SWIGGY").

`group` identifies a service family (used to detect overlapping subscriptions) and `parent`
links a paid membership to the merchant whose activity proves the membership is being used
(e.g. Swiggy One is worthless if there were no Swiggy orders in the last 45 days).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MerchantRule:
    pattern: str
    name: str
    category: str
    group: str | None = None
    parent: str | None = None

    @property
    def regex(self) -> re.Pattern[str]:
        return _compiled(self.pattern)


_CACHE: dict[str, re.Pattern[str]] = {}


def _compiled(pattern: str) -> re.Pattern[str]:
    rx = _CACHE.get(pattern)
    if rx is None:
        rx = re.compile(rf"(?<![A-Z0-9])(?:{pattern})(?![A-Z0-9])")
        _CACHE[pattern] = rx
    return rx


R = MerchantRule

MERCHANT_RULES: list[MerchantRule] = [
    # ---- Memberships (before their parent brands) -------------------------------------------
    R(r"SWIGGY ?ONE", "Swiggy One", "subscriptions", "delivery_membership", "Swiggy"),
    R(r"ZOMATO ?(GOLD|PRO)", "Zomato Gold", "subscriptions", "delivery_membership", "Zomato"),
    R(r"UBER ?ONE", "Uber One", "subscriptions", "ride_membership", "Uber"),
    R(
        r"AMAZON ?PRIME|PRIME ?MEMBERSHIP|AMZN ?PRIME",
        "Amazon Prime",
        "subscriptions",
        "shopping_membership",
        "Amazon",
    ),
    R(r"PRIME ?VIDEO|PRIMEVIDEO", "Prime Video", "subscriptions", "video_streaming"),
    R(r"FLIPKART ?(PLUS|VIP|BLACK)", "Flipkart VIP", "subscriptions", "shopping_membership", "Flipkart"),
    R(r"TIMES ?PRIME", "Times Prime", "subscriptions", "lifestyle_membership"),
    # ---- Streaming / software subscriptions ---------------------------------------------------
    R(r"NETFLIX", "Netflix", "subscriptions", "video_streaming"),
    R(r"HOTSTAR|DISNEY ?PLUS|DISNEYPLUS|JIO ?HOTSTAR", "Disney+ Hotstar", "subscriptions", "video_streaming"),
    R(r"SONY ?LIV", "SonyLIV", "subscriptions", "video_streaming"),
    R(r"ZEE ?5", "ZEE5", "subscriptions", "video_streaming"),
    R(r"JIO ?CINEMA", "JioCinema", "subscriptions", "video_streaming"),
    R(
        r"YOUTUBE ?PREMIUM|GOOGLE YOUTUBE|YOUTUBEPREMIUM",
        "YouTube Premium",
        "subscriptions",
        "video_streaming",
    ),
    R(r"HBO ?MAX|HULU|PARAMOUNT ?PLUS|PEACOCK", "Streaming service", "subscriptions", "video_streaming"),
    R(r"SPOTIFY", "Spotify", "subscriptions", "music_streaming"),
    R(r"APPLE ?MUSIC", "Apple Music", "subscriptions", "music_streaming"),
    R(r"GAANA", "Gaana", "subscriptions", "music_streaming"),
    R(r"WYNK", "Wynk Music", "subscriptions", "music_streaming"),
    R(r"JIO ?SAAVN", "JioSaavn", "subscriptions", "music_streaming"),
    R(r"AUDIBLE", "Audible", "subscriptions", "audiobooks"),
    R(r"KINDLE ?UNLIMITED", "Kindle Unlimited", "subscriptions", "ebooks"),
    R(r"ICLOUD|APPLE COM BILL|APPLE ?ONE", "Apple iCloud", "subscriptions", "cloud_storage"),
    R(r"GOOGLE ?ONE|GOOGLE ?STORAGE", "Google One", "subscriptions", "cloud_storage"),
    R(r"DROPBOX", "Dropbox", "subscriptions", "cloud_storage"),
    R(r"MICROSOFT ?365|MSFT|OFFICE ?365", "Microsoft 365", "subscriptions", "productivity"),
    R(r"ADOBE", "Adobe", "subscriptions", "creative_software"),
    R(r"CANVA", "Canva", "subscriptions", "creative_software"),
    R(r"NOTION", "Notion", "subscriptions", "productivity"),
    R(r"OPENAI|CHATGPT", "ChatGPT Plus", "subscriptions", "ai_assistant"),
    R(r"ANTHROPIC|CLAUDE AI", "Claude Pro", "subscriptions", "ai_assistant"),
    R(r"LINKEDIN", "LinkedIn Premium", "subscriptions", "professional"),
    R(r"GITHUB", "GitHub", "subscriptions", "developer_tools"),
    R(r"DUOLINGO", "Duolingo", "subscriptions", "learning_app"),
    # ---- Fitness memberships ------------------------------------------------------------------
    R(r"CULT ?FIT|CULTFIT|CURE ?FIT", "Cult.fit", "health", "fitness_membership"),
    R(
        r"GOLD ?S GYM|ANYTIME ?FITNESS|PLANET ?FITNESS|SNAP ?FITNESS",
        "Gym membership",
        "health",
        "fitness_membership",
    ),
    # ---- Groceries (quick-commerce before delivery brands) ------------------------------------
    R(r"SWIGGY ?INSTAMART|INSTAMART", "Swiggy Instamart", "groceries"),
    R(r"BLINKIT|GROFERS", "Blinkit", "groceries"),
    R(r"ZEPTO", "Zepto", "groceries"),
    R(r"BIG ?BASKET|BBNOW|BB ?DAILY", "BigBasket", "groceries"),
    R(r"D ?MART|AVENUE SUPERMARTS", "DMart", "groceries"),
    R(r"JIO ?MART", "JioMart", "groceries"),
    R(r"RELIANCE (FRESH|SMART|RETAIL)", "Reliance Fresh", "groceries"),
    R(r"MORE (RETAIL|SUPERMARKET|MEGASTORE)", "More Retail", "groceries"),
    R(r"NATURE ?S BASKET", "Nature's Basket", "groceries"),
    R(r"SPENCER", "Spencer's", "groceries"),
    R(r"STAR BAZAAR", "Star Bazaar", "groceries"),
    R(r"DUNZO", "Dunzo", "groceries"),
    R(r"COUNTRY ?DELIGHT|MILK ?BASKET", "Country Delight", "groceries"),
    R(
        r"WHOLE ?FOODS|TRADER ?JOE|KROGER|SAFEWAY|ALDI|LIDL|TESCO|SAINSBURY|COSTCO",
        "Supermarket",
        "groceries",
    ),
    # ---- Food delivery ------------------------------------------------------------------------
    R(r"UBER ?EATS", "Uber Eats", "food_delivery"),
    R(r"SWIGGY|BUNDL TECHNOLOGIES", "Swiggy", "food_delivery"),
    R(r"ZOMATO|ETERNAL LTD", "Zomato", "food_delivery"),
    R(r"EAT ?SURE|FAASOS|BEHROUZ|REBEL FOODS|BOX8", "EatSure", "food_delivery"),
    R(r"DOMINO ?S|DOMINOS|JUBILANT FOODWORKS", "Domino's", "food_delivery"),
    R(r"DOORDASH|GRUBHUB|DELIVEROO|JUST ?EAT", "Food delivery", "food_delivery"),
    # ---- Dining -------------------------------------------------------------------------------
    R(r"STARBUCKS|TATA STARBUCKS", "Starbucks", "dining"),
    R(r"MCDONALD ?S|MC ?DONALDS|HARDCASTLE", "McDonald's", "dining"),
    R(r"KFC", "KFC", "dining"),
    R(r"BURGER ?KING", "Burger King", "dining"),
    R(r"PIZZA ?HUT", "Pizza Hut", "dining"),
    R(r"SUBWAY", "Subway", "dining"),
    R(r"CAFE ?COFFEE ?DAY|CCD", "Café Coffee Day", "dining"),
    R(r"CHAAYOS|CHAI ?POINT|THIRD ?WAVE ?COFFEE|BLUE ?TOKAI", "Café", "dining"),
    R(r"HALDIRAM", "Haldiram's", "dining"),
    R(r"BARBEQUE ?NATION|TOIT ?BREW|MAINLAND ?CHINA", "Restaurant", "dining"),
    # ---- Transport ----------------------------------------------------------------------------
    R(r"UBER|UBER ?INDIA|UBER ?TRIP", "Uber", "transport"),
    R(r"OLA ?CABS|ANI TECHNOLOGIES|OLACABS|OLA", "Ola", "transport"),
    R(r"RAPIDO|ROPPEN", "Rapido", "transport"),
    R(r"BLU ?SMART|NAMMA ?YATRI|YULU|LYFT", "Ride-hailing", "transport"),
    R(r"DMRC|BMRCL|METRO ?RAIL|MUMBAI ?METRO|NAMMA ?METRO", "Metro", "transport"),
    R(
        r"INDIAN ?OIL|IOCL|BHARAT ?PETROLEUM|BPCL|HPCL|HINDUSTAN ?PETROLEUM|SHELL|NAYARA|JIO ?BP",
        "Fuel",
        "transport",
    ),
    R(r"FASTAG|NETC|TOLL ?PLAZA", "FASTag toll", "transport"),
    # ---- Shopping -----------------------------------------------------------------------------
    R(r"AMAZON|AMZN|AMAZON ?PAY ?INDIA", "Amazon", "shopping"),
    R(r"FLIPKART", "Flipkart", "shopping"),
    R(r"MYNTRA", "Myntra", "shopping"),
    R(r"AJIO", "AJIO", "shopping"),
    R(r"MEESHO", "Meesho", "shopping"),
    R(r"TATA ?CLIQ", "Tata CLiQ", "shopping"),
    R(r"CROMA|RELIANCE ?DIGITAL|VIJAY ?SALES", "Electronics store", "shopping"),
    R(r"DECATHLON", "Decathlon", "shopping"),
    R(r"IKEA|PEPPERFRY|URBAN ?LADDER|HOME ?CENTRE", "Home furnishing", "shopping"),
    R(r"LIFESTYLE|WESTSIDE|PANTALOONS|SHOPPERS ?STOP|MAX ?FASHION|ZARA|UNIQLO", "Fashion store", "shopping"),
    R(r"LENSKART", "Lenskart", "shopping"),
    R(r"FIRSTCRY", "FirstCry", "shopping"),
    R(r"APPLE ?STORE|BEST ?BUY|TARGET|WALMART|EBAY|ETSY|SHEIN", "Retail store", "shopping"),
    # ---- Utilities ----------------------------------------------------------------------------
    R(r"AIRTEL|BHARTI", "Airtel", "utilities", "telecom"),
    R(
        r"RELIANCE ?JIO|JIO ?PREPAID|JIO ?POSTPAID|JIO ?RECHARGE|JIO ?FIBER|JIOFIBER",
        "Jio",
        "utilities",
        "telecom",
    ),
    R(r"VODAFONE|VI ?POSTPAID|VI ?PREPAID|VODAFONE ?IDEA", "Vi", "utilities", "telecom"),
    R(r"BSNL", "BSNL", "utilities", "telecom"),
    R(r"ACT ?FIBERNET|HATHWAY|EXCITEL|YOU ?BROADBAND", "Broadband", "utilities", "internet"),
    R(r"TATA ?PLAY|TATA ?SKY|DISH ?TV|SUN ?DIRECT", "DTH TV", "utilities", "tv"),
    R(
        r"BESCOM|TATA ?POWER|ADANI ?ELECTRICITY|MSEDCL|BSES|TANGEDCO|TNEB|CESC|KSEB|ELECTRICITY ?BOARD",
        "Electricity",
        "utilities",
        "electricity",
    ),
    R(r"BWSSB|WATER ?BOARD|JAL ?BOARD", "Water", "utilities", "water"),
    R(r"INDANE|HP ?GAS|BHARAT ?GAS|MAHANAGAR ?GAS|IGL|ADANI ?GAS", "Cooking gas", "utilities", "gas"),
    R(r"COMCAST|XFINITY|VERIZON|T ?MOBILE|CON ?EDISON", "Utility", "utilities"),
    # ---- Housing ------------------------------------------------------------------------------
    R(r"NOBROKER ?HOOD|MYGATE|SOCIETY ?MAINT", "Society maintenance", "rent"),
    R(r"NO ?BROKER|HOUSE ?RENT|RENT ?PAYMENT|MONTHLY ?RENT|RENT", "Rent", "rent"),
    # ---- Health -------------------------------------------------------------------------------
    R(r"APOLLO ?PHARMACY|APOLLO ?247", "Apollo Pharmacy", "health"),
    R(r"MEDPLUS|PHARMEASY|NETMEDS|TATA ?1 ?MG|1MG|TRUEMEDS", "Pharmacy", "health"),
    R(
        r"PRACTO|APOLLO ?HOSPITAL|FORTIS|MANIPAL ?HOSP|MAX ?HEALTHCARE|NARAYANA ?HEALTH|ASTER",
        "Hospital/Clinic",
        "health",
    ),
    R(r"THYROCARE|LAL ?PATH|METROPOLIS|REDCLIFFE|HEALTHIANS", "Diagnostics lab", "health"),
    R(r"CVS|WALGREENS", "Pharmacy", "health"),
    # ---- Education ----------------------------------------------------------------------------
    R(
        r"UDEMY|COURSERA|BYJU|UNACADEMY|UPGRAD|SCALER|SIMPLILEARN|PHYSICS ?WALLAH|SKILLSHARE|EDX|GREAT ?LEARNING",
        "Online course",
        "education",
    ),
    # ---- Travel -------------------------------------------------------------------------------
    # "MMT/IMPS" is ICICI's Mobile Money Transfer, not MakeMyTrip
    R(r"MAKEMYTRIP|MAKE ?MY ?TRIP|MMT(?! ?IMPS)", "MakeMyTrip", "travel"),
    R(r"GOIBIBO|CLEARTRIP|YATRA|IXIGO|EASEMYTRIP", "Travel booking", "travel"),
    R(r"IRCTC|INDIAN ?RAIL(WAYS?)?|INDIAN ?RAI", "IRCTC", "travel"),
    R(
        r"INDIGO|INTERGLOBE|AIR ?INDIA|VISTARA|SPICEJET|AKASA|EMIRATES|LUFTHANSA|UNITED ?AIRLINES",
        "Airline",
        "travel",
    ),
    R(
        r"OYO|AIRBNB|BOOKING ?COM|EXPEDIA|AGODA|TAJ ?HOTELS|MARRIOTT|HYATT|HILTON|TREEBO|FABHOTELS",
        "Hotel booking",
        "travel",
    ),
    R(r"REDBUS|ABHIBUS", "Bus booking", "travel"),
    # ---- Entertainment ------------------------------------------------------------------------
    R(r"BOOKMYSHOW|BIGTREE", "BookMyShow", "entertainment"),
    R(r"PVR|INOX|CINEPOLIS", "Cinema", "entertainment"),
    R(r"PAYTM ?INSIDER|INSIDER ?IN|ZOMATO ?DISTRICT", "Events", "entertainment"),
    R(r"STEAM|PLAYSTATION|XBOX|NINTENDO|EPIC ?GAMES", "Gaming", "entertainment"),
    R(r"DREAM ?11|MPL|MY11CIRCLE|WINZO", "Fantasy gaming", "entertainment"),
    # ---- Personal care ------------------------------------------------------------------------
    R(r"URBAN ?COMPANY|URBANCLAP", "Urban Company", "personal_care"),
    R(r"LAKME ?SALON|NATURALS ?SALON|ENRICH|JAWED ?HABIB|TONI ?GUY|BODYCRAFT", "Salon", "personal_care"),
    R(r"NYKAA|PURPLLE|MAMAEARTH|SUGAR ?COSMETICS", "Beauty", "personal_care"),
    # ---- Loans / EMIs -------------------------------------------------------------------------
    R(
        r"BAJAJ ?FIN(ANCE|SERV)?|HOME ?CREDIT|TATA ?CAPITAL|HDFC ?LTD ?LOAN|LOAN ?EMI|EMI ?(DEBIT|PAYMENT)|NACH ?EMI",
        "Loan EMI",
        "loans",
    ),
    # ---- Bank-to-bank transfers (family, own accounts) ----------------------------------------
    R(
        r"MMT ?IMPS|TRANSFER ?TO ?FAM|SELF ?TRANSFER|OWN ?ACCOUNT|FUND ?TRANSFER",
        "Bank transfer",
        "transfers",
    ),
    # ---- Explicit credit-card repayments; an app name alone does not prove a bill payment ----
    R(
        r"CREDIT ?CARD ?(BILL|PAYMENT)|CC ?BILL ?PAY|CARD ?BILL ?PAYMENT",
        "Credit card payment",
        "credit_card_payment",
    ),
    # ---- Insurance ----------------------------------------------------------------------------
    R(
        r"LIC ?OF ?INDIA|LIFE ?INSURANCE ?CORP|HDFC ?LIFE|ICICI ?PRU|SBI ?LIFE|MAX ?LIFE|TATA ?AIA|STAR ?HEALTH|NIVA ?BUPA|"
        r"CARE ?HEALTH|ACKO|GO ?DIGIT|POLICYBAZAAR|TATA ?AIG|BAJAJ ?ALLIANZ|LIC",
        "Insurance premium",
        "insurance",
    ),
    # ---- Investments --------------------------------------------------------------------------
    R(
        r"ZERODHA|GROWW|UPSTOX|KUVERA|ANGEL ?ONE|SMALLCASE|PAYTM ?MONEY|INDMONEY|ROBINHOOD|VANGUARD|FIDELITY",
        "Brokerage / MF",
        "investments",
    ),
    R(
        # ICCL = Indian Clearing Corporation, which collects mutual-fund SIP autopays
        r"BSE ?STAR ?MF|MUTUAL ?FUND|SIP ?(DEBIT|INSTALLMENT)?|NPS ?TRUST|PPF ?DEPOSIT|RECURRING ?DEPOSIT|"
        r"ICCL|INDIAN ?CLEARING|MF ?AUTOPAY|CAMS|KFINTECH|KARVY",
        "SIP / MF",
        "investments",
    ),
    # ---- Cash ---------------------------------------------------------------------------------
    R(r"ATM ?(WDL|WITHDRAWAL|CASH)?|CASH ?(WDL|WITHDRAWAL)|NWD", "ATM withdrawal", "cash"),
    # ---- Fees ---------------------------------------------------------------------------------
    R(
        r"LATE ?(PAYMENT ?)?FEE|LATE ?PAYMENT ?CHARGE|PENALTY|MIN ?BAL(ANCE)? ?CHARGE|NON ?MAINT(ENANCE)? ?CHARGE|"
        r"OVERDRAFT ?FEE|BOUNCE ?CHARGE|ANNUAL ?FEE|SMS ?CHARGES|DEBIT ?CARD ?FEE|FINANCE ?CHARGE",
        "Bank fee",
        "fees",
    ),
    # ---- Income -------------------------------------------------------------------------------
    R(r"SALARY|SAL ?CR|PAYROLL", "Salary", "income"),
    R(r"UPWORK|FIVERR|TOPTAL|DEEL", "Freelance income", "income"),
    R(r"INT(EREST)? ?(PD|PAID|CREDIT|CR)|SB ?INTEREST|DIVIDEND", "Interest / dividend", "income"),
    R(r"CASHBACK|REFUND|REVERSAL", "Refund / cashback", "income"),
    # ---- Gifts & donations --------------------------------------------------------------------
    R(r"FERNS ?N ?PETALS|FNP|IGP ?COM|ARCHIES", "Gift shop", "gifts_donations"),
    R(
        r"GIVEINDIA|GIVE ?INDIA|KETTO|MILAAP|UNICEF|CRY ?FOUNDATION|AKSHAYA ?PATRA|DONATION|CHARITY",
        "Donation",
        "gifts_donations",
    ),
]

# Generic keyword evidence for merchants we have never seen. Weight ~ how diagnostic the token is.
KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "groceries": [
        ("SUPERMARKET", 0.9),
        ("HYPERMARKET", 0.9),
        ("KIRANA", 0.9),
        ("GROCERY", 0.9),
        ("GROCERIES", 0.9),
        ("PROVISION", 0.8),
        ("PROVISIONS", 0.8),
        ("GENERAL STORE", 0.7),
        ("MART", 0.6),
        ("FRESH", 0.5),
        ("VEGETABLES", 0.8),
        ("FRUITS", 0.7),
        ("DAIRY", 0.8),
        ("MILK", 0.6),
        ("SUPER MARKET", 0.9),
    ],
    "dining": [
        ("RESTAURANT", 0.9),
        ("RESTRO", 0.8),
        ("CAFE", 0.85),
        ("COFFEE", 0.75),
        ("BISTRO", 0.85),
        ("DHABA", 0.9),
        ("BAR", 0.6),
        ("PUB", 0.8),
        ("BREW", 0.7),
        ("BREWERY", 0.85),
        ("BAKERY", 0.8),
        ("BAKERS", 0.75),
        ("SWEETS", 0.7),
        ("BIRYANI", 0.85),
        ("PIZZA", 0.7),
        ("BURGER", 0.7),
        ("KITCHEN", 0.6),
        ("EATERY", 0.85),
        ("CANTEEN", 0.8),
        ("TIFFIN", 0.8),
        ("BHAVAN", 0.6),
        ("BHAWAN", 0.6),
        ("UDUPI", 0.8),
        ("CHAAT", 0.8),
        ("JUICE", 0.6),
        ("FOOD COURT", 0.8),
        ("GRILL", 0.7),
        ("DINER", 0.85),
        ("TEA", 0.5),
        ("CHAI", 0.6),
        ("IDLI", 0.8),
        ("DOSA", 0.8),
        ("MESS", 0.6),
    ],
    "health": [
        ("PHARMACY", 0.95),
        ("PHARMA", 0.8),
        ("MEDICAL", 0.85),
        ("MEDICALS", 0.9),
        ("CHEMIST", 0.9),
        ("DRUG", 0.8),
        ("CLINIC", 0.9),
        ("HOSPITAL", 0.95),
        ("DIAGNOSTIC", 0.9),
        ("DIAGNOSTICS", 0.9),
        ("PATHOLOGY", 0.9),
        ("LABS", 0.5),
        ("DENTAL", 0.9),
        ("DENTIST", 0.9),
        ("HEALTH", 0.5),
        ("GYM", 0.8),
        ("FITNESS", 0.75),
        ("YOGA", 0.7),
        ("PHYSIO", 0.85),
        ("OPTICAL", 0.6),
        ("NURSING", 0.8),
    ],
    "transport": [
        ("PETROL", 0.95),
        ("FUEL", 0.9),
        ("FUELS", 0.9),
        ("FILLING STATION", 0.95),
        ("SERVICE STATION", 0.7),
        ("PETROLEUM", 0.9),
        ("CAB", 0.7),
        ("CABS", 0.75),
        ("TAXI", 0.85),
        ("PARKING", 0.9),
        ("TOLL", 0.9),
        ("METRO", 0.7),
        ("AUTO RICKSHAW", 0.8),
        ("TRANSPORT", 0.6),
        ("MOTORS", 0.4),
        ("TYRES", 0.6),
    ],
    "utilities": [
        ("ELECTRICITY", 0.95),
        ("POWER", 0.6),
        ("ENERGY", 0.5),
        ("WATER SUPPLY", 0.85),
        ("GAS", 0.6),
        ("BROADBAND", 0.95),
        ("FIBER", 0.7),
        ("FIBRE", 0.7),
        ("INTERNET", 0.8),
        ("TELECOM", 0.85),
        ("RECHARGE", 0.75),
        ("POSTPAID", 0.9),
        ("PREPAID", 0.7),
        ("DTH", 0.9),
        ("BILL PAYMENT", 0.4),
        ("BILLDESK", 0.4),
    ],
    "education": [
        ("SCHOOL", 0.9),
        ("COLLEGE", 0.85),
        ("UNIVERSITY", 0.9),
        ("ACADEMY", 0.7),
        ("TUITION", 0.95),
        ("COACHING", 0.9),
        ("INSTITUTE", 0.75),
        ("CLASSES", 0.7),
        ("EDUCATION", 0.85),
        ("LEARNING", 0.6),
        ("BOOK DEPOT", 0.7),
        ("BOOKS", 0.5),
        ("STATIONERY", 0.6),
        ("EXAM FEE", 0.85),
    ],
    "shopping": [
        ("FASHION", 0.8),
        ("APPAREL", 0.85),
        ("CLOTHING", 0.85),
        ("GARMENTS", 0.85),
        ("FOOTWEAR", 0.85),
        ("SHOES", 0.75),
        ("ELECTRONICS", 0.85),
        ("MOBILES", 0.7),
        ("JEWELLERS", 0.85),
        ("JEWELLERY", 0.85),
        ("TEXTILES", 0.8),
        ("BOUTIQUE", 0.85),
        ("COLLECTION", 0.5),
        ("STORE", 0.35),
        ("STORES", 0.35),
        ("MALL", 0.6),
        ("RETAIL", 0.5),
        ("GIFTS", 0.3),
        ("TOYS", 0.75),
        ("HARDWARE", 0.7),
        ("FURNITURE", 0.8),
        ("OPTICALS", 0.5),
        ("WATCH", 0.5),
        ("SAREES", 0.85),
        ("SILKS", 0.7),
    ],
    "personal_care": [
        ("SALON", 0.95),
        ("SPA", 0.85),
        ("PARLOUR", 0.9),
        ("PARLOR", 0.9),
        ("BEAUTY", 0.75),
        ("GROOMING", 0.85),
        ("BARBER", 0.9),
        ("UNISEX", 0.8),
        ("COSMETICS", 0.8),
        ("HAIR", 0.6),
    ],
    "travel": [
        ("RESORT", 0.85),
        ("RESORTS", 0.85),
        ("INN", 0.5),
        ("LODGE", 0.75),
        ("AIRLINES", 0.95),
        ("AIRWAYS", 0.95),
        ("AIRPORT", 0.6),
        ("TRAVELS", 0.8),
        ("TOURS", 0.8),
        ("HOLIDAYS", 0.75),
        ("HOMESTAY", 0.85),
        ("RAILWAY", 0.75),
        ("HOSTELS", 0.6),
    ],
    "entertainment": [
        ("CINEMA", 0.9),
        ("CINEMAS", 0.9),
        ("MOVIES", 0.8),
        ("MULTIPLEX", 0.9),
        ("GAMING", 0.8),
        ("GAMES", 0.6),
        ("EVENTS", 0.6),
        ("CONCERT", 0.85),
        ("AMUSEMENT", 0.85),
        ("BOWLING", 0.85),
        ("THEATRE", 0.8),
        ("WATER PARK", 0.85),
        ("ARCADE", 0.8),
        ("ENTERTAINMENT", 0.7),
        ("FUN", 0.4),
    ],
    "rent": [
        ("RENT", 0.9),
        ("LANDLORD", 0.95),
        ("PG ACCOMMODATION", 0.9),
        ("PAYING GUEST", 0.9),
        ("MAINTENANCE", 0.6),
        ("SOCIETY", 0.6),
        ("APARTMENT", 0.6),
        ("APARTMENTS", 0.6),
        ("HOUSING", 0.5),
    ],
    "loans": [("EMI", 0.9), ("LOAN", 0.85), ("FINANCE", 0.45), ("LENDING", 0.7), ("CREDIT", 0.2)],
    "insurance": [("INSURANCE", 0.95), ("ASSURANCE", 0.9), ("POLICY", 0.6), ("LIFE", 0.3)],
    "investments": [
        ("MUTUAL", 0.85),
        ("SIP", 0.8),
        ("INVEST", 0.75),
        ("INVESTMENT", 0.8),
        ("SECURITIES", 0.75),
        ("BROKING", 0.85),
        ("STOCK", 0.6),
        ("FIXED DEPOSIT", 0.9),
        ("WEALTH", 0.6),
    ],
    "fees": [
        ("CHARGES", 0.6),
        ("CHARGE", 0.5),
        ("FEE", 0.45),
        ("PENALTY", 0.95),
        ("FINE", 0.6),
        ("GST ON", 0.6),
    ],
    "gifts_donations": [
        ("GIFT", 0.6),
        ("FLOWERS", 0.7),
        ("FLORIST", 0.85),
        ("DONATION", 0.95),
        ("CHARITY", 0.95),
        ("FOUNDATION", 0.6),
        ("TRUST", 0.45),
        ("TEMPLE", 0.8),
        ("CHURCH", 0.8),
        ("MOSQUE", 0.8),
        ("GURUDWARA", 0.8),
        ("NGO", 0.8),
    ],
    "subscriptions": [("SUBSCRIPTION", 0.9), ("MEMBERSHIP", 0.8), ("PREMIUM", 0.5), ("MONTHLY PLAN", 0.7)],
    "income": [
        ("SALARY", 0.95),
        ("PAYROLL", 0.95),
        ("STIPEND", 0.9),
        ("BONUS", 0.8),
        ("DIVIDEND", 0.9),
        ("INTEREST", 0.6),
        ("REFUND", 0.8),
        ("CASHBACK", 0.85),
        ("REIMBURSEMENT", 0.9),
        ("PAYOUT", 0.6),
    ],
    "transfers": [
        ("SELF TRANSFER", 0.95),
        ("OWN ACCOUNT", 0.9),
        ("TO SELF", 0.9),
        ("FUND TRANSFER", 0.7),
        ("TRANSFER TO", 0.6),
        ("SENT TO", 0.5),
    ],
}

_KEYWORD_RX: dict[str, list[tuple[re.Pattern[str], float]]] = {
    cat: [(re.compile(rf"(?<![A-Z]){re.escape(tok)}(?![A-Z])"), w) for tok, w in toks]
    for cat, toks in KEYWORDS.items()
}


def match_merchant(norm_text: str) -> MerchantRule | None:
    for rule in MERCHANT_RULES:
        if rule.regex.search(norm_text):
            return rule
    return None


def keyword_scores(norm_text: str) -> dict[str, float]:
    """Noisy-OR combination of keyword evidence per category."""
    scores: dict[str, float] = {}
    for cat, pats in _KEYWORD_RX.items():
        p_not = 1.0
        for rx, w in pats:
            if rx.search(norm_text):
                p_not *= 1.0 - w
        if p_not < 1.0:
            scores[cat] = 1.0 - p_not
    return scores


def rule_by_name(name: str) -> MerchantRule | None:
    for rule in MERCHANT_RULES:
        if rule.name == name:
            return rule
    return None
