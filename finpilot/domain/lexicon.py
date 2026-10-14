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
    R(r"AMAZON ?FRESH|FLIPKART ?MINUTES|FLIPKART ?QUICK", "Quick commerce", "groceries"),
    R(
        r"RATNADEEP|SPAR ?HYPERMARKET|METRO ?CASH|VISHAL ?MEGA ?MART|SMART ?BAZAAR|EASYDAY",
        "Supermarket",
        "groceries",
    ),
    R(r"LICIOUS|FRESH ?TO ?HOME|FRESHTOHOME|ZAPPFRESH|MEATIGO", "Meat & fish", "groceries"),
    R(r"OTIPY|KPN ?FARM ?FRESH|MILKY ?MIST|AKSHAYAKALPA", "Fresh produce & dairy", "groceries"),
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
    R(
        r"ADYAR ?ANANDA ?BHAVAN|A2B|SARAVANA ?BHAVAN|MEGHANA ?FOODS|PARADISE ?BIRYANI|BEHROUZ ?BIRYANI|"
        r"BIKANERVALA|SAGAR ?RATNA|NANDHANA|EMPIRE ?RESTAURANT|RAMESHWARAM ?CAFE|BRAHMIN ?S ?COFFEE|"
        r"SOCIAL ?OFFLINE|HARD ?ROCK ?CAFE|BEER ?CAFE|ONESTA|LA ?PINO ?Z|LAPINOZ|NANDOS|NANDO ?S",
        "Restaurant",
        "dining",
    ),
    R(
        r"WOW ?MOMO|BURGER ?SINGH|TACO ?BELL|WENDY ?S|POPEYES|CHAI ?SUTTA ?BAR|CHAI ?KINGS",
        "Quick-service restaurant",
        "dining",
    ),
    R(
        r"BASKIN ?ROBBINS|NATURALS ?ICE ?CREAM|KRISPY ?KREME|DUNKIN|THEOBROMA|BARISTA|COSTA ?COFFEE|AMUL ?PARLOUR",
        "Dessert & coffee",
        "dining",
    ),
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
    R(r"PARK ?\+|PARKPLUS|PARK ?PLUS|CHALO ?APP|ZOOMCAR|REVV|BOUNCE ?INFINITY|VOGO", "Mobility", "transport"),
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
    R(r"NYKAA ?FASHION", "Nykaa Fashion", "shopping"),
    R(
        r"RELIANCE ?TRENDS|H ?M HENNES|HENNES|LEVI ?S|LEVIS|NIKE|ADIDAS|PUMA|SKECHERS|WOODLAND|MARKS ?SPENCER",
        "Fashion store",
        "shopping",
    ),
    R(r"SNAPDEAL|SHOPSY|NYKAA ?MAN|BEWAKOOF|SOULED ?STORE", "Online shopping", "shopping"),
    R(
        r"SAMSUNG|ONEPLUS|XIAOMI|MI ?STORE|SANGEETHA ?MOBILES|BAJAJ ?ELECTRONICS|GIRIAS|CROMA",
        "Electronics store",
        "shopping",
    ),
    R(r"WAKEFIT|SLEEPYCAT|DUROFLEX|SLEEPWELL|NILKAMAL", "Home furnishing", "shopping"),
    R(r"TANISHQ|MALABAR ?GOLD|CARATLANE|KALYAN ?JEWELLERS|JOYALUKKAS|GIVA", "Jewellery", "shopping"),
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
    R(
        r"UPPCL|PSPCL|TSSPDCL|APSPDCL|APEPDCL|DHBVN|UHBVN|JVVNL|AVVNL|WBSEDCL|TORRENT ?POWER|GESCOM|HESCOM|"
        r"MESCOM|CESCOM|MPEB|MPPKVVCL|CSPDCL|KESCO|NBPDCL|SBPDCL",
        "Electricity",
        "utilities",
        "electricity",
    ),
    R(r"HMWSSB|DELHI ?JAL|MCGM ?WATER", "Water", "utilities", "water"),
    R(r"GUJARAT ?GAS|TORRENT ?GAS|GAIL ?GAS|MNGL|ASSAM ?GAS", "Piped gas", "utilities", "gas"),
    R(r"INDANE|HP ?GAS|BHARAT ?GAS|MAHANAGAR ?GAS|IGL|ADANI ?GAS", "Cooking gas", "utilities", "gas"),
    R(r"COMCAST|XFINITY|VERIZON|T ?MOBILE|CON ?EDISON", "Utility", "utilities"),
    # ---- Housing ------------------------------------------------------------------------------
    R(r"NOBROKER ?HOOD|MYGATE|SOCIETY ?MAINT", "Society maintenance", "rent"),
    R(r"NESTAWAY|ZOLO ?STAYS|STANZA ?LIVING|COLIVE|ISTHARA|HELLO ?WORLD ?COLIVING", "Co-living rent", "rent"),
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
    R(
        r"MEDIBUDDY|HEALTHKART|CLOUDNINE|RAINBOW ?HOSPITAL|KAUVERY|MEDANTA|AIIMS|SANKARA ?NETHRALAYA",
        "Healthcare",
        "health",
    ),
    # ---- Education ----------------------------------------------------------------------------
    R(
        r"UDEMY|COURSERA|BYJU|UNACADEMY|UPGRAD|SCALER|SIMPLILEARN|PHYSICS ?WALLAH|SKILLSHARE|EDX|GREAT ?LEARNING",
        "Online course",
        "education",
    ),
    R(
        r"CROSSWORD ?BOOKSTORE|HIGGINBOTHAMS|BRITISH ?COUNCIL|VEDANTU|ALLEN ?CAREER|AAKASH ?INSTITUTE|FIITJEE|"
        r"CHEGG|MASTERCLASS|LEETCODE",
        "Education",
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
    R(r"KSRTC|MSRTC|TSRTC|APSRTC|GSRTC|UPSRTC|RSRTC|HRTC", "State transport bus", "travel"),
    R(r"THOMAS ?COOK|SOTC|AKBAR ?TRAVELS|VEENA ?WORLD", "Travel agency", "travel"),
    # ---- Entertainment ------------------------------------------------------------------------
    R(r"BOOKMYSHOW|BIGTREE", "BookMyShow", "entertainment"),
    R(r"PVR|INOX|CINEPOLIS", "Cinema", "entertainment"),
    R(r"PAYTM ?INSIDER|INSIDER ?IN|ZOMATO ?DISTRICT", "Events", "entertainment"),
    R(r"STEAM|PLAYSTATION|XBOX|NINTENDO|EPIC ?GAMES", "Gaming", "entertainment"),
    R(r"DREAM ?11|MPL|MY11CIRCLE|WINZO", "Fantasy gaming", "entertainment"),
    R(
        r"IMAGICA|ESSEL ?WORLD|SNOW ?WORLD|TIMEZONE|FUN ?CITY|KIDZANIA|SKYJUMPER",
        "Amusement",
        "entertainment",
    ),
    # ---- Personal care ------------------------------------------------------------------------
    R(r"URBAN ?COMPANY|URBANCLAP", "Urban Company", "personal_care"),
    R(r"LAKME ?SALON|NATURALS ?SALON|ENRICH|JAWED ?HABIB|TONI ?GUY|BODYCRAFT", "Salon", "personal_care"),
    R(r"NYKAA|PURPLLE|MAMAEARTH|SUGAR ?COSMETICS", "Beauty", "personal_care"),
    R(r"BBLUNT|YLG ?SALON|GEETANJALI ?SALON|VLCC|AFFINITY ?SALON|LOOKS ?SALON", "Salon", "personal_care"),
    # ---- Loans / EMIs -------------------------------------------------------------------------
    R(
        r"BAJAJ ?FIN(ANCE|SERV)?|HOME ?CREDIT|TATA ?CAPITAL|HDFC ?LTD ?LOAN|LOAN ?EMI|EMI ?(DEBIT|PAYMENT)|NACH ?EMI",
        "Loan EMI",
        "loans",
    ),
    R(
        r"KREDITBEE|MONEYVIEW|MONEY ?VIEW|NAVI ?FINSERV|FIBE|EARLYSALARY|LAZYPAY|SIMPL|ZESTMONEY|MPOKKET",
        "Digital lender",
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
        r"(?:ICICI ?PRU(?:DENTIAL)?|HDFC|SBI|AXIS|KOTAK|NIPPON|ADITYA ?BIRLA|TATA|DSP|UTI|MIRAE|PARAG ?PARIKH|PPFAS) ?(?:MF|MUTUAL ?FUND|AMC)",
        "Mutual fund",
        "investments",
    ),
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
        r"ET ?MONEY|DHAN ?(?:APP|HQ)|RAISE ?SECURITIES|5 ?PAISA|MOTILAL ?OSWAL|ICICI ?DIRECT|ICICIDIRECT|HDFC ?SEC|KOTAK ?SEC|SHAREKHAN|COIN ?ZERODHA",
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
        ("MILK PARLOUR", 0.95),
        ("MILK BOOTH", 0.9),
        ("MILK CENTRE", 0.85),
        ("FRUIT", 0.6),
        ("VEGETABLE", 0.8),
        ("SUPERMART", 0.85),
        ("KIRANA STORE", 0.95),
        ("PROVISION STORE", 0.9),
        ("DEPARTMENTAL STORE", 0.75),
        ("FLOUR MILL", 0.85),
        ("MEAT", 0.7),
        ("CHICKEN CENTRE", 0.85),
        ("MUTTON", 0.8),
        ("FISH MARKET", 0.85),
        ("EGGS", 0.6),
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
        ("ICE CREAM", 0.9),
        ("ICE CREAMS", 0.9),
        ("ICECREAM", 0.9),
        ("ICE CREAM PARLOUR", 0.95),
        ("CREAMERY", 0.85),
        ("DARSHINI", 0.85),
        ("FAST FOOD", 0.9),
        ("SNACKS", 0.7),
        ("TANDOOR", 0.8),
        ("BBQ", 0.8),
        ("BARBEQUE", 0.85),
        ("BARBECUE", 0.85),
        ("MOMO", 0.8),
        ("MOMOS", 0.85),
        ("SHAWARMA", 0.9),
        ("BHOJANALAYA", 0.9),
        ("TIFFINS", 0.85),
        ("IDLY", 0.8),
        ("PANI PURI", 0.85),
        ("FALOODA", 0.85),
        ("LASSI", 0.75),
        ("KULFI", 0.8),
        ("WAFFLE", 0.8),
        ("DESSERT", 0.75),
        ("DESSERTS", 0.8),
        ("PATISSERIE", 0.85),
        ("CAKES", 0.6),
        ("DONUTS", 0.8),
        ("TEA STALL", 0.85),
        ("JUICE PARLOUR", 0.9),
        ("JUICE CENTRE", 0.9),
        ("JUICE BAR", 0.9),
        ("REFRESHMENT", 0.75),
        ("REFRESHMENTS", 0.75),
        ("BREAKFAST", 0.75),
        ("MEALS", 0.7),
        ("THALI", 0.85),
        ("FOODS", 0.45),
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
        ("OPTICIANS", 0.8),
        ("EYE CARE", 0.85),
        ("EYE HOSPITAL", 0.95),
        ("AYURVEDA", 0.8),
        ("AYURVEDIC", 0.8),
        ("HOMEOPATHY", 0.85),
        ("VETERINARY", 0.85),
        ("PET CLINIC", 0.85),
        ("NURSING HOME", 0.95),
        ("POLYCLINIC", 0.95),
        ("MEDICINE", 0.75),
        ("MEDICINES", 0.8),
        ("HEALTHCARE", 0.75),
        ("PHYSIOTHERAPY", 0.9),
        ("SCAN CENTRE", 0.85),
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
        ("AUTO WORKS", 0.85),
        ("MOTOR WORKS", 0.85),
        ("GARAGE", 0.8),
        ("TYRE", 0.6),
        ("CAR WASH", 0.85),
        ("BIKE SERVICE", 0.85),
        ("CAR SERVICE", 0.7),
        ("CNG", 0.85),
        ("EV CHARGING", 0.85),
        ("CHARGING STATION", 0.8),
        ("AUTOMOBILES", 0.6),
        ("AUTOMOTIVE", 0.7),
        ("BUS PASS", 0.8),
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
        ("GAS AGENCY", 0.9),
        ("GAS AGENCIES", 0.9),
        ("CYLINDER", 0.8),
        ("CABLE TV", 0.85),
        ("WIFI", 0.75),
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
        ("BOOK HOUSE", 0.85),
        ("BOOK STORE", 0.85),
        ("BOOKSTORE", 0.85),
        ("BOOK CENTRE", 0.85),
        ("BOOK CENTER", 0.85),
        ("STATIONERS", 0.65),
        ("TUTORIAL", 0.8),
        ("TUTORIALS", 0.85),
        ("SCHOOL FEE", 0.95),
        ("SCHOOL FEES", 0.95),
        ("COLLEGE FEES", 0.95),
        ("TUITION FEE", 0.95),
        ("LIBRARY", 0.7),
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
        ("INTERIORS", 0.8),
        ("DECOR", 0.8),
        ("FURNISHING", 0.85),
        ("FURNISHINGS", 0.85),
        ("HANDLOOM", 0.85),
        ("HANDLOOMS", 0.85),
        ("HANDICRAFTS", 0.85),
        ("EMPORIUM", 0.7),
        ("COMPUTERS", 0.7),
        ("LAPTOP", 0.75),
        ("APPLIANCES", 0.85),
        ("CRAFTS", 0.55),
        ("KIDS WEAR", 0.85),
        ("MENS WEAR", 0.85),
        ("SAREE", 0.85),
        ("ETHNIC", 0.7),
        ("LEATHER", 0.6),
        ("CYCLES", 0.6),
        ("SPORTS", 0.55),
        ("ELECTRICALS", 0.7),
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
        ("BEAUTY PARLOUR", 0.95),
        ("BEAUTY PARLOR", 0.95),
        ("LADIES PARLOUR", 0.95),
        ("MENS SALON", 0.95),
        ("MAKEOVER", 0.8),
        ("NAILS", 0.7),
        ("TATTOO", 0.8),
        ("LAUNDRY", 0.65),
        ("DRY CLEAN", 0.7),
        ("DRY CLEANERS", 0.75),
        ("SALON", 0.95),
        ("SPA", 0.85),
        ("PARLOUR", 0.6),
        ("PARLOR", 0.6),
        ("BEAUTY", 0.75),
        ("GROOMING", 0.85),
        ("BARBER", 0.9),
        ("UNISEX", 0.8),
        ("COSMETICS", 0.8),
        ("HAIR", 0.6),
    ],
    "travel": [
        ("GUEST HOUSE", 0.85),
        ("HOMESTAYS", 0.85),
        ("TOURISM", 0.8),
        ("FLIGHT", 0.85),
        ("AIR TICKET", 0.9),
        ("TRAIN TICKET", 0.85),
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
        ("THEME PARK", 0.9),
        ("GO KARTING", 0.85),
        ("KARTING", 0.8),
        ("TRAMPOLINE", 0.85),
        ("ESCAPE ROOM", 0.85),
        ("PAINTBALL", 0.85),
        ("SNOW PARK", 0.85),
        ("GAME ZONE", 0.85),
        ("PLAY ZONE", 0.8),
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
        ("COLIVING", 0.85),
        ("CO LIVING", 0.85),
        ("LEASE RENTAL", 0.85),
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
        ("GIFT SHOP", 0.85),
        ("GIFT CENTRE", 0.85),
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
    """Noisy-OR combination of keyword evidence per category.

    A matched phrase hides the shorter keywords inside it: "MILK PARLOUR" is a dairy, not a
    beauty parlour, and "ICE CREAM PARLOUR" is not evidence for a salon.
    """
    hits: list[tuple[int, int, str, int, float]] = []
    for cat, pats in _KEYWORD_RX.items():
        for idx, (rx, w) in enumerate(pats):
            for m in rx.finditer(norm_text):
                hits.append((m.start(), m.end(), cat, idx, w))
    kept = [
        h for h in hits if not any(o[0] <= h[0] and h[1] <= o[1] and o[1] - o[0] > h[1] - h[0] for o in hits)
    ]
    p_not: dict[str, float] = {}
    seen: set[tuple[str, int]] = set()
    for _, _, cat, idx, w in kept:
        if (cat, idx) in seen:
            continue  # the same keyword twice is one piece of evidence
        seen.add((cat, idx))
        p_not[cat] = p_not.get(cat, 1.0) * (1.0 - w)
    return {cat: 1.0 - p for cat, p in p_not.items() if p < 1.0}


# Rules that cover several different brands under one family label. A transaction keeps the
# brand that actually matched ("Third Wave Coffee", "Croma"), not the label ("Café"), so people
# can ask about the merchant they know.
FAMILY_RULE_NAMES = frozenset(
    {
        "Café", "Restaurant", "Quick-service restaurant", "Dessert & coffee", "Supermarket", "Quick commerce",
        "Meat & fish", "Fresh produce & dairy", "Food delivery", "Ride-hailing", "Metro", "Fuel", "Mobility",
        "Electronics store", "Home furnishing", "Fashion store", "Online shopping", "Retail store", "Jewellery",
        "Broadband", "DTH TV", "Electricity", "Water", "Cooking gas", "Piped gas", "Utility", "Pharmacy",
        "Hospital/Clinic", "Diagnostics lab", "Healthcare", "Online course", "Education", "Airline",
        "Hotel booking", "Travel booking", "Bus booking", "State transport bus", "Travel agency", "Cinema",
        "Events", "Gaming", "Fantasy gaming", "Amusement", "Salon", "Beauty", "Gym membership",
        "Streaming service", "Loan EMI", "Digital lender", "Insurance premium", "Brokerage / MF", "Mutual fund",
        "Co-living rent", "Gift shop", "Donation", "Freelance income",
    }
)  # fmt: skip


def _norm(text: str) -> str:
    s = re.sub(r"[^A-Z0-9 ]+", " ", (text or "").upper().replace("'", ""))
    return re.sub(r"\s+", " ", s).strip()


_SHORT_WORDS = frozenset(
    [
        "OIL",
        "AIR",
        "PAY",
        "CAB",
        "BUS",
        "BAR",
        "PUB",
        "TEA",
        "ONE",
        "NEW",
        "BIG",
        "TOP",
        "BOX",
        "CAR",
        "GAS",
        "FUN",
        "INN",
        "DAY",
    ]
)


def _title(text: str) -> str:
    """Title-case a matched brand while keeping acronyms (KFC, PVR, BMRCL) upper-case."""
    return " ".join(
        w if (len(w) <= 3 and w not in _SHORT_WORDS) or not re.search(r"[AEIOU]", w) else w.capitalize()
        for w in text.split()
    )


def display_name(rule: MerchantRule, norm_text: str) -> str:
    """Merchant name for a rule match: the brand itself for family rules, the rule's name otherwise."""
    if rule.name not in FAMILY_RULE_NAMES:
        return rule.name
    m = rule.regex.search(norm_text)
    matched = m.group(0).strip() if m else ""
    if not matched or set(matched.split()) & set(_norm(rule.name).split()):
        return rule.name
    return _title(matched)


def rule_by_name(name: str) -> MerchantRule | None:
    for rule in MERCHANT_RULES:
        if rule.name == name:
            return rule
    return match_merchant(_norm(name)) if name else None  # brand names from family rules
