"""Merchant enrichment: what a known merchant *is*, in plain words.

The lexicon says "Starbucks → Dining Out"; enrichment adds "coffee chain, café". Retrieval uses
these descriptors so a question about "coffee" finds Starbucks and Café Coffee Day even though the
word "coffee" never appears in a narration, and "medicines" finds Apollo and 1mg. Keys are lexicon
rule names (see `finpilot.domain.lexicon.MERCHANT_RULES`).
"""

from __future__ import annotations

from finpilot.domain.lexicon import match_merchant
from finpilot.domain.normalize import normalize_text

BRAND_TAGS: dict[str, str] = {
    # food & groceries
    "Swiggy": "food delivery app order meals restaurant takeaway",
    "Zomato": "food delivery app order meals restaurant takeaway",
    "Uber Eats": "food delivery app order meals",
    "EatSure": "food delivery app meals",
    "Domino's": "pizza delivery fast food",
    "Swiggy Instamart": "quick commerce groceries vegetables milk",
    "Blinkit": "quick commerce groceries vegetables milk",
    "Zepto": "quick commerce groceries vegetables milk",
    "BigBasket": "online groceries vegetables fruits",
    "DMart": "supermarket groceries household",
    "JioMart": "online groceries supermarket",
    "Reliance Fresh": "supermarket groceries vegetables",
    "Country Delight": "milk dairy delivery",
    "Meat & fish": "meat chicken fish butcher",
    "Fresh produce & dairy": "milk dairy vegetables fruits",
    "Starbucks": "coffee cafe chain beverages",
    "Café Coffee Day": "coffee cafe chain",
    "Café": "coffee tea cafe",
    "Dessert & coffee": "dessert ice cream coffee bakery cakes",
    "McDonald's": "burger fast food",
    "KFC": "fried chicken fast food",
    "Burger King": "burger fast food",
    "Pizza Hut": "pizza restaurant fast food",
    "Subway": "sandwich fast food",
    "Haldiram's": "snacks sweets restaurant",
    "Restaurant": "restaurant meal dinner lunch eating out",
    "Quick-service restaurant": "fast food restaurant snacks",
    # transport & travel
    "Uber": "cab taxi ride commute",
    "Ola": "cab taxi auto ride commute",
    "Rapido": "bike taxi auto ride commute",
    "Ride-hailing": "cab taxi ride commute",
    "Metro": "metro train commute card recharge",
    "Fuel": "petrol diesel fuel pump car bike",
    "FASTag toll": "toll highway car fastag",
    "Mobility": "vehicle rental bike scooter",
    "MakeMyTrip": "flights hotels holiday booking trip",
    "Travel booking": "flights hotels holiday booking trip",
    "IRCTC": "train railway tickets",
    "Airline": "flight airline tickets airport",
    "Hotel booking": "hotel stay lodging trip",
    "Bus booking": "bus tickets intercity travel",
    "State transport bus": "bus tickets government transport",
    "Travel agency": "holiday tour package trip",
    # shopping
    "Amazon": "online shopping ecommerce orders delivery",
    "Flipkart": "online shopping ecommerce electronics",
    "Myntra": "clothes fashion apparel shoes",
    "AJIO": "clothes fashion apparel",
    "Meesho": "clothes fashion online shopping",
    "Tata CLiQ": "online shopping electronics fashion",
    "Electronics store": "electronics phone laptop gadgets appliances",
    "Decathlon": "sports equipment shoes fitness gear",
    "Home furnishing": "furniture home decor furnishing",
    "Fashion store": "clothes fashion apparel shoes",
    "Lenskart": "glasses spectacles eyewear lenses",
    "FirstCry": "baby kids products toys",
    "Nykaa Fashion": "beauty cosmetics fashion",
    "Jewellery": "gold jewellery ornaments",
    # health & personal care
    "Cult.fit": "gym fitness workout membership",
    "Gym membership": "gym fitness workout",
    "Apollo Pharmacy": "pharmacy medicines medical chemist hospital",
    "Pharmacy": "pharmacy medicines medical medicals chemist drugs",
    "Hospital/Clinic": "hospital clinic doctor consultation",
    "Diagnostics lab": "lab tests blood test diagnostics",
    "Healthcare": "doctor medical health",
    "Urban Company": "home services salon cleaning repair",
    "Salon": "salon haircut grooming beauty",
    "Beauty": "beauty cosmetics skincare salon",
    # bills & home
    "Airtel": "mobile phone recharge broadband internet",
    "Jio": "mobile phone recharge internet",
    "Vi": "mobile phone recharge",
    "BSNL": "mobile phone landline broadband",
    "Broadband": "internet wifi broadband",
    "DTH TV": "television cable dth recharge",
    "Electricity": "electricity power bill",
    "Water": "water bill",
    "Piped gas": "gas bill cooking",
    "Cooking gas": "lpg gas cylinder cooking",
    "Society maintenance": "apartment maintenance society housing",
    "Co-living rent": "rent housing accommodation",
    "Rent": "rent housing landlord",
    # entertainment & subscriptions
    "BookMyShow": "movie movies tickets events cinema",
    "Cinema": "movie movies cinema film tickets popcorn",
    "Events": "concert event tickets",
    "Gaming": "video games gaming",
    "Fantasy gaming": "fantasy cricket gaming betting",
    "Amusement": "theme park amusement fun",
    "Netflix": "streaming films series shows ott",
    "Prime Video": "streaming films series shows ott",
    "Disney+ Hotstar": "streaming films cricket ott",
    "Spotify": "music streaming songs",
    "YouTube Premium": "video music streaming",
    "Audible": "audiobooks",
    "Microsoft 365": "office software",
    "Adobe": "design software",
    "ChatGPT Plus": "ai assistant software",
    "Claude Pro": "ai assistant software",
    # money
    "Loan EMI": "loan emi repayment",
    "Digital lender": "loan app emi repayment",
    "Mutual fund": "mutual fund sip investment",
    "Brokerage / MF": "stocks shares trading investment",
    "SIP / MF": "mutual fund sip investment",
    "Insurance premium": "insurance policy premium",
    "ATM withdrawal": "cash atm withdrawal",
    "Credit card payment": "credit card bill repayment",
    "Salary": "salary payroll employer income",
    "Interest / dividend": "interest dividend earnings",
    "Donation": "donation charity temple",
    "Gift shop": "gifts presents",
    "Online course": "course learning tuition",
    "Education": "school college tuition fees course",
}


def rule_category(rule_name: str) -> str | None:
    from finpilot.domain.lexicon import MERCHANT_RULES

    return next((r.category for r in MERCHANT_RULES if r.name == rule_name), None)


def word_forms(word: str) -> set[str]:
    """Singular/plural forms of an English word, upper-case ("MOVIES" → MOVIES, MOVIE)."""
    w = word.upper()
    out = {w}
    if len(w) > 4 and w.endswith("IES"):
        out |= {w[:-3] + "Y", w[:-1]}
    elif len(w) > 3 and w.endswith("ES"):
        out |= {w[:-2], w[:-1]}
    elif len(w) > 3 and w.endswith("S"):
        out.add(w[:-1])
    return out


def describe(merchant: str, description: str = "") -> str:
    """Descriptor words for a merchant ('' when it is not a known brand or kind of business)."""
    rule = match_merchant(normalize_text(f"{merchant} {description}"))
    if rule is None:
        return ""
    return BRAND_TAGS.get(rule.name, "")
