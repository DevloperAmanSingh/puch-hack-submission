from typing import Annotated, List, Dict, Optional
import os
import httpx
import json
from datetime import datetime, timedelta
from fastmcp import FastMCP
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import BaseModel, Field
import re
from os import getenv
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from upstash_redis import Redis

REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")
_redis = Redis(url=REDIS_URL, token=REDIS_TOKEN) if (REDIS_URL and REDIS_TOKEN) else None


def _rate_limit_allow(key: str, limit: int, window_seconds: int) -> bool:
    if _redis is None:
        return True
    # Use a rolling counter with TTL
    try:
        pipe = _redis.pipeline()
        pipe.incr(key)
        pipe.ttl(key)
        res = pipe.exec()
        count = int(res[0])
        ttl = int(res[1]) if res[1] is not None else -1
        if ttl == -1:
            _redis.expire(key, window_seconds)
        if count > limit:
            return False
        return True
    except Exception:
        return True


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


class FlightInfo(BaseModel):
    airline: str
    flight_number: str
    departure_time: str
    arrival_time: str
    duration: str
    price: str
    booking_link: str
    stops: str


class FlightSearchService:
    """Flight search service using RapidAPI (Booking.com) one-way flights API."""
    # RapidAPI (Booking.com)
    RAPIDAPI_KEY: Optional[str] = "812bb753c6mshbd86f69a010307dp1f49adjsn99fc9f36dc34"
    RAPIDAPI_HOST: str = os.getenv("RAPIDAPI_HOST", "booking-com18.p.rapidapi.com")
    RAPIDAPI_BASE: str = "https://booking-com18.p.rapidapi.com"

    @classmethod
    async def _get_usd_to_inr_rate(cls) -> float:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=10)
                r.raise_for_status()
                rates = (r.json() or {}).get("rates", {})
                return float(rates.get("INR", 83.0))
        except Exception:
            return 83.0

    @classmethod
    async def _rapidapi_search_oneway(
        cls,
        depart_id: str,
        arrival_id: str,
        depart_date: str,
        top_k: int,
    ) -> Optional[List[Dict]]:
        """Call RapidAPI one-way search and return normalized flight list."""
        if not cls.RAPIDAPI_KEY:
            print("[RAPIDAPI] Missing RAPIDAPI_KEY in env")
            return None

        url = f"{cls.RAPIDAPI_BASE}/flights/v2/search-oneway"
        params = {
            "departId": depart_id,
            "arrivalId": arrival_id,
            "departDate": depart_date,
        }
        headers = {
            "x-rapidapi-host": cls.RAPIDAPI_HOST,
            "x-rapidapi-key": cls.RAPIDAPI_KEY,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, params=params, timeout=40)
                resp.raise_for_status()
                body = resp.json()
        except Exception:
            return None

        data = (body or {}).get("data", {})
        flight_offers = (data or {}).get("flightOffers")
        if not flight_offers:
            aggregation = (data or {}).get("aggregation", {})
            flight_offers = aggregation.get("flightOffers", [])

        if isinstance(flight_offers, dict):
            sorted_items = [flight_offers[k] for k in sorted(flight_offers.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)]
        elif isinstance(flight_offers, list):
            sorted_items = flight_offers
        else:
            sorted_items = []

        flights: List[Dict] = []
        usd_to_inr: float = await cls._get_usd_to_inr_rate()

        def _parse_money_dict(m: Dict) -> Optional[float]:
            try:
                if not isinstance(m, dict):
                    return None
                # Common shapes
                if {"units", "nanos"}.issubset(m.keys()):
                    units = float(m.get("units", 0))
                    nanos = float(m.get("nanos", 0))
                    return units + nanos / 1_000_000_000
                if "amount" in m and isinstance(m["amount"], dict):
                    return _parse_money_dict(m["amount"])
                if "value" in m and isinstance(m["value"], (int, float, str)):
                    return float(str(m["value"]).replace(",", ""))
                if "raw" in m and isinstance(m["raw"], (int, float, str)):
                    return float(str(m["raw"]).replace(",", ""))
            except Exception:
                return None
            return None

        def _format_time(ts: Optional[str]) -> str:
            if not ts or not isinstance(ts, str):
                return "-"
            try:
                # Expecting e.g. 2025-08-09T23:45:00
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00").split("+", 1)[0])
                return dt.strftime("%H:%M")
            except Exception:
                try:
                    # Fallback to split at 'T'
                    return ts.split("T", 1)[1][:5]
                except Exception:
                    return ts

        def _extract_price(offer: Dict) -> Optional[float]:
            candidate_paths = [
                ["price", "grandTotal"],
                ["price", "total"],
                ["price", "amount"],
                ["totalPrice"],
                ["displayPrice"],
                ["minPrice"],
                ["amount"],
                ["fare", "total"],
                ["pricing", "total"],
            ]
            for path in candidate_paths:
                val: Dict | float | str | None = offer
                for key in path:
                    if isinstance(val, dict) and key in val:
                        val = val[key]
                    else:
                        val = None
                        break
                if val is not None:
                    if isinstance(val, dict):
                        parsed = _parse_money_dict(val)
                        if parsed is not None:
                            return parsed * usd_to_inr
                    try:
                        return float(str(val).replace(",", "").replace("₹", "").strip()) * usd_to_inr
                    except Exception:
                        continue
            pb = offer.get("priceBreakdown") or offer.get("pricing")
            if isinstance(pb, dict):
                try:
                    for key in ["raw", "value", "total", "grandTotal", "gross", "amount"]:
                        if key in pb:
                            if isinstance(pb[key], dict):
                                parsed = _parse_money_dict(pb[key])
                                if parsed is not None:
                                    return parsed * usd_to_inr
                            try:
                                return float(str(pb[key]).replace(",", "")) * usd_to_inr
                            except Exception:
                                continue
                except Exception:
                    pass
            return None

        for idx, offer in enumerate(sorted_items[: max(1, min(top_k, 5))]):
            try:
                segments = offer.get("segments") or []
                segment0 = segments[0] if segments else {}
                legs = segment0.get("legs") or []
                leg0 = legs[0] if legs else {}

                dep_time = _format_time(leg0.get("departureTime") or segment0.get("departureTime"))
                arr_time = _format_time(leg0.get("arrivalTime") or segment0.get("arrivalTime"))

                carriers_data = (
                    leg0.get("carriersData")
                    or segment0.get("carriersData")
                    or offer.get("carriersData")
                    or []
                )
                airline_name = None
                airline_code = None
                airline_logo = None
                if isinstance(carriers_data, list) and carriers_data:
                    airline_name = carriers_data[0].get("name")
                    airline_code = carriers_data[0].get("code")
                    airline_logo = carriers_data[0].get("logo")
                if not airline_name:
                    carriers = offer.get("carriers") or []
                    if carriers:
                        airline_code = carriers[0]
                        airline_name = carriers[0]
                airline_name = airline_name or airline_code or "Unknown"

                flight_info = leg0.get("flightInfo") or {}
                flight_number = flight_info.get("flightNumber", "")
                plane_type = flight_info.get("planeType", "")
                carrier_info = flight_info.get("carrierInfo", {})
                operating_carrier = carrier_info.get("operatingCarrier", "")
                marketing_carrier = carrier_info.get("marketingCarrier", "")

                dep_airport = leg0.get("departureAirport") or segment0.get("departureAirport") or {}
                arr_airport = leg0.get("arrivalAirport") or segment0.get("arrivalAirport") or {}
                dep_airport_name = dep_airport.get("name", "")
                arr_airport_name = arr_airport.get("name", "")
                dep_city = dep_airport.get("cityName", "")
                arr_city = arr_airport.get("cityName", "")

                total_time = leg0.get("totalTime") or segment0.get("totalTime")
                flight_stops = leg0.get("flightStops") or []
                stops_count = len(flight_stops) if flight_stops else 0
                duration_label = "Direct" if stops_count == 0 else f"{stops_count} stop{'s' if stops_count > 1 else ''}"

                price_val = _extract_price(offer)
                flights.append(
                    {
                        "airline": airline_name,
                        "airline_code": airline_code,
                        "airline_logo": airline_logo,
                        "flight_number": flight_number,
                        "plane_type": plane_type,
                        "operating_carrier": operating_carrier,
                        "marketing_carrier": marketing_carrier,
                        "dep_time": dep_time,
                        "arr_time": arr_time,
                        "dep_airport_name": dep_airport_name,
                        "arr_airport_name": arr_airport_name,
                        "dep_city": dep_city,
                        "arr_city": arr_city,
                        "duration": duration_label,
                        "total_time": total_time,
                        "price": price_val,  # may be None
                        "depart_id": depart_id,
                        "arrival_id": arrival_id,
                        "date": depart_date,
                    }
                )
            except Exception:
                pass

        return flights

    @classmethod
    async def _amadeus_resolve_city_code(cls, city_name: str) -> Optional[str]:
        return None

    @classmethod
    async def _amadeus_search(
        cls,
        from_city: str,
        to_city: str,
        date: str,
        top_k: int,
    ) -> Optional[List[Dict]]:
        return None

    @classmethod
    async def search_flights(
        cls, 
        depart_id: str,
        arrival_id: str,
        date: str = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Search for flights using RapidAPI Booking.com one-way endpoint."""

        # If no date provided, use tomorrow
        if not date:
            tomorrow = datetime.now() + timedelta(days=1)
            date = tomorrow.strftime("%Y-%m-%d")

        flights = await cls._rapidapi_search_oneway(depart_id, arrival_id, date, top_k)
        if not flights:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="RapidAPI returned no results or failed"))
        return flights
    
    @classmethod
    def _get_demo_flights(cls, from_city: str, to_city: str, date: str, top_k: int) -> List[Dict]:
        """Deprecated: demo flights removed as per request."""
        return []


def _get_iata_code(city_or_code: str) -> str:
    """Convert city name to IATA code or return as-is if already a code."""
    city_mapping = {
        # Major Indian cities
        "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM", "bombay": "BOM",
        "bangalore": "BLR", "banglore": "BLR", "bengaluru": "BLR", "chennai": "MAA", "madras": "MAA",
        "hyderabad": "HYD", "kolkata": "CCU", "calcutta": "CCU", "pune": "PNQ",
        "ahmedabad": "AMD", "jaipur": "JAI", "lucknow": "LKO", "guwahati": "GAU",
        "patna": "PAT", "bhopal": "BHO", "indore": "IDR", "vadodara": "BDQ",
        "nagpur": "NAG", "coimbatore": "CJB", "kochi": "COK", "thiruvananthapuram": "TRV",
        "visakhapatnam": "VTZ", "bhubaneswar": "BBI", "raipur": "RPR", "jabalpur": "JLR",
        "amritsar": "ATQ", "varanasi": "VNS", "srinagar": "SXR", "leh": "IXL",
        "shimla": "SLV", "dehradun": "DED", "chandigarh": "IXC", "goa": "GOI",
        "port blair": "IXZ", "imphal": "IMF", "agartala": "IXA", "aizawl": "AJL",
        "dimapur": "DMU", "jodhpur": "JDH", "udaipur": "UDR", "jaisalmer": "JSA",
        "bikaner": "BKB", "ajmer": "AJM", "kota": "KTU", "bundi": "BUN",
        # International cities
        "london": "LHR", "new york": "JFK", "dubai": "DXB", "singapore": "SIN",
        "bangkok": "BKK", "tokyo": "NRT", "paris": "CDG", "frankfurt": "FRA",
        "amsterdam": "AMS", "rome": "FCO", "madrid": "MAD", "barcelona": "BCN",
        "milan": "MXP", "zurich": "ZRH", "vienna": "VIE", "prague": "PRG",
        "budapest": "BUD", "warsaw": "WAW", "moscow": "SVO", "istanbul": "IST",
        "doha": "DOH", "abu dhabi": "AUH", "muscat": "MCT", "riyadh": "RUH",
        "jeddah": "JED", "cairo": "CAI", "nairobi": "NBO", "lagos": "LOS",
        "johannesburg": "JNB", "cape town": "CPT", "sydney": "SYD", "melbourne": "MEL",
        "brisbane": "BNE", "perth": "PER", "adelaide": "ADL", "auckland": "AKL",
        "wellington": "WLG", "fiji": "NAN", "port moresby": "POM", "honolulu": "HNL",
        "los angeles": "LAX", "san francisco": "SFO", "chicago": "ORD", "miami": "MIA",
        "toronto": "YYZ", "vancouver": "YVR", "montreal": "YUL", "calgary": "YYC",
        "edmonton": "YEG", "winnipeg": "YWG", "ottawa": "YOW", "halifax": "YHZ",
        "quebec": "YQB", "victoria": "YYJ", "kelowna": "YLW", "regina": "YQR",
        "saskatoon": "YXE", "thunder bay": "YQT", "sudbury": "YSB", "windsor": "YQG",
        "hamilton": "YHM", "london": "YXU", "kingston": "YGK", "saint john": "YSJ",
        "fredericton": "YFC", "charlottetown": "YYG", "st johns": "YYT", "whitehorse": "YXY",
        "yellowknife": "YZF", "iqaluit": "YFB", "rankin inlet": "YRT", "kuujjuaq": "YVP",
        "val d or": "YVO", "rouyn noranda": "YUY", "saguenay": "YBG", "gaspe": "YGP",
        "bagotville": "YBG", "valcartier": "YOY", "goose bay": "YYR", "deer lake": "YDF",
        "gander": "YQX", "stephenville": "YJT", "corner brook": "YDF", "grand falls": "YQX",
        "labrador city": "YLR", "churchill": "YYQ", "thompson": "YTH", "brandon": "YBR",
        "saskatoon": "YXE", "regina": "YQR", "prince albert": "YPA", "lloydminster": "YLL",
        "medicine hat": "YXH", "lethbridge": "YQL", "red deer": "YQF", "grande prairie": "YQU",
        "fort mcmurray": "YMM", "yellowknife": "YZF", "whitehorse": "YXY", "dawson creek": "YDQ",
        "fort st john": "YXJ", "prince george": "YXS", "kamloops": "YKA", "kelowna": "YLW",
        "victoria": "YYJ", "nanaimo": "YCD", "comox": "YQQ", "campbell river": "YBL",
        "terrace": "YXT", "prince rupert": "YPR", "fort nelson": "YYE", "fort st john": "YXJ",
        "dawson creek": "YDQ", "grande prairie": "YQU", "edmonton": "YEG", "calgary": "YYC",
        "red deer": "YQF", "lethbridge": "YQL", "medicine hat": "YXH", "lloydminster": "YLL",
        "prince albert": "YPA", "regina": "YQR", "saskatoon": "YXE", "brandon": "YBR",
        "thompson": "YTH", "churchill": "YYQ", "labrador city": "YLR", "grand falls": "YQX",
        "corner brook": "YDF", "stephenville": "YJT", "gander": "YQX", "deer lake": "YDF",
        "goose bay": "YYR", "valcartier": "YOY", "bagotville": "YBG", "gaspe": "YGP",
        "saguenay": "YBG", "rouyn noranda": "YUY", "val d or": "YVO", "kuujjuaq": "YVP",
        "rankin inlet": "YRT", "iqaluit": "YFB", "yellowknife": "YZF", "whitehorse": "YXY",
        "st johns": "YYT", "charlottetown": "YYG", "fredericton": "YFC", "saint john": "YSJ",
        "kingston": "YGK", "london": "YXU", "hamilton": "YHM", "windsor": "YQG",
        "sudbury": "YSB", "thunder bay": "YQT", "regina": "YQR", "saskatoon": "YXE",
        "kelowna": "YLW", "victoria": "YYJ", "halifax": "YHZ", "ottawa": "YOW",
        "winnipeg": "YWG", "edmonton": "YEG", "calgary": "YYC", "vancouver": "YVR",
        "montreal": "YUL", "toronto": "YYZ", "miami": "MIA", "chicago": "ORD",
        "san francisco": "SFO", "los angeles": "LAX", "honolulu": "HNL", "port moresby": "POM",
        "fiji": "NAN", "wellington": "WLG", "auckland": "AKL", "adelaide": "ADL",
        "perth": "PER", "brisbane": "BNE", "melbourne": "MEL", "sydney": "SYD",
        "cape town": "CPT", "johannesburg": "JNB", "lagos": "LOS", "nairobi": "NBO",
        "cairo": "CAI", "jeddah": "JED", "riyadh": "RUH", "muscat": "MCT",
        "abu dhabi": "AUH", "doha": "DOH", "istanbul": "IST", "moscow": "SVO",
        "warsaw": "WAW", "budapest": "BUD", "prague": "PRG", "vienna": "VIE",
        "zurich": "ZRH", "milan": "MXP", "barcelona": "BCN", "madrid": "MAD",
        "rome": "FCO", "amsterdam": "AMS", "frankfurt": "FRA", "paris": "CDG",
        "tokyo": "NRT", "bangkok": "BKK", "singapore": "SIN", "dubai": "DXB",
        "new york": "JFK", "london": "LHR", "bundi": "BUN", "kota": "KTU",
        "ajmer": "AJM", "bikaner": "BKB", "jaisalmer": "JSA", "udaipur": "UDR",
        "jodhpur": "JDH", "dimapur": "DMU", "aizawl": "AJL", "agartala": "IXA",
        "imphal": "IMF", "port blair": "IXZ", "goa": "GOI", "chandigarh": "IXC",
        "dehradun": "DED", "shimla": "SLV", "leh": "IXL", "srinagar": "SXR",
        "varanasi": "VNS", "amritsar": "ATQ", "jabalpur": "JLR", "raipur": "RPR",
        "bhubaneswar": "BBI", "visakhapatnam": "VTZ", "thiruvananthapuram": "TRV",
        "kochi": "COK", "coimbatore": "CJB", "nagpur": "NAG", "vadodara": "BDQ",
        "indore": "IDR", "bhopal": "BHO", "patna": "PAT", "guwahati": "GAU",
        "lucknow": "LKO", "jaipur": "JAI", "ahmedabad": "AMD", "pune": "PNQ",
        "calcutta": "CCU", "kolkata": "CCU", "madras": "MAA", "bengaluru": "BLR",
        "bombay": "BOM", "new delhi": "DEL"
    }

    # If it's already a 3-letter code, return as-is
    if len(city_or_code) == 3 and city_or_code.isupper():
        return city_or_code

    # Look up in mapping (case-insensitive)
    return city_mapping.get(city_or_code.lower(), city_or_code.upper())


def _parse_freeform_query_to_params(message: str) -> Optional[Dict[str, str]]:
    """Use OpenAI to extract flight params from a natural language message.
    Returns dict with keys: from, to, date (YYYY-MM-DD), or None on failure.
    """
    if not message or not isinstance(message, str):
        return None

    if OpenAI is None:
        return None

    api_key = getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    system = (
        "You extract structured flight search parameters from a user's message. "
        "Return a compact JSON object with keys: from, to, date (YYYY-MM-DD). "
        "Resolve ambiguous or misspelled city names to the correct major airport city. "
        "If only month/day words are given, infer year as next occurrence in the future."
    )
    prompt = f"Message: {message}\nRespond only with JSON."
    try:
        resp = client.chat.completions.create(
            model=getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        # Extract JSON object from response
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            return None
        parsed = json.loads(m.group(0))
        out = {
            "from": _get_iata_code(str(parsed.get("from", "")).strip()),
            "to": _get_iata_code(str(parsed.get("to", "")).strip()),
            "date": str(parsed.get("date", "")).strip(),
        }
        return out if out["from"] and out["to"] and out["date"] else None
    except Exception:
        return None


def register(mcp: FastMCP) -> None:
    """Register flight search tools with MCP server"""
    
    # Main flight search tool
    FlightSearchDesc = RichToolDescription(
        description="Search for one-way flights between cities or airports with detailed flight information",
        use_when="When user wants to find flights between two cities or airports with pricing and schedule details",
        side_effects="Returns detailed flight options with airline info, times, airports, and prices"
    )
    
    @mcp.tool(description=FlightSearchDesc.model_dump_json())
    async def search_flights(
        puch_user_id: Annotated[str, Field(description="Opaque string reference for the user")],
        from_airport: Annotated[str, Field(description="Departure city/airport (e.g., Delhi, DEL, Mumbai, BOM)")],
        to_airport: Annotated[str, Field(description="Arrival city/airport (e.g., Chennai, MAA, Bangalore, BLR)")],
        date: Annotated[Optional[str], Field(description="Travel date (YYYY-MM-DD format, optional)")] = None,
        top_results: Annotated[int, Field(description="Number of top results to show", ge=1, le=10)] = 5,
    ) -> List[TextContent]:
        """Search for one-way flights and return formatted text output."""
        try:
            # Rate limit: 3 requests / 4 hours per user
            window_seconds = 4 * 60 * 60
            rl_key = f"rate:flights:{puch_user_id}"
            if not _rate_limit_allow(rl_key, 2, window_seconds):
                return [TextContent(type="text", text="Rate limit exceeded for flights. Try again after 4 hours.")]

            # AI parse path: if user passes a freeform message in from_airport and leaves to/date blank
            if (not date and to_airport.strip().lower() in {"", "?", "-"}) or (len(from_airport.split()) > 2 and not date):
                combined = from_airport
                parsed = _parse_freeform_query_to_params(combined)
                if parsed:
                    from_iata = parsed["from"]
                    to_iata = parsed["to"]
                    date = parsed["date"]
                else:
                    # fallback to mapping
                    from_iata = _get_iata_code(from_airport)
                    to_iata = _get_iata_code(to_airport or "")
            else:
                from_iata = _get_iata_code(from_airport)
                to_iata = _get_iata_code(to_airport)
            
            flights = await FlightSearchService.search_flights(from_iata, to_iata, date, top_results)
            
            if not flights:
                return [TextContent(type="text", text=f"No flights found from {from_airport} to {to_airport}")]

            try:
                date_label = datetime.strptime(flights[0]["date"], "%Y-%m-%d").strftime("%d %b %Y")
            except Exception:
                date_label = flights[0]["date"]

            parts: List[str] = [
                f"🛩️ {from_iata} ➡️ {to_iata} — {date_label}",
                "",
            ]

            for f in flights:
                price_label = (
                    f"₹{int(f['price']):,}" if isinstance(f.get("price"), (int, float)) else "₹-"
                )
                airline_display = f"{f['airline']}"
                if f.get('flight_number'):
                    airline_display += f" ({f['flight_number']})"
                if f.get('plane_type'):
                    airline_display += f" • {f['plane_type']}"

                dep_info = f"{f['dep_city']} ({f['depart_id']})" if f.get('dep_city') else f"{f['depart_id']}"
                arr_info = f"{f['arr_city']} ({f['arrival_id']})" if f.get('arr_city') else f"{f['arrival_id']}"

                parts.append(f"✈️ {airline_display}")
                parts.append(f"🛫 {f['dep_time']} • {dep_info}")
                parts.append(f"🛬 {f['arr_time']} • {arr_info}")
                parts.append(f"⏱️ {f['duration']} • 💰 {price_label}")
                parts.append("")

            formatted_response = "\n".join(parts).strip()
            return [TextContent(type="text", text=formatted_response)]
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Flight search failed: {e}"))
    
