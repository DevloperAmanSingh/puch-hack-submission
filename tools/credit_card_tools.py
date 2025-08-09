from typing import Annotated, List, Dict, Optional

from fastmcp import FastMCP
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import BaseModel, Field

from services.credit_card_search import get_search_service
from upstash_redis import Redis
import os

REDIS_URL = os.getenv("UPSTASH_REDIS_URL")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN")
_cc_redis = Redis(url=REDIS_URL, token=REDIS_TOKEN) if (REDIS_URL and REDIS_TOKEN) else None


def _cc_rate_limit_allow(key: str, limit: int, window_seconds: int) -> bool:
    if _cc_redis is None:
        return True
    try:
        p = _cc_redis.pipeline()
        p.incr(key)
        p.ttl(key)
        res = p.exec()
        count = int(res[0])
        ttl = int(res[1]) if res[1] is not None else -1
        if ttl == -1:
            _cc_redis.expire(key, window_seconds)
        return count <= limit
    except Exception:
        return True


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: Optional[str] = None


class CreditCardResponse(BaseModel):
    answer: str
    cards_mentioned: List[str] = []
    banks_mentioned: List[str] = []
    query: str
    sources_count: int = 0


class CreditCardComparison(BaseModel):
    comparison_result: str
    cards_compared: List[Dict[str, str]]
    winner: Optional[str] = None
    criteria_used: List[str] = []


def _extract_cards_banks(sources: List[Dict]) -> tuple[List[str], List[str]]:
    cards: List[str] = []
    banks: List[str] = []
    for s in sources or []:
        c = s.get("card_name", "")
        b = s.get("bank", "")
        if c and c not in cards:
            cards.append(c)
        if b and b not in banks:
            banks.append(b)
    return cards, banks


def register(mcp: FastMCP) -> None:
    SearchDesc = RichToolDescription(
        description="Search for credit card information, recommendations, and detailed analysis",
        use_when="When user asks about credit cards, features, benefits, fees, eligibility, or wants recommendations",
        side_effects="Returns structured information with sources from Pinecone",
    )

    @mcp.tool(description=SearchDesc.model_dump_json())
    async def search_credit_cards(
        puch_user_id: Annotated[str, Field(description="Opaque string reference for the user")],
        query: Annotated[str, Field(description="Credit card question or search query")],
        include_sources: Annotated[
            bool,
            Field(default=True, description="Include source info"),
        ] = True,
    ) -> List[TextContent]:
        if not query.strip():
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Query cannot be empty"))
        # Rate limit: 5 per 4 hours
        window_seconds = 4 * 60 * 60
        rl_key = f"rate:cc:search:{puch_user_id}"
        if not _cc_rate_limit_allow(rl_key, 5, window_seconds):
            return [TextContent(type="text", text="Rate limit exceeded for credit card search. Try again after 4 hours.")]
        try:
            svc = get_search_service()
            result = svc.search(query, include_sources=include_sources)
            cards, banks = _extract_cards_banks(result.get("sources", []))
            response = CreditCardResponse(
                answer=result["answer"],
                cards_mentioned=cards,
                banks_mentioned=banks,
                query=query,
                sources_count=len(result.get("sources", [])),
            )
            out = f"""## Credit Card Search Results

**Query:** {query}

**Answer:**
{result['answer']}

**Cards Mentioned:** {', '.join(cards) if cards else 'None'}
**Banks Mentioned:** {', '.join(banks) if banks else 'None'}
**Sources Used:** {response.sources_count}
"""
            if include_sources and result.get("sources"):
                out += "\n**Detailed Sources:**\n"
                for i, s in enumerate(result["sources"], 1):
                    out += f"\n{i}. **{s.get('card_name','')}** ({s.get('bank','')})\n"
                    out += f"   └─ {s.get('content_snippet','')}\n"
            return [TextContent(type="text", text=out)]
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Search failed: {e}"))

    CompDesc = RichToolDescription(
        description="Compare multiple credit cards based on specific criteria",
        use_when="When user wants comparisons or help choosing between options",
        side_effects="Returns detailed comparison with structured analysis",
    )

    @mcp.tool(description=CompDesc.model_dump_json())
    async def compare_credit_cards(
        puch_user_id: Annotated[str, Field(description="Opaque string reference for the user")],
        cards_or_criteria: Annotated[
            str,
            Field(description="Cards to compare or comparison criteria"),
        ],
        comparison_factors: Annotated[
            str,
            Field(
                default="fees, benefits, rewards, eligibility",
                description="Factors to compare",
            ),
        ] = "fees, benefits, rewards, eligibility",
    ) -> List[TextContent]:
        # Rate limit: 5 per 4 hours
        window_seconds = 4 * 60 * 60
        rl_key = f"rate:cc:compare:{puch_user_id}"
        if not _cc_rate_limit_allow(rl_key, 5, window_seconds):
            return [TextContent(type="text", text="Rate limit exceeded for credit card comparison. Try again after 4 hours.")]
        try:
            svc = get_search_service()
            comparison_query = f"Compare {cards_or_criteria} focusing on {comparison_factors}"
            result = svc.search(comparison_query, include_sources=True)
            cards, banks = _extract_cards_banks(result.get("sources", []))
            cards_compared = [{"card_name": c, "bank": b} for c, b in zip(cards, banks)]
            criteria_list = [x.strip() for x in comparison_factors.split(",")]
            comparison = CreditCardComparison(
                comparison_result=result["answer"],
                cards_compared=cards_compared,
                criteria_used=criteria_list,
            )
            out = f"""## Credit Card Comparison

**Comparison Request:** {cards_or_criteria}
**Factors Analyzed:** {comparison_factors}

**Comparison Result:**
{result['answer']}

**Cards in Comparison:**
"""
            for i, info in enumerate(cards_compared, 1):
                out += f"{i}. {info['card_name']} ({info['bank']})\n"

            out += f"""
**Analysis Criteria:** {', '.join(criteria_list)}

---
**Structured Comparison Data:**
```json
{comparison.model_dump_json(indent=2)}
```
"""
            return [TextContent(type="text", text=out)]
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Comparison failed: {e}"))

    RecDesc = RichToolDescription(
        description="Get personalized credit card recommendations",
        use_when="When user needs cards matched to spending patterns or preferences",
        side_effects="Returns recommendations with explanations",
    )

    @mcp.tool(description=RecDesc.model_dump_json())
    async def recommend_credit_cards(
        puch_user_id: Annotated[str, Field(description="Opaque string reference for the user")],
        user_profile: Annotated[
            str,
            Field(description="Spending pattern, income range, or preferences"),
        ],
        preferences: Annotated[
            str,
            Field(
                default="low fees, good rewards",
                description="Specific preferences",
            ),
        ] = "low fees, good rewards",
        max_recommendations: Annotated[
            int,
            Field(default=5, ge=1, le=10, description="Max number of cards"),
        ] = 5,
    ) -> List[TextContent]:
        # Rate limit: 5 per 4 hours
        window_seconds = 4 * 60 * 60
        rl_key = f"rate:cc:recommend:{puch_user_id}"
        if not _cc_rate_limit_allow(rl_key, 5, window_seconds):
            return [TextContent(type="text", text="Rate limit exceeded for credit card recommendations. Try again after 4 hours.")]
        try:
            svc = get_search_service()
            q = (
                f"Recommend {max_recommendations} best credit cards for {user_profile} "
                f"with preferences for {preferences}"
            )
            result = svc.search(q, include_sources=True)
            cards, banks = _extract_cards_banks(result.get("sources", []))
            out = f"""## Personalized Credit Card Recommendations

**User Profile:** {user_profile}
**Preferences:** {preferences}
**Max Recommendations:** {max_recommendations}

**Recommendations:**
{result['answer']}

**Recommended Cards:**
"""
            for i, (c, b) in enumerate(zip(cards, banks), 1):
                if i <= max_recommendations:
                    out += f"{i}. **{c}** - {b}\n"

            out += f"""
**Sources Analyzed:** {len(result.get('sources', []))} credit card documents
"""
            return [TextContent(type="text", text=out)]
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Recommendation failed: {e}"))

    HealthDesc = RichToolDescription(
        description="Check health status of the credit card search service",
        use_when="Verify service and database connectivity",
        side_effects="Returns service and DB status",
    )

    @mcp.tool(description=HealthDesc.model_dump_json())
    async def health_check() -> List[TextContent]:
        try:
            svc = get_search_service()
            health = svc.get_health_status()
            import json

            out = f"""## Credit Card Service Health Check

**Service Status:** {health.get('status','unknown')}
**Database Status:** {health.get('vectordb_status','unknown')}
**Documents in Database:** {health.get('documents_count','unknown')}

**Health Status:**
```json
{json.dumps(health, indent=2)}
```
"""
            return [TextContent(type="text", text=out)]
        except Exception as e:
            return [TextContent(type="text", text=f"## Health Check Failed\n\n**Error:** {e}\n\n**Status:** Service unavailable")]


