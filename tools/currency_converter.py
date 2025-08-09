from typing import Annotated, List
import httpx
import json
from fastmcp import FastMCP
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import BaseModel, Field


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


class CurrencyConverter:
    """Currency converter using free exchange rate API"""
    
    BASE_URL = "https://api.exchangerate-api.com/v4/latest/{}"
    
    @classmethod
    async def get_exchange_rates(cls, base_currency: str = "INR") -> dict:
        """Fetch current exchange rates for a given base currency (3-letter code)."""
        base = (base_currency or "").upper()
        if len(base) != 3:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid base currency code"))
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(cls.BASE_URL.format(base), timeout=10)
                response.raise_for_status()
                data = response.json()
                return data.get("rates", {})
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch exchange rates: {e}"))
    
    @classmethod
    async def convert_any(cls, amount: float, from_currency: str, to_currency: str) -> dict:
        """Convert amount from any currency to any currency using live rates."""
        if amount <= 0:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Amount must be greater than 0"))
        src = (from_currency or "").upper()
        dst = (to_currency or "").upper()
        if len(src) != 3 or len(dst) != 3:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Currencies must be 3-letter ISO codes"))
        if src == dst:
            return {
                "from_currency": src,
                "to_currency": dst,
                "amount": amount,
                "converted_amount": round(amount, 2),
                "exchange_rate": 1.0,
                "last_updated": "Current rates from exchangerate-api.com",
            }
        rates = await cls.get_exchange_rates(src)
        if dst not in rates:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Currency {dst} not supported"))
        rate = float(rates[dst])
        converted_amount = amount * rate
        return {
            "from_currency": src,
            "to_currency": dst,
            "amount": amount,
            "converted_amount": round(converted_amount, 2),
            "exchange_rate": rate,
            "last_updated": "Current rates from exchangerate-api.com",
        }


def register(mcp: FastMCP) -> None:
    """Register currency converter tools with MCP server"""
    
    # Single currency conversion tool
    SingleConversionDesc = RichToolDescription(
        description="Convert between any two currencies using real-time exchange rates",
        use_when="When user wants to convert money from one currency to another",
        side_effects="Returns converted amount with current exchange rate"
    )
    
    @mcp.tool(description=SingleConversionDesc.model_dump_json())
    async def convert_currency(
        amount: Annotated[float, Field(description="Amount to convert", ge=0.01)],
        from_currency: Annotated[str, Field(description="Source currency (ISO 4217, e.g., INR, USD, EUR)", pattern="^[A-Za-z]{3}$")],
        to_currency: Annotated[str, Field(description="Target currency (ISO 4217, e.g., USD, EUR, INR)", pattern="^[A-Za-z]{3}$")]
    ) -> List[TextContent]:
        """Convert between any two currencies"""
        try:
            result = await CurrencyConverter.convert_any(amount, from_currency, to_currency)
            
            formatted_response = f"""## Currency Conversion

**From:** {result['amount']} {result['from_currency']}
**To:** {result['converted_amount']} {result['to_currency']}
**Exchange Rate:** 1 {result['from_currency']} = {result['exchange_rate']} {result['to_currency']}

**JSON Data:**
```json
{json.dumps(result, indent=2)}
```"""
            
            return [TextContent(type="text", text=formatted_response)]
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Conversion failed: {e}"))
    
    # Exchange rate checker tool
    RateCheckDesc = RichToolDescription(
        description="Get current exchange rates for a base currency",
        use_when="When user wants to check current exchange rates without converting",
        side_effects="Returns current exchange rates from reliable source"
    )
    
    @mcp.tool(description=RateCheckDesc.model_dump_json())
    async def check_exchange_rates(
        base_currency: Annotated[str, Field(description="Base currency (ISO 4217, e.g., INR, USD)", pattern="^[A-Za-z]{3}$")] = "INR"
    ) -> List[TextContent]:
        """Get current exchange rates for a base currency"""
        try:
            rates = await CurrencyConverter.get_exchange_rates(base_currency)
            base = base_currency.upper()
            
            # Show a few common rates if present
            common = {k: rates[k] for k in ["USD", "EUR", "INR", "GBP", "JPY", "AED", "AUD"] if k in rates}
            
            formatted_response = f"""## Current Exchange Rates (Base: {base})

```json
{json.dumps({'base': base, 'rates': common}, indent=2)}
```
"""
            
            return [TextContent(type="text", text=formatted_response)]
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch rates: {e}"))
