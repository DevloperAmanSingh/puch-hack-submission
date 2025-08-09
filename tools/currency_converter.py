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
    """Simple currency converter using free exchange rate API"""
    
    BASE_URL = "https://api.exchangerate-api.com/v4/latest/INR"
    
    @classmethod
    async def get_exchange_rates(cls) -> dict:
        """Fetch current exchange rates for INR"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(cls.BASE_URL, timeout=10)
                response.raise_for_status()
                data = response.json()
                return data.get("rates", {})
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch exchange rates: {e}"))
    
    @classmethod
    async def convert_inr(cls, amount: float, target_currency: str) -> dict:
        """Convert INR amount to target currency"""
        if amount <= 0:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Amount must be greater than 0"))
        
        rates = await cls.get_exchange_rates()
        
        if target_currency not in rates:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Currency {target_currency} not supported"))
        
        rate = rates[target_currency]
        converted_amount = amount * rate
        
        return {
            "from_currency": "INR",
            "to_currency": target_currency,
            "amount": amount,
            "converted_amount": round(converted_amount, 2),
            "exchange_rate": rate,
            "last_updated": "Current rates from exchangerate-api.com"
        }


def register(mcp: FastMCP) -> None:
    """Register currency converter tools with MCP server"""
    
    # Single currency conversion tool
    SingleConversionDesc = RichToolDescription(
        description="Convert INR to EUR or USD using real-time exchange rates",
        use_when="When user wants to convert Indian Rupees to Euro or US Dollar",
        side_effects="Returns converted amount with current exchange rate"
    )
    
    @mcp.tool(description=SingleConversionDesc.model_dump_json())
    async def convert_currency(
        amount: Annotated[float, Field(description="Amount in INR to convert", ge=0.01)],
        target_currency: Annotated[str, Field(description="Target currency (EUR or USD)", pattern="^(EUR|USD)$")] = "USD"
    ) -> List[TextContent]:
        """Convert INR to EUR or USD"""
        try:
            result = await CurrencyConverter.convert_inr(amount, target_currency)
            
            formatted_response = f"""## Currency Conversion

**From:** {result['amount']} INR
**To:** {result['converted_amount']} {result['to_currency']}
**Exchange Rate:** 1 INR = {result['exchange_rate']} {result['to_currency']}

**Conversion Details:**
- Amount: ₹{result['amount']:,.2f}
- Converted: {result['converted_amount']:,.2f} {result['to_currency']}
- Rate Source: {result['last_updated']}

---
**JSON Data:**
```json
{json.dumps(result, indent=2)}
```"""
            
            return [TextContent(type="text", text=formatted_response)]
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Conversion failed: {e}"))
    
    # Multi-currency conversion tool
    MultiConversionDesc = RichToolDescription(
        description="Convert INR to both EUR and USD simultaneously",
        use_when="When user wants to see INR converted to both Euro and US Dollar at once",
        side_effects="Returns conversions to both currencies with exchange rates"
    )
    
    @mcp.tool(description=MultiConversionDesc.model_dump_json())
    async def convert_inr_to_both(
        amount: Annotated[float, Field(description="Amount in INR to convert to both EUR and USD", ge=0.01)]
    ) -> List[TextContent]:
        """Convert INR to both EUR and USD"""
        try:
            usd_result = await CurrencyConverter.convert_inr(amount, "USD")
            eur_result = await CurrencyConverter.convert_inr(amount, "EUR")
            
            formatted_response = f"""## INR to EUR & USD Conversion

**Original Amount:** ₹{amount:,.2f} INR

**USD Conversion:**
- Amount: {usd_result['converted_amount']:,.2f} USD
- Rate: 1 INR = {usd_result['exchange_rate']} USD

**EUR Conversion:**
- Amount: {eur_result['converted_amount']:,.2f} EUR
- Rate: 1 INR = {eur_result['exchange_rate']} EUR

**Summary:**
- ₹{amount:,.2f} INR = ${usd_result['converted_amount']:,.2f} USD
- ₹{amount:,.2f} INR = €{eur_result['converted_amount']:,.2f} EUR

---
**Exchange Rates:**
- INR to USD: {usd_result['exchange_rate']}
- INR to EUR: {eur_result['exchange_rate']}
- Source: {usd_result['last_updated']}
"""
            
            return [TextContent(type="text", text=formatted_response)]
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Multi-conversion failed: {e}"))
    
    # Exchange rate checker tool
    RateCheckDesc = RichToolDescription(
        description="Get current exchange rates for INR to EUR and USD",
        use_when="When user wants to check current exchange rates without converting",
        side_effects="Returns current exchange rates from reliable source"
    )
    
    @mcp.tool(description=RateCheckDesc.model_dump_json())
    async def check_exchange_rates() -> List[TextContent]:
        """Get current exchange rates for INR"""
        try:
            rates = await CurrencyConverter.get_exchange_rates()
            
            usd_rate = rates.get("USD", "N/A")
            eur_rate = rates.get("EUR", "N/A")
            
            formatted_response = f"""## Current Exchange Rates (INR)

**USD Rate:** 1 INR = {usd_rate} USD
**EUR Rate:** 1 INR = {eur_rate} EUR

**Quick Conversions:**
- ₹100 INR = ${100 * usd_rate:.2f} USD
- ₹100 INR = €{100 * eur_rate:.2f} EUR
- ₹1,000 INR = ${1000 * usd_rate:.2f} USD
- ₹1,000 INR = €{1000 * eur_rate:.2f} EUR

**Source:** exchangerate-api.com
**Last Updated:** Current rates
"""
            
            return [TextContent(type="text", text=formatted_response)]
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch rates: {e}"))
