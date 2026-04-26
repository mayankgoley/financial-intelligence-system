import yfinance as yf
from datetime import datetime


def get_market_data(ticker: str) -> dict:
    # yfinance sometimes returns info={}, no caching either - todo: redis it
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "current_price": info.get("currentPrice", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "revenue": info.get("totalRevenue", "N/A"),
            "revenue_growth": info.get("revenueGrowth", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
            "target_price": info.get("targetMeanPrice", "N/A"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def format_market_context(ticker: str) -> str:
    d = get_market_data(ticker)
    if "error" in d:
        return f"[Market data unavailable for {ticker}]"

    def fmt(v, pre="", suf=""):
        if v == "N/A" or v is None: return "N/A"
        if isinstance(v, (int,float)):
            if abs(v) >= 1e9: return f"{pre}{v/1e9:.1f}B{suf}"
            if abs(v) >= 1e6: return f"{pre}{v/1e6:.1f}M{suf}"
            return f"{pre}{v:.2f}{suf}"
        return f"{pre}{v}{suf}"

    return f"""=== LIVE MARKET DATA: {d['ticker']} ===
Company: {d['company_name']}
Price: {fmt(d['current_price'],'$')} | 52W: {fmt(d['52w_low'],'$')}-{fmt(d['52w_high'],'$')}
Market Cap: {fmt(d['market_cap'],'$')}
P/E: {fmt(d['pe_ratio'])} | Forward P/E: {fmt(d['forward_pe'])}
Revenue: {fmt(d['revenue'],'$')} | Growth: {fmt(d['revenue_growth'])}
EPS: {fmt(d['eps'],'$')} | Margin: {fmt(d['profit_margin'])}
Analyst Target: {fmt(d['target_price'],'$')} | Rating: {d['recommendation']}
As of: {d['timestamp']}
=== END MARKET DATA ==="""


if __name__ == "__main__":
    print(format_market_context("NVDA"))
