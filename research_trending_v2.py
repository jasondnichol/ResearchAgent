"""Research BETTER trending strategies with failure feedback"""
import os
import anthropic
import json
from datetime import datetime
from dotenv import load_dotenv

def get_api_key():
    load_dotenv()
    return os.getenv("CLAUDE_API_KEY")

class ImprovedTrendingResearch:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=get_api_key())
        self.model = "claude-sonnet-4-20250514"
    
    def research_with_failure_feedback(self):
        """Research trending strategies with feedback on what failed"""
        
        prompt = """You are a quantitative trading researcher. You previously suggested EMA Crossover for trending markets, but it FAILED badly in backtesting.

FAILURE ANALYSIS:
Strategy: EMA Crossover (12/26)
Expected Win Rate: 68%
Actual Win Rate: 23.1% ❌
Profit Factor: 0.36 (terrible)
Total P&L: -7.57%

WHY IT FAILED:
- EMA crossovers are LAGGING indicators
- By the time EMAs cross, trend is already established
- Entry is too late (missed most of the move)
- Exits too late (gives back profits)
- Gets whipsawed in choppy trends

MARKET CONTEXT (4-year Bitcoin cycle):
- TRENDING periods: 48.7% of time (613 days)
- VOLATILE periods: 31.5% (397 days)  
- RANGING periods: 19.8% (250 days)
- Bitcoin has strong parabolic moves followed by sharp corrections
- Trends can last weeks/months but with frequent pullbacks

CURRENT WORKING STRATEGY:
- Williams %R for RANGING: 55% win rate, 2.09 profit factor ✅

TASK:
Find 3 trend-following strategies that:
1. Enter EARLY in trends (not lagging like EMA crossover)
2. Handle Bitcoin's parabolic moves well
3. Exit before trend exhaustion (not after like EMA)
4. Can handle pullbacks without exiting prematurely
5. Work in BOTH uptrends and downtrends

AVOID:
- Simple moving average crossovers (too slow)
- EMA crossovers (already failed)
- Strategies that wait for "confirmation" (too late)

PREFER:
- Momentum-based entries (catch moves early)
- Breakout strategies (enter as trend starts)
- Volatility-based signals (capture beginning of moves)
- Multiple timeframe confirmation (avoid false signals)

Format as JSON:
{
  "learnings_from_failure": "Why EMA failed and what to do instead",
  "strategies": [
    {
      "name": "Strategy Name",
      "entry_type": "Early momentum/Breakout/Volatility",
      "indicators": ["Specific indicators with params"],
      "entry_conditions": "When to enter (EARLY in trend)",
      "exit_conditions": "When to exit (BEFORE exhaustion)",
      "why_better_than_ema": "Specific advantages",
      "estimated_win_rate": 0.60,
      "risk_level": "Medium"
    }
  ]
}

Respond ONLY with valid JSON."""

        print("🔬 Researching IMPROVED trending strategies...")
        print("💭 Giving Claude feedback on EMA failure...\n")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            findings = json.loads(response_text)
            
            print(f"✅ Research complete!")
            print(f"📊 Cost: ${(response.usage.input_tokens * 0.000015 + response.usage.output_tokens * 0.000075):.4f}\n")
            
            return findings
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse: {e}")
            return None
    
    def display_findings(self, findings):
        """Display improved strategies"""
        print("="*80)
        print("🧠 LEARNINGS FROM EMA FAILURE")
        print("="*80)
        print(findings.get('learnings_from_failure', 'N/A'))
        
        print("\n" + "="*80)
        print("📋 IMPROVED TRENDING STRATEGIES")
        print("="*80 + "\n")
        
        for i, strategy in enumerate(findings.get('strategies', []), 1):
            print(f"Strategy {i}: {strategy['name']}")
            print(f"  Entry Type: {strategy.get('entry_type', 'N/A')}")
            print(f"  Risk Level: {strategy['risk_level']}")
            print(f"  Est. Win Rate: {strategy['estimated_win_rate']*100:.1f}%")
            print(f"  Indicators: {', '.join(strategy['indicators'])}")
            print(f"  Entry: {strategy['entry_conditions']}")
            print(f"  Exit: {strategy['exit_conditions']}")
            print(f"  Why Better than EMA: {strategy.get('why_better_than_ema', 'N/A')}")
            print()
    
    def save_findings(self, findings):
        """Save findings"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"research_TRENDING_improved_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(findings, f, indent=2)
        
        print(f"💾 Saved to: {filename}")
        return filename

def main():
    print("="*80)
    print("🤖 IMPROVED TRENDING RESEARCH (Learning from Failure)")
    print("="*80 + "\n")
    
    researcher = ImprovedTrendingResearch()
    
    findings = researcher.research_with_failure_feedback()
    
    if findings:
        researcher.display_findings(findings)
        researcher.save_findings(findings)
        
        print("\n✅ Improved research complete!")
        print("📁 Next: Backtest these on 4-year TRENDING periods")
    else:
        print("\n❌ Research failed")

if __name__ == "__main__":
    main()