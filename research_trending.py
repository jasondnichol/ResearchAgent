"""Research strategies specifically for TRENDING markets"""
import os
import anthropic
import json
from datetime import datetime
from dotenv import load_dotenv

def get_api_key():
    load_dotenv()
    return os.getenv("CLAUDE_API_KEY")

class TrendingResearchAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=get_api_key())
        self.model = "claude-sonnet-4-20250514"
    
    def research_trending_strategies(self):
        """Research strategies for trending markets"""
        print("🔬 Researching TRENDING market strategies...")
        
        prompt = """You are a quantitative trading researcher specializing in trend-following strategies for cryptocurrency.

MARKET CONTEXT:
- Regime Type: TRENDING
- Characteristics: Strong directional moves, ADX > 25, clear uptrends or downtrends
- Historical data shows: 44% of time markets are in trending regime
- Goal: Capture momentum and ride trends until exhaustion

CURRENT LIBRARY:
- RANGING strategies: Williams %R Mean Reversion (55% win rate)
- TRENDING strategies: NONE (need to build this)

TASK:
Research 3 proven trend-following strategies specifically for cryptocurrency trending markets.

Requirements:
- Must work in BOTH uptrends and downtrends
- Should identify trend early and ride it
- Clear exit when trend exhausts
- NOT mean reversion (we have that for ranging)

For each strategy provide:
1. Name
2. Indicators with specific parameters
3. Entry conditions (trend confirmation)
4. Exit conditions (trend exhaustion detection)
5. Why this works in trending crypto markets
6. Estimated win rate in trending conditions
7. Risk level

Format as valid JSON:
{
  "regime": "TRENDING",
  "strategies": [
    {
      "name": "Strategy Name",
      "suitable_for_regime": "TRENDING",
      "trend_direction": "Both uptrend and downtrend",
      "indicators": ["Indicator 1 (params)", "Indicator 2 (params)"],
      "entry_conditions": "Specific entry when trend confirmed",
      "exit_conditions": "Exit when trend shows exhaustion",
      "rationale": "Why this works in trending markets",
      "estimated_win_rate": 0.65,
      "risk_level": "Medium"
    }
  ]
}

Respond ONLY with valid JSON."""

        print("💭 Querying Claude for TRENDING strategies...")
        
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
            print(f"📊 Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
            print(f"💰 Cost: ${(response.usage.input_tokens * 0.000015 + response.usage.output_tokens * 0.000075):.4f}")
            
            return findings
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            return None
    
    def display_findings(self, findings):
        """Display trending strategy findings"""
        print("\n" + "="*80)
        print("📋 TRENDING MARKET STRATEGIES")
        print("="*80 + "\n")
        
        for i, strategy in enumerate(findings.get('strategies', []), 1):
            print(f"Strategy {i}: {strategy['name']}")
            print(f"  Trend Direction: {strategy.get('trend_direction', 'N/A')}")
            print(f"  Risk Level: {strategy['risk_level']}")
            print(f"  Est. Win Rate: {strategy['estimated_win_rate']*100:.1f}%")
            print(f"  Indicators: {', '.join(strategy['indicators'])}")
            print(f"  Entry: {strategy['entry_conditions']}")
            print(f"  Exit: {strategy['exit_conditions']}")
            print(f"  Rationale: {strategy['rationale']}")
            print()
    
    def save_findings(self, findings):
        """Save findings"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"research_TRENDING_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(findings, f, indent=2)
        
        print(f"💾 Saved to: {filename}")
        return filename

def main():
    print("="*80)
    print("🤖 TRENDING MARKET RESEARCH AGENT")
    print("="*80 + "\n")
    
    agent = TrendingResearchAgent()
    
    findings = agent.research_trending_strategies()
    
    if findings:
        agent.display_findings(findings)
        agent.save_findings(findings)
        
        print("\n✅ Research complete!")
        print("📁 Next: Backtest these strategies on TRENDING historical periods")
    else:
        print("\n❌ Research failed")

if __name__ == "__main__":
    main()