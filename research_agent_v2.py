"""Regime-Aware Strategy Research Agent"""
import os
import anthropic
import json
from datetime import datetime
from dotenv import load_dotenv
from market_regime import MarketRegimeDetector

def get_api_key():
    """Get Claude API key from environment"""
    load_dotenv()
    return os.getenv("CLAUDE_API_KEY")

class RegimeAwareResearchAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=get_api_key())
        self.model = "claude-sonnet-4-20250514"
        self.regime_detector = MarketRegimeDetector()
    
    def research_strategies(self, regime):
        """Research strategies suited for current market regime"""
        print("\n🔬 Starting regime-aware research cycle...")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Build regime-specific prompt
        prompt = f"""You are a quantitative trading researcher specializing in cryptocurrency technical analysis.

CURRENT MARKET REGIME ANALYSIS:
- Symbol: {regime['symbol']}
- Current Price: ${regime['current_price']:,.2f}
- Regime Type: {regime['regime_type']}
- Trend Direction: {regime['trend_direction']}
- Trend Strength: {regime['trend_strength']}
- Volatility: {regime['volatility_pct']:.2f}%
- ADX: {regime['adx']:.1f}
- 30-Day Change: {regime['price_change_30d']:+.2f}%

RECOMMENDED STRATEGY TYPES FOR THIS REGIME:
{', '.join(regime['recommended_strategy_types'])}

CURRENT STRATEGIES IN USE:
- RSI (14), MACD (12/26/9), Moving Averages (20/50), Bollinger Bands (20, 2σ)

TASK:
Research and identify 2-3 NEW technical analysis strategies specifically suited for the CURRENT {regime['regime_type']} market regime.

CRITICAL: Only suggest strategies from these types: {', '.join(regime['recommended_strategy_types'])}

For each strategy, provide:
1. Name of the strategy
2. Indicator(s) involved (specific parameters)
3. Entry conditions (when to buy)
4. Exit conditions (when to sell)
5. Why this strategy works in {regime['regime_type']} markets
6. Estimated win rate in {regime['regime_type']} conditions
7. Risk level (Low/Medium/High)

Format as valid JSON:
{{
  "regime_context": {{
    "regime_type": "{regime['regime_type']}",
    "analyzed_at": "{datetime.now().isoformat()}"
  }},
  "strategies": [
    {{
      "name": "Strategy Name",
      "suitable_for_regime": "{regime['regime_type']}",
      "indicators": ["Indicator 1 (params)", "Indicator 2 (params)"],
      "entry_conditions": "Specific buy conditions",
      "exit_conditions": "Specific sell conditions",
      "rationale": "Why this works in {regime['regime_type']} markets",
      "estimated_win_rate": 0.60,
      "risk_level": "Medium"
    }}
  ]
}}

Respond ONLY with valid JSON, no extra text."""

        print(f"💭 Querying Claude for {regime['regime_type']}-specific strategies...")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        try:
            # Clean JSON
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
    
    def display_findings(self, findings, regime):
        """Display findings with regime context"""
        print("\n" + "="*80)
        print(f"📋 REGIME-AWARE RESEARCH FINDINGS")
        print("="*80)
        print(f"Market Regime: {regime['regime_type']}")
        print(f"Optimal Strategy Types: {', '.join(regime['recommended_strategy_types'])}")
        print("="*80 + "\n")
        
        for i, strategy in enumerate(findings.get('strategies', []), 1):
            print(f"Strategy {i}: {strategy['name']}")
            print(f"  Suitable For: {strategy.get('suitable_for_regime', 'N/A')}")
            print(f"  Risk Level: {strategy['risk_level']}")
            print(f"  Est. Win Rate: {strategy['estimated_win_rate']*100:.1f}%")
            print(f"  Indicators: {', '.join(strategy['indicators'])}")
            print(f"  Entry: {strategy['entry_conditions']}")
            print(f"  Exit: {strategy['exit_conditions']}")
            print(f"  Rationale: {strategy['rationale']}")
            print()
    
    def save_findings(self, findings, regime):
        """Save research findings with regime context"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"research_{regime['regime_type']}_{timestamp}.json"
        
        output = {
            'regime': regime,
            'findings': findings,
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Saved findings to: {filename}")
        return filename

def main():
    print("="*80)
    print("🤖 REGIME-AWARE STRATEGY RESEARCH AGENT v2.0")
    print("="*80 + "\n")
    
    agent = RegimeAwareResearchAgent()
    
    # Step 1: Detect current market regime
    regime = agent.regime_detector.detect_regime('BTC-USD')
    
    if not regime:
        print("❌ Could not detect market regime")
        return
    
    agent.regime_detector.display_regime(regime)
    
    # Step 2: Research strategies for this regime
    findings = agent.research_strategies(regime)
    
    if findings:
        agent.display_findings(findings, regime)
        agent.save_findings(findings, regime)
        
        print("\n✅ Regime-aware research cycle complete!")
        print(f"📁 Strategies optimized for {regime['regime_type']} market")
    else:
        print("\n❌ Research failed")

if __name__ == "__main__":
    main()