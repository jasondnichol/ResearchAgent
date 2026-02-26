"""Strategy Research Agent - Finds and analyzes trading strategies"""
import os
import anthropic
import json
from datetime import datetime
import subprocess
from dotenv import load_dotenv

def get_api_key():
    """Get Claude API key from environment"""
    load_dotenv()
    return os.getenv("CLAUDE_API_KEY")

class ResearchAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=get_api_key())
        self.model = "claude-sonnet-4-20250514"
    
    def research_strategies(self):
        """Research new trading strategies"""
        print("🔬 Starting research cycle...")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        prompt = """You are a quantitative trading researcher specializing in cryptocurrency technical analysis.

Current Context:
- Asset: Bitcoin (BTC), Ethereum (ETH), Curve (CRV)
- Current strategies in use: RSI (14), MACD (12/26/9), Moving Averages (20/50), Bollinger Bands (20, 2σ)
- Trading timeframe: 5 minutes to 1 hour
- Goal: Find NEW indicator combinations or improvements to existing strategies

Task:
Research and identify 2-3 promising technical analysis strategies for crypto trading that we're NOT currently using.

For each strategy, provide:
1. Name of the strategy
2. Indicator(s) involved (be specific with parameters)
3. Entry conditions (when to buy)
4. Exit conditions (when to sell)
5. Why this might work for crypto (brief rationale)
6. Estimated win rate (based on similar strategies)
7. Risk level (Low/Medium/High)

Format your response as valid JSON with this structure:
{
  "strategies": [
    {
      "name": "Strategy Name",
      "indicators": ["Indicator 1 (params)", "Indicator 2 (params)"],
      "entry_conditions": "Specific buy conditions",
      "exit_conditions": "Specific sell conditions",
      "rationale": "Why this works",
      "estimated_win_rate": 0.60,
      "risk_level": "Medium"
    }
  ]
}

Respond ONLY with valid JSON, no extra text."""

        print("💭 Querying Claude for strategy research...")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract response
        response_text = response.content[0].text
        
        # Parse JSON
        try:
            # Remove markdown code blocks if present
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
            print(f"Response: {response_text[:200]}...")
            return None
    
    def save_findings(self, findings):
        """Save research findings to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"research_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(findings, f, indent=2)
        
        print(f"💾 Saved findings to: {filename}")
        return filename
    
    def display_findings(self, findings):
        """Display findings in readable format"""
        print("\n" + "="*80)
        print("📋 RESEARCH FINDINGS")
        print("="*80 + "\n")
        
        for i, strategy in enumerate(findings.get('strategies', []), 1):
            print(f"Strategy {i}: {strategy['name']}")
            print(f"  Risk Level: {strategy['risk_level']}")
            print(f"  Est. Win Rate: {strategy['estimated_win_rate']*100:.1f}%")
            print(f"  Indicators: {', '.join(strategy['indicators'])}")
            print(f"  Entry: {strategy['entry_conditions']}")
            print(f"  Exit: {strategy['exit_conditions']}")
            print(f"  Rationale: {strategy['rationale']}")
            print()

def main():
    print("="*80)
    print("🤖 STRATEGY RESEARCH AGENT")
    print("="*80 + "\n")
    
    agent = ResearchAgent()
    
    # Run research
    findings = agent.research_strategies()
    
    if findings:
        agent.display_findings(findings)
        agent.save_findings(findings)
        
        print("\n✅ Research cycle complete!")
        print("📁 Check the research_*.json file for full details")
    else:
        print("\n❌ Research failed")

if __name__ == "__main__":
    main()