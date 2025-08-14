#!/usr/bin/env python3
import json
from requirements_agent_simple import SimpleRequirementsAgent

def main():
    print("\n🤖 AI Agent Framework - Your First Working Agent!\n")
    print("=" * 50)
    
    # Test description
    description = """
    I need an e-commerce website where customers can:
    - Browse products by category
    - Add items to shopping cart
    - Make secure payments
    - Track their orders
    - Leave product reviews
    """
    
    print("📋 Project Description:")
    print("-" * 40)
    print(description)
    print("-" * 40)
    
    print("\n⚙️  Running Requirements Agent...")
    
    # Create and run agent
    agent = SimpleRequirementsAgent()
    result = agent.execute({"description": description})
    
    print(f"\n✅ SUCCESS! Generated {result['total_stories']} user stories")
    print(f"✅ Grouped into {result['total_epics']} epics\n")
    
    # Display results
    print("📝 Generated User Stories:")
    print("=" * 50)
    
    for story in result['user_stories'][:3]:  # Show first 3 stories
        print(f"\n📌 {story['title']}")
        print(f"   User: {story['user_type']}")
        print(f"   Want: {story['feature']}")
        print(f"   Value: {story['benefit']}")
        print(f"   Priority: {story['priority']} | Points: {story['story_points']}")
    
    print(f"\n... and {len(result['user_stories']) - 3} more stories")
    
    # Save results
    with open("generated_requirements.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("\n💾 Full results saved to: generated_requirements.json")
    print("\n🎉 Your AI agent is working! Next steps:")
    print("   1. Add real LLM integration (OpenAI/Claude)")
    print("   2. Build more agents (Design, Code, Test)")
    print("   3. Connect agents to work together")
    print("   4. Package and deploy your framework")
    
    return result

if __name__ == "__main__":
    result = main()