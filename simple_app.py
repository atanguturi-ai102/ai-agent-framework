#!/usr/bin/env python3
import json
import sys
from requirements_agent_simple import SimpleRequirementsAgent

def main():
    print("\n🤖 AI Agent Framework - Requirements Generator\n")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        description = """
        I need a task management application where users can:
        - Create, edit, and delete tasks
        - Set due dates and priorities
        - Organize tasks into projects
        """
    else:
        print("\nDescribe your project (press Enter twice when done):\n")
        lines = []
        empty_count = 0
        
        while empty_count < 2:
            line = input()
            if line == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)
        
        description = '\n'.join(lines)
    
    if not description.strip():
        print("❌ No description provided")
        return
    
    print("\n📋 Project Description:")
    print("-" * 40)
    print(description)
    print("-" * 40)
    
    print("\n⚙️  Generating user stories...")
    
    agent = SimpleRequirementsAgent()
    result = agent.execute({"description": description})
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n✅ Generated {result['total_stories']} user stories")
    print(f"✅ Grouped into {result['total_epics']} epics\n")
    
    print("📝 User Stories:")
    print("=" * 50)
    
    for i, story in enumerate(result['user_stories'], 1):
        print(f"\n{i}. {story['title']}")
        print(f"   As a {story['user_type']}")
        print(f"   I want to {story['feature']}")
        print(f"   So that {story['benefit']}")
        print(f"   Priority: {story['priority']} | Points: {story['story_points']}")
        print(f"   Acceptance Criteria:")
        for criterion in story['acceptance_criteria']:
            print(f"      - {criterion}")
    
    print("\n📊 Epics:")
    print("-" * 40)
    for epic in result['epics']:
        print(f"  • {epic['name']}: {len(epic['stories'])} stories, {epic['total_points']} points")
    
    save = input("\n💾 Save results to file? (y/n): ")
    if save.lower() == 'y':
        filename = input("Filename (default: requirements.json): ").strip()
        if not filename:
            filename = "requirements.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Saved to {filename}")
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()