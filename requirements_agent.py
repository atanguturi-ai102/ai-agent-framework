import os
import json
from typing import List, Dict, Any
from base_agent import BaseAgent, AgentRole
from dotenv import load_dotenv
import openai
from anthropic import Anthropic

load_dotenv()

class RequirementsAgent(BaseAgent):
    def __init__(self, llm_provider: str = "openai"):
        super().__init__("RequirementsAgent", AgentRole.REQUIREMENTS)
        self.llm_provider = llm_provider
        
        if llm_provider == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif llm_provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        self.story_count = 0
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_input(task):
            return {"error": "Invalid input"}
        
        project_description = task["description"]
        
        self.log_action("Generating user stories", project_description)
        
        user_stories = self.generate_user_stories(project_description)
        epics = self.group_into_epics(user_stories)
        
        result = {
            "user_stories": user_stories,
            "epics": epics,
            "total_stories": len(user_stories),
            "total_epics": len(epics)
        }
        
        self.log_action("Generated requirements", result)
        return result
    
    def generate_user_stories(self, description: str) -> List[Dict]:
        prompt = f"""
        Given this project description: {description}
        
        Generate 5-10 user stories in JSON format. Each story should have:
        - id: unique identifier
        - title: short descriptive title
        - user_type: who will use this feature
        - feature: what they want to do
        - benefit: why they want it
        - acceptance_criteria: list of testable criteria
        - priority: High/Medium/Low
        - story_points: 1, 2, 3, 5, 8, or 13
        
        Return ONLY valid JSON array of stories.
        """
        
        if self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a requirements analyst. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            stories_text = response.choices[0].message.content
            
        elif self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            stories_text = response.content[0].text
        
        try:
            stories = json.loads(stories_text)
            if not isinstance(stories, list):
                stories = [stories]
        except json.JSONDecodeError:
            stories = self.parse_text_to_stories(stories_text)
        
        for story in stories:
            self.story_count += 1
            if 'id' not in story:
                story['id'] = f"US-{self.story_count:03d}"
        
        return stories
    
    def parse_text_to_stories(self, text: str) -> List[Dict]:
        stories = []
        self.story_count += 1
        
        story = {
            "id": f"US-{self.story_count:03d}",
            "title": "Parsed story",
            "user_type": "user",
            "feature": text[:100],
            "benefit": "Provides value",
            "acceptance_criteria": ["Feature works as expected"],
            "priority": "Medium",
            "story_points": 5
        }
        stories.append(story)
        
        return stories
    
    def group_into_epics(self, stories: List[Dict]) -> List[Dict]:
        epics = {}
        
        for story in stories:
            epic_name = self.determine_epic_category(story)
            
            if epic_name not in epics:
                epics[epic_name] = {
                    "name": epic_name,
                    "stories": [],
                    "total_points": 0
                }
            
            epics[epic_name]["stories"].append(story["id"])
            epics[epic_name]["total_points"] += story.get("story_points", 0)
        
        return list(epics.values())
    
    def determine_epic_category(self, story: Dict) -> str:
        feature = story.get("feature", "").lower()
        
        if any(word in feature for word in ["login", "auth", "user", "account", "profile"]):
            return "User Management"
        elif any(word in feature for word in ["data", "database", "storage", "crud"]):
            return "Data Management"
        elif any(word in feature for word in ["ui", "interface", "display", "view", "page"]):
            return "User Interface"
        elif any(word in feature for word in ["api", "integration", "service"]):
            return "Integration"
        elif any(word in feature for word in ["report", "analytics", "dashboard"]):
            return "Analytics"
        else:
            return "Core Features"
    
    def get_required_fields(self) -> List[str]:
        return ["description"]

class MockRequirementsAgent(RequirementsAgent):
    def __init__(self):
        BaseAgent.__init__(self, "MockRequirementsAgent", AgentRole.REQUIREMENTS)
        self.story_count = 0
    
    def generate_user_stories(self, description: str) -> List[Dict]:
        mock_stories = [
            {
                "id": "US-001",
                "title": "User Registration",
                "user_type": "new user",
                "feature": "register for an account",
                "benefit": "access the application features",
                "acceptance_criteria": [
                    "User can enter email and password",
                    "Email validation is performed",
                    "Password meets security requirements",
                    "Confirmation email is sent"
                ],
                "priority": "High",
                "story_points": 5
            },
            {
                "id": "US-002",
                "title": "User Login",
                "user_type": "registered user",
                "feature": "log into the application",
                "benefit": "access personalized content",
                "acceptance_criteria": [
                    "User can enter credentials",
                    "Invalid credentials show error",
                    "Successful login redirects to dashboard",
                    "Session is maintained"
                ],
                "priority": "High",
                "story_points": 3
            },
            {
                "id": "US-003",
                "title": "Create Item",
                "user_type": "authenticated user",
                "feature": "create a new item",
                "benefit": "add content to the system",
                "acceptance_criteria": [
                    "Form with required fields is displayed",
                    "Validation is performed on submit",
                    "Item is saved to database",
                    "Success message is shown"
                ],
                "priority": "High",
                "story_points": 5
            }
        ]
        
        return mock_stories