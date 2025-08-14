from typing import List, Dict, Any
import json

class SimpleRequirementsAgent:
    def __init__(self):
        self.story_count = 0
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if "description" not in task:
            return {"error": "No description provided"}
        
        description = task["description"]
        user_stories = self.generate_user_stories(description)
        epics = self.group_into_epics(user_stories)
        
        return {
            "user_stories": user_stories,
            "epics": epics,
            "total_stories": len(user_stories),
            "total_epics": len(epics)
        }
    
    def generate_user_stories(self, description: str) -> List[Dict]:
        stories = []
        desc_lower = description.lower()
        
        # Generate stories based on keywords in description
        if "task" in desc_lower or "todo" in desc_lower:
            stories.extend(self._generate_task_stories())
        
        if "user" in desc_lower or "account" in desc_lower or "login" in desc_lower:
            stories.extend(self._generate_user_stories())
        
        if "project" in desc_lower or "organize" in desc_lower:
            stories.extend(self._generate_project_stories())
        
        if "team" in desc_lower or "share" in desc_lower or "collaborate" in desc_lower:
            stories.extend(self._generate_collaboration_stories())
        
        # If no specific keywords, generate generic stories
        if not stories:
            stories.extend(self._generate_generic_stories(description))
        
        return stories
    
    def _generate_task_stories(self) -> List[Dict]:
        return [
            {
                "id": f"US-{self._next_id()}",
                "title": "Create Task",
                "user_type": "authenticated user",
                "feature": "create a new task with title and description",
                "benefit": "I can track my work items",
                "acceptance_criteria": [
                    "User can enter task title (required)",
                    "User can enter task description (optional)",
                    "Task is saved to database",
                    "User sees confirmation message"
                ],
                "priority": "High",
                "story_points": 5
            },
            {
                "id": f"US-{self._next_id()}",
                "title": "Edit Task",
                "user_type": "task owner",
                "feature": "edit an existing task",
                "benefit": "I can update task information",
                "acceptance_criteria": [
                    "User can modify task title and description",
                    "Changes are saved to database",
                    "Edit history is maintained"
                ],
                "priority": "High",
                "story_points": 3
            },
            {
                "id": f"US-{self._next_id()}",
                "title": "Delete Task",
                "user_type": "task owner",
                "feature": "delete a task",
                "benefit": "I can remove completed or unwanted tasks",
                "acceptance_criteria": [
                    "User sees confirmation dialog",
                    "Task is soft-deleted from database",
                    "Related data is handled appropriately"
                ],
                "priority": "Medium",
                "story_points": 2
            },
            {
                "id": f"US-{self._next_id()}",
                "title": "Set Due Date",
                "user_type": "task owner",
                "feature": "set and modify due dates on tasks",
                "benefit": "I can manage deadlines",
                "acceptance_criteria": [
                    "Date picker is available",
                    "Past dates show warning",
                    "Due date is displayed on task"
                ],
                "priority": "High",
                "story_points": 3
            }
        ]
    
    def _generate_user_stories(self) -> List[Dict]:
        return [
            {
                "id": f"US-{self._next_id()}",
                "title": "User Registration",
                "user_type": "new user",
                "feature": "register for an account",
                "benefit": "I can access the application",
                "acceptance_criteria": [
                    "Email validation is performed",
                    "Password strength requirements are enforced",
                    "Duplicate emails are prevented",
                    "Welcome email is sent"
                ],
                "priority": "High",
                "story_points": 5
            },
            {
                "id": f"US-{self._next_id()}",
                "title": "User Login",
                "user_type": "registered user",
                "feature": "log into my account",
                "benefit": "I can access my personal data",
                "acceptance_criteria": [
                    "Email and password fields are required",
                    "Invalid credentials show error message",
                    "Successful login creates session",
                    "Remember me option available"
                ],
                "priority": "High",
                "story_points": 3
            }
        ]
    
    def _generate_project_stories(self) -> List[Dict]:
        return [
            {
                "id": f"US-{self._next_id()}",
                "title": "Create Project",
                "user_type": "authenticated user",
                "feature": "create projects to organize tasks",
                "benefit": "I can group related tasks together",
                "acceptance_criteria": [
                    "Project name is required",
                    "Project description is optional",
                    "Color coding available",
                    "Project is saved to database"
                ],
                "priority": "Medium",
                "story_points": 5
            },
            {
                "id": f"US-{self._next_id()}",
                "title": "Assign Tasks to Project",
                "user_type": "project owner",
                "feature": "assign tasks to projects",
                "benefit": "I can organize my work",
                "acceptance_criteria": [
                    "Dropdown shows available projects",
                    "Tasks can be moved between projects",
                    "Unassigned tasks section exists"
                ],
                "priority": "Medium",
                "story_points": 3
            }
        ]
    
    def _generate_collaboration_stories(self) -> List[Dict]:
        return [
            {
                "id": f"US-{self._next_id()}",
                "title": "Share Task",
                "user_type": "task owner",
                "feature": "share tasks with team members",
                "benefit": "I can collaborate with others",
                "acceptance_criteria": [
                    "User can search for team members",
                    "Permission levels available (view/edit)",
                    "Notification sent to invited user",
                    "Shared tasks appear in recipient's list"
                ],
                "priority": "Low",
                "story_points": 8
            }
        ]
    
    def _generate_generic_stories(self, description: str) -> List[Dict]:
        return [
            {
                "id": f"US-{self._next_id()}",
                "title": "Core Feature",
                "user_type": "user",
                "feature": "use the main functionality",
                "benefit": "I can achieve my goals",
                "acceptance_criteria": [
                    "Feature works as described",
                    "User interface is intuitive",
                    "Performance is acceptable"
                ],
                "priority": "High",
                "story_points": 5
            }
        ]
    
    def group_into_epics(self, stories: List[Dict]) -> List[Dict]:
        epics = {}
        
        for story in stories:
            epic_name = self._determine_epic(story)
            
            if epic_name not in epics:
                epics[epic_name] = {
                    "name": epic_name,
                    "stories": [],
                    "total_points": 0
                }
            
            epics[epic_name]["stories"].append(story["id"])
            epics[epic_name]["total_points"] += story.get("story_points", 0)
        
        return list(epics.values())
    
    def _determine_epic(self, story: Dict) -> str:
        title_lower = story.get("title", "").lower()
        
        if any(word in title_lower for word in ["user", "login", "registration", "account"]):
            return "User Management"
        elif any(word in title_lower for word in ["task", "todo", "item"]):
            return "Task Management"
        elif any(word in title_lower for word in ["project", "organize", "group"]):
            return "Project Organization"
        elif any(word in title_lower for word in ["share", "team", "collaborate"]):
            return "Collaboration"
        else:
            return "Core Features"
    
    def _next_id(self) -> str:
        self.story_count += 1
        return f"{self.story_count:03d}"