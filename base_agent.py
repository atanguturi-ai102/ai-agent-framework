from abc import ABC, abstractmethod
from typing import Dict, List, Any
from enum import Enum
import logging
from datetime import datetime

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    REQUIREMENTS = "requirements"
    DESIGN = "design"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEVOPS = "devops"

class BaseAgent(ABC):
    def __init__(self, name: str, role: AgentRole):
        self.name = name
        self.role = role
        self.logger = logging.getLogger(name)
        self.context = {}
        self.history = []
        
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    def log_action(self, action: str, details: Any = None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        self.history.append(log_entry)
        self.logger.info(f"{action}: {details}")
    
    def validate_input(self, task: Dict[str, Any]) -> bool:
        required_fields = self.get_required_fields()
        missing = [f for f in required_fields if f not in task]
        
        if missing:
            self.logger.error(f"Missing required fields: {missing}")
            return False
        return True
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        pass