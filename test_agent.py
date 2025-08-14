#!/usr/bin/env python3
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from requirements_agent import MockRequirementsAgent, RequirementsAgent
import os
from dotenv import load_dotenv

load_dotenv()

console = Console()

def test_requirements_agent():
    console.print("\n[bold cyan]🚀 Testing Requirements Agent[/bold cyan]\n")
    
    llm_provider = os.getenv("LLM_PROVIDER", "mock")
    
    if llm_provider == "mock":
        console.print("[yellow]Using Mock Agent (no API key needed)[/yellow]\n")
        agent = MockRequirementsAgent()
    else:
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")
        if not api_key:
            console.print(f"[red]Error: {llm_provider.upper()}_API_KEY not found in .env file[/red]")
            console.print("[yellow]Using Mock Agent instead[/yellow]\n")
            agent = MockRequirementsAgent()
        else:
            console.print(f"[green]Using {llm_provider} LLM[/green]\n")
            agent = RequirementsAgent(llm_provider)
    
    project_description = """
    I need a task management application where users can:
    - Create, edit, and delete tasks
    - Set due dates and priorities
    - Organize tasks into projects
    - Share tasks with team members
    - Track task completion status
    """
    
    console.print(Panel(project_description, title="Project Description", border_style="blue"))
    
    console.print("\n[cyan]Generating user stories...[/cyan]\n")
    
    task = {"description": project_description}
    result = agent.execute(task)
    
    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return
    
    console.print(f"[green]✓ Generated {result['total_stories']} user stories[/green]")
    console.print(f"[green]✓ Grouped into {result['total_epics']} epics[/green]\n")
    
    table = Table(title="User Stories", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=8)
    table.add_column("Title", style="white", width=20)
    table.add_column("User Type", style="yellow", width=15)
    table.add_column("Priority", style="green", width=10)
    table.add_column("Points", style="blue", width=8)
    
    for story in result['user_stories']:
        table.add_row(
            story.get('id', 'N/A'),
            story.get('title', 'N/A')[:20],
            story.get('user_type', 'N/A')[:15],
            story.get('priority', 'N/A'),
            str(story.get('story_points', 0))
        )
    
    console.print(table)
    
    console.print("\n[bold cyan]Epics:[/bold cyan]")
    for epic in result['epics']:
        console.print(f"  • [yellow]{epic['name']}[/yellow]: {len(epic['stories'])} stories, {epic['total_points']} points")
    
    with open("generated_requirements.json", "w") as f:
        json.dump(result, f, indent=2)
    
    console.print("\n[green]✓ Results saved to generated_requirements.json[/green]")

if __name__ == "__main__":
    test_requirements_agent()