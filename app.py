#!/usr/bin/env python3
import click
import json
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from requirements_agent import RequirementsAgent, MockRequirementsAgent
import os
from dotenv import load_dotenv

load_dotenv()
console = Console()

@click.group()
def cli():
    """AI Agent Framework - Build applications with AI agents"""
    pass

@cli.command()
@click.option('--description', '-d', help='Project description')
@click.option('--file', '-f', help='Read description from file')
@click.option('--mock', is_flag=True, help='Use mock agent (no API needed)')
def requirements(description, file, mock):
    """Generate user stories from project description"""
    
    console.print("\n[bold cyan]📋 Requirements Agent[/bold cyan]\n")
    
    if file:
        with open(file, 'r') as f:
            description = f.read()
    elif not description:
        console.print("[yellow]Enter your project description (press Ctrl+D when done):[/yellow]")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            description = '\n'.join(lines)
    
    if not description:
        console.print("[red]Error: No description provided[/red]")
        return
    
    console.print(Panel(description, title="Project Description", border_style="blue"))
    
    if mock:
        agent = MockRequirementsAgent()
        console.print("[yellow]Using Mock Agent[/yellow]\n")
    else:
        llm_provider = os.getenv("LLM_PROVIDER", "mock")
        if llm_provider == "mock" or not os.getenv(f"{llm_provider.upper()}_API_KEY"):
            agent = MockRequirementsAgent()
            console.print("[yellow]Using Mock Agent (no API key found)[/yellow]\n")
        else:
            agent = RequirementsAgent(llm_provider)
            console.print(f"[green]Using {llm_provider} LLM[/green]\n")
    
    with console.status("[cyan]Generating user stories..."):
        result = agent.execute({"description": description})
    
    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return
    
    console.print(f"\n[green]✓ Generated {result['total_stories']} user stories[/green]")
    console.print(f"[green]✓ Grouped into {result['total_epics']} epics[/green]\n")
    
    for i, story in enumerate(result['user_stories'], 1):
        console.print(f"[bold cyan]{i}. {story['title']}[/bold cyan]")
        console.print(f"   As a [yellow]{story['user_type']}[/yellow]")
        console.print(f"   I want to [white]{story['feature']}[/white]")
        console.print(f"   So that [green]{story['benefit']}[/green]")
        console.print(f"   Priority: [magenta]{story['priority']}[/magenta] | Points: [blue]{story['story_points']}[/blue]\n")
    
    if Confirm.ask("Save results to file?"):
        filename = Prompt.ask("Filename", default="requirements.json")
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        console.print(f"[green]✓ Saved to {filename}[/green]")

@cli.command()
def interactive():
    """Interactive mode to build a complete application"""
    
    console.print("\n[bold cyan]🤖 AI Application Builder[/bold cyan]\n")
    console.print("This will guide you through building a complete application.\n")
    
    app_type = Prompt.ask(
        "What type of application?",
        choices=["web", "api", "cli", "mobile"],
        default="web"
    )
    
    console.print("\nDescribe your application:")
    description = Prompt.ask("Description")
    
    console.print("\n[cyan]Step 1: Generating Requirements[/cyan]")
    agent = MockRequirementsAgent()
    requirements = agent.execute({"description": description})
    console.print(f"[green]✓ Generated {requirements['total_stories']} user stories[/green]")
    
    console.print("\n[cyan]Step 2: Design (Coming Soon)[/cyan]")
    console.print("[yellow]- Database schema[/yellow]")
    console.print("[yellow]- API endpoints[/yellow]")
    console.print("[yellow]- UI components[/yellow]")
    
    console.print("\n[cyan]Step 3: Development (Coming Soon)[/cyan]")
    console.print("[yellow]- Generate code[/yellow]")
    console.print("[yellow]- Create tests[/yellow]")
    
    console.print("\n[cyan]Step 4: Deployment (Coming Soon)[/cyan]")
    console.print("[yellow]- Docker configuration[/yellow]")
    console.print("[yellow]- CI/CD pipeline[/yellow]")
    
    console.print("\n[green]✓ Requirements phase complete![/green]")
    console.print("[yellow]Other agents coming soon...[/yellow]")

@cli.command()
def status():
    """Check system status and configuration"""
    
    console.print("\n[bold cyan]System Status[/bold cyan]\n")
    
    items = {
        "Python": "✓",
        "Base Agent": "✓",
        "Requirements Agent": "✓",
        "Mock Agent": "✓"
    }
    
    for item, status in items.items():
        console.print(f"[green]{status}[/green] {item}")
    
    console.print("\n[bold cyan]LLM Configuration[/bold cyan]\n")
    
    llm_provider = os.getenv("LLM_PROVIDER", "mock")
    console.print(f"Provider: [yellow]{llm_provider}[/yellow]")
    
    if llm_provider != "mock":
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY")
        if api_key:
            console.print(f"API Key: [green]✓ Configured[/green]")
        else:
            console.print(f"API Key: [red]✗ Not found[/red]")
            console.print("\n[yellow]To use AI features:[/yellow]")
            console.print("1. Copy .env.example to .env")
            console.print("2. Add your API key")
            console.print("3. Set LLM_PROVIDER to 'openai' or 'anthropic'")

if __name__ == "__main__":
    cli()