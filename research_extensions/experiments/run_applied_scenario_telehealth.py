"""
run_applied_scenario_telehealth.py

High-level wrapper that runs the telehealth pipeline demo and
produces application-level outputs:
- Emotional risk timeline plot
- Session-level risk distribution summary
"""

from research_extensions.scenarios.telehealth_pipeline_demo import main as telehealth_demo_main


def main():
    telehealth_demo_main()


if __name__ == "__main__":
    main()

