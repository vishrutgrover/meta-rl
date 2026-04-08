from client import GTMEnv
from models import GTMAction

with GTMEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="market_dominator")
    while not result.done:
        action = GTMAction(
            budget_allocation={"paid_search": 0.5, "paid_social": 0.3, "email_lifecycle": 0.2},
            segment_targeting={"startup_founders": 0.6, "smb_owners": 0.4},
            messaging={"performance": 0.3, "innovation": 0.3, "ease_of_use": 0.2, "cost_savings": 0.1, "reliability": 0.05, "security": 0.05},
        )
        result = env.step(action)
    print(f"Score: {result.observation.reward}")