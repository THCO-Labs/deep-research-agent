import os
import sys
sys.path.insert(0, ".")
from deep_research.settings import Settings
from deep_research.model_router import model_for_role

os.environ["TOGETHER_API_KEY"] = "tgp_v1_4ZrbdE0dFCK6RnuZxiTtog6wesCBrX3ApQLpUiN3Bxg"
print("Loading Settings from environment...")
settings = Settings.from_env(project_root=".")

print("Instantiating together model...")
try:
    model = model_for_role(settings, "orchestrator", "together:google/gemma-4-31B-it")
    print("Successfully instantiated model_for_role!")
    res = model.invoke("Say hello in one word.")
    print("Response content:", res.content)
except Exception as e:
    print("Instantiation failed:", e)
