from env import Env

class PRBORewardManager:
    def __init__(self, env: Env, rubric: str):
        self.env = env
        self.rubric = rubric

    async def _compute_single(self, policy_output: str) -> float:
        return await self.env.get_prbo_reward(policy_output, self.rubric)