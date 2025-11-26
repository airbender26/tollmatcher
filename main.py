import argparse
from typing import Optional
from stratego.env.stratego_env import StrategoEnv
from stratego.models.ollama_model import OllamaAgent
from stratego.prompts import get_prompt_pack
from stratego.utils.parsing import extract_board_block_lines

def build_agent(spec: str,  prompt_name: str, host: Optional[str] = None):
    kind, name = spec.split(":", 1)
    if kind == "ollama":
        return OllamaAgent(model_name=name, temperature=0.2, num_predict=32,
                           prompt_pack=get_prompt_pack(prompt_name), host=host)
    raise ValueError(f"Unknown agent spec: {spec}")

# Later we can make printing board method as more in detail.
def print_board(observation: str):
    block = extract_board_block_lines(observation)
    if block:
        print("\n".join(block))

# With those arguments, user can change game setting
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--p0", default="ollama:gpt-oss:latest")
    p.add_argument("--p1", default="ollama:qwen3-vl:30b")
    p.add_argument("--prompt", default="base", help="Prompt preset name (e.g. base|concise|adaptive)")
    p.add_argument("--ollama-host", default=None, help="Base URL of Ollama server (e.g. http://host:11434)")
    p.add_argument("--env_id", default="Stratego-v0", help="TextArena environment id")
    args = p.parse_args()

    agents = {
        0: build_agent(args.p0, args.prompt, args.ollama_host),
        1: build_agent(args.p1, args.prompt, args.ollama_host),
    }
    env = StrategoEnv(env_id=args.env_id)
    env.reset(num_players=2)

    done = False
    while not done:
        player_id, observation = env.get_observation()
        print_board(observation)

        action = agents[player_id](observation)
        print(f"{agents[player_id].model_name} -> {action}")

        done, _ = env.step(action=action)

    rewards, game_info = env.close()
    print("Game finished.", rewards, game_info)

if __name__ == "__main__":
    cli()