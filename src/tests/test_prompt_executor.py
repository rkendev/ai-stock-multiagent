from agents.prompt_executor import PromptExecutor

def test_prompt_executor_openai(monkeypatch):
    exec = PromptExecutor()
    result = exec.execute_prompt("Hello")
    assert "OpenAIExecutor" in result

def test_prompt_executor_llama(monkeypatch):
    exec = PromptExecutor(use_local_llama=True)
    result = exec.execute_prompt("Hello")
    assert "LocalLlamaExecutor" in result
