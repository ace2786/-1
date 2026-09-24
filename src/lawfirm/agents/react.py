"""ReAct multi-agent: model plans tool calls, every step audited.

Tools are plain async functions with JSON schemas; the loop feeds the model an
observation after each call and terminates on a final answer (or step budget).
This is the 自主规划/执行轨迹/审计日志 requirement — deliberately bounded so an
8B model stays reliable.
"""
import json, re
from dataclasses import dataclass, field
from ..llm import generate
from ..observe import audit, log, Metrics
from . import tools as T


@dataclass
class Step:
    thought: str
    action: str | None = None
    action_input: dict = field(default_factory=dict)
    observation: str | None = None


REACT_SYSTEM = '''你是离线律所办案AI助手，使用 ReAct 模式工作：Thought(思考) -> Action(动作) -> Observation(观察)，循环直到 Final Answer。

可用工具（Action 必须严格从下列选择，Action Input 为JSON）：
{tool_specs}

输出格式（严格遵守）：
Thought: <一句话推理>
Action: <工具名>
Action Input: <{{...}}JSON参数>

重要纪律：Final Answer 只能基于上面出现过的 Observation 内容，禁止编造未在观察中出现的事实、页码或数量；若观察信息不足，如实说明缺口。

或最终回答时：
Thought: <总结>
Final Answer: <给用户的完整中文回答，引用来源文档与页码>'''


async def ask(question: str, case_id: str, page_context: str | None = None,
              max_steps: int = 6, heavy: bool = False) -> dict:
    specs = "\n".join(f'- {t['name']}: {t['desc']} 参数:{json.dumps(t['params'], ensure_ascii=False)}'
                      for t in T.TOOL_SPECS)
    convo = [f'问题：{question}',
             '（系统提示：必须先至少调用一次工具获取事实，禁止跳过工具直接给 Final Answer）']
    if page_context:
        convo.insert(0, f'当前页面上下文摘要：\n{page_context[:3000]}')
    steps: list[Step] = []
    trace = []
    sys_prompt = REACT_SYSTEM.format(tool_specs=specs)

    for i in range(max_steps):
        user_msg = "\n\n".join(convo)
        out = await generate(user_msg, system=sys_prompt, heavy=heavy, temperature=0.1)
        thought = _extract(out, "Thought:")
        final = _extract(out, "Final Answer:")
        if final:
            steps.append(Step(thought=thought or "", ))
            trace.append({"step": i + 1, "type": "final", "thought": thought})
            audit("agent_final", case_id=case_id, question=question, steps=i + 1)
            Metrics.inc("agent_answers")
            return {"answer": final.strip(), "steps": trace}
        action = _extract(out, "Action:")
        raw_in = _extract(out, "Action Input:")
        if not action:
            convo.append(f'Observation: 格式错误，请重新按格式输出。\n模型上次输出:\n{out[:500]}')
            continue
        try:
            args = json.loads(raw_in) if raw_in else {}
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw_in, re.S)
            args = json.loads(m.group(0)) if m else {}
        obs = await T.execute(action, args, case_id=case_id)
        steps.append(Step(thought=thought, action=action, action_input=args, observation=obs[:400]))
        trace.append({"step": i + 1, "type": "tool", "thought": thought,
                      "action": action, "input": args, "observation": obs[:1200]})
        audit("agent_tool_call", case_id=case_id, action=action, input=args,
              observation_len=len(obs))
        convo.append(f'Thought: {thought}\nAction: {action}\nAction Input: {raw_in}\nObservation: {obs}')

    # step budget exhausted -> force summarize
    out = await generate("\n\n".join(convo) + "\n\n已达最大步骤数，请基于以上观察直接给出 Final Answer。",
                         system=sys_prompt, heavy=heavy)
    final = _extract(out, "Final Answer:") or out
    trace.append({"step": len(trace) + 1, "type": "forced_final"})
    audit("agent_forced_final", case_id=case_id, question=question)
    return {"answer": final.strip(), "steps": trace}


def _extract(text: str, marker: str) -> str | None:
    pat = re.escape(marker) + r"\s*(.*?)(?=\n(?:Thought|Action|Action Input|Observation|Final Answer):|\Z)"
    m = re.search(pat, text, re.S)
    return m.group(1).strip() if m else None