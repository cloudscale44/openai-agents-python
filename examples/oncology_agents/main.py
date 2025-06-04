from __future__ import annotations as _annotations

import asyncio

from agents import Agent, Runner, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX


# Define prompt instructions for each agent.
PRE_SCREENING_PROMPT = (
    f"{RECOMMENDED_PROMPT_PREFIX}\n"
    "You are the pre-screening agent for the oncology department. "
    "Your job is to collect basic symptoms and relevant medical history from the patient. "
    "Be concise and empathetic. After gathering enough information, you may handoff to the post consultation agent."
)

POST_CONSULTATION_PROMPT = (
    f"{RECOMMENDED_PROMPT_PREFIX}\n"
    "You are the post consultation care agent for the oncology department. "
    "Explain prescriptions, upcoming procedures, or any ordered lab tests in simple terms. "
    "If the patient mentions a lab report they don't understand, handoff to the lab report agent."
)

LAB_REPORT_PROMPT = (
    f"{RECOMMENDED_PROMPT_PREFIX}\n"
    "You are the lab report explanation agent for the oncology department. "
    "Interpret lab report results for the patient in clear, non-technical language."
)


# Create the agents.
pre_screening_agent = Agent(
    name="PreScreeningAgent",
    instructions=PRE_SCREENING_PROMPT,
)

post_consultation_agent = Agent(
    name="PostConsultationAgent",
    instructions=POST_CONSULTATION_PROMPT,
)

lab_report_agent = Agent(
    name="LabReportAgent",
    instructions=LAB_REPORT_PROMPT,
)

# Wire up handoffs between agents.
pre_screening_agent.handoffs.append(post_consultation_agent)
post_consultation_agent.handoffs.append(handoff(agent=lab_report_agent))
lab_report_agent.handoffs.append(post_consultation_agent)


async def main() -> None:
    current_agent = pre_screening_agent
    conversation: list[str] = []

    while True:
        user_input = input("Patient: ")
        conversation.append(f"Patient: {user_input}")
        result = await Runner.run(current_agent, conversation)

        for item in result.new_items:
            if item.agent is not None:
                print(f"{item.agent.name}: {item.content}")

        conversation = result.to_input_list()
        current_agent = result.last_agent


if __name__ == "__main__":
    asyncio.run(main())
