from langsmith import evaluate
from app.agent.qa import qa_graph
from langchain.messages import HumanMessage
from app.agent.utils import setup_langsmith

setup_langsmith()
dataset_name = "qasper-qa-e2e"


def qa_agent_wrapper(dataset_input: dict):
    initial_state = {
        "messages": [HumanMessage(content=dataset_input["question"])],
        "selected_ids": [dataset_input["paper_id"]]
    }
    result_state = qa_graph.invoke(initial_state)
    return result_state["messages"][-1].content

def dummy_evaluator(run, example) -> dict:
    return {
        "key": "retrieval_hit_score",
        "score": 1.0
    }

def main(): 
    results = evaluate(
        qa_agent_wrapper,
        data=dataset_name,
        evaluators=[dummy_evaluator]
    )
    print(results)

if __name__ == "__main__":
    main()
