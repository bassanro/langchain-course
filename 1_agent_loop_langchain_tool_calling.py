from dotenv import load_dotenv
from langchain_openai import tools
load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# - Tools
@tool
def get_product_price(product_name: str) -> float:
    """
    Get the price of a product.
    """
    # Simulate fetching product price from a database or API
    product_prices = {
        "laptop": "999",
        "smartphone": "699",
        "headphones": "199",
    }
    return product_prices.get(product_name.lower(), 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price.
    Available discount tiers: bronze (5%), silver (12%), gold (23%).
    """
    try:
        discount_percentage = {
            "bronze": 5,
            "silver": 12,
            "gold": 23
        }
        print(f"Applying discount: {discount_tier} on price: {price}")
        discount_percentage = discount_percentage.get(discount_tier.lower(), 0)
        return round(price * (1 - discount_percentage / 100), 2)
    except ValueError:
        return 0.0



# -- Agent Loop --- 

@traceable(name="Langchain agent loop", tags=["agent_loop"])
def run_agent(question: str): 
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Running agent with question: {question}")
    print("=" * 50)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        #if no tool calls were made, this is the final answer
        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content

        tool_calls = tool_calls[0]
        tool_name = tool_calls.get('name')
        tool_args = tool_calls.get('args', {})
        tool_call_id = tool_calls.get('id')

        print(f"Tool call: {tool_name} with args: {tool_args}")
        tool_to_use = tool_dict.get(tool_name)
        if not tool_to_use:
            print(f"Tool {tool_name} not found.")
            raise ValueError(f"Tool {tool_name} not found.")

        observation = tool_to_use.invoke(tool_args)
        print(f"Tool Result: {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_name=tool_name, tool_call_id=tool_call_id))
        
    print(f"No final answer after {MAX_ITERATIONS} iterations.")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent!")
    print()
    result = run_agent("What is the price of a laptop with a silver discount?")
    print(result)