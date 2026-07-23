import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from dotenv import load_dotenv


''' Output of one step becomes the input of the next step. This is a simple example of a chain. 
the chain lets us go ahead and combine multiple steps together. In this example, we will use a prompt template to generate a prompt for the model, and then we will use the model to generate a response. 
The output of the model will be printed to the console.
'''
load_dotenv()


def main():
    print("Hello from langchain-course!")
    information  = """
                    Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman and former public official who is the CEO and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire in terms of US dollars in June 2026; as of July 10, 2026, Forbes estimates his net worth to be US$797 billion.
                    Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
                    In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service; he implemented significant changes and rebranded it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion.
                """

    summary_template = f"""
    given the information {information}, about a person I want you to create:
    1. A short summary
    2. Two interesting facts about them.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    #llm = ChatOpenAI(model_name="gpt-4o",temperature=0)
    '''Notice how Two interesting facts about them. is not generated'''
    llm = ChatOllama(temperature=0, model="gemma3:270m")

    ''' LCEL - LangChain Execution Language - is a new way to define chains in LangChain. It allows you to define chains in a more declarative way, using a simple syntax. In this example, we will use LCEL to define a chain that takes the output of the prompt template and passes it to the LLM. The output of the LLM will be printed to the console.'''
    ''' input of left component is given to right template'''
    '''' Resulting is runnable object that can be invoked to get the final output.'''
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
