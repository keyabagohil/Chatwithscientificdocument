import timeit
from mistral.llm.wrapper import setup_qa_chain
# from mistral.llm.wrapper import query_embeddings


def generate_response(query):

    # parser = argparse.ArgumentParser()
    # print(parser)
    # parser.add_argument('input',
    #                     type=str,
    #                     default='What is the invoice number value?',
    #                     help='Enter the query to pass into the LLM')
    # parser.add_argument('--semantic_search',
    #                     type=bool,
    #                     default=False,
    #                     help='Enter True if you want to run semantic search, else False')
    # args = parser.parse_args()
    # query=input()
    start = timeit.default_timer()
    # if args.semantic_search:
    #     semantic_search = query_embeddings(query)
    #     print(f'Semantic search: {semantic_search}')
    # else:
    ans=[]
    qa_chain = setup_qa_chain()
    response = qa_chain({'query': query})
    ans.append(f'\nAnswer: {response["result"]}')
    print(f'\nAnswer: {response["result"]}')
    # ans.append(f"SOURCE:")
    print("SOURCE:")
    x=' '.join(list(set([doc.metadata['source'] for doc in response['source_documents']])))
    print(x)
    print('=' * 50)
    end = timeit.default_timer()
    ans.append(f'\nSource: {x}')
    print(f"Time to retrieve answer: {end - start}")
    ans.append(f"Time to retrieve answer: {end - start}")
    print(ans)
    return ans
