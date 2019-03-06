import jsonlines
import os.path
from convenion import *
from elasticsearch import Elasticsearch
import json
import random

PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/iphone-6-32gb-gold.jl'
# PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/QA-example.jl'
PATH_QUESTION_ANSWER_INDEXER = './elastic/qa_indexer.jl'

URL_SEARCH = 'http://127.0.0.1:9200/qa_tgdd/_search'


def raw_index_file():
    with jsonlines.open(PATH_QUESTION_ANSWER_INDEXER, mode='w') as writer:
        with jsonlines.open(PATH_QUESTION_ANSWER) as reader:
            for qa in reader:
                if not is_valid_qa(qa):
                    continue
                id_doc = qa['id_cmt']
                question = qa['question']
                answer = qa['answer']
                question_custom = customize_string(question)
                answer_custom = customize_string(answer)
                question_removed_stopword = customize_and_remove_stopword(question)
                answer_removed_stopword = customize_and_remove_stopword(answer)
                # print(question_custom)
                # print(answer_custom)
                # print(question_removed_stopword)
                # print(answer_removed_stopword)
                doc_id = {"index": {"_id": id_doc}}
                doc = {
                    "question": question,
                    "answer": answer,
                    "question_custom": question_custom,
                    "answer_custom": answer_custom,
                    "question_removed_stopword": question_removed_stopword,
                    "answer_removed_stopword": answer_removed_stopword,
                }
                writer.write(doc_id)
                writer.write(doc)


def get_search_result(query, page=0, size=10, field_search="question", **kwargs):
    es = Elasticsearch()
    body = {
        "query": {
            "match": {
                field_search: query
            }
        },
        # "from": 1,
        "size": 200
    }
    res = es.search(index='qa_tgdd', body=body)
    results = res['hits']
    current_hits = res['hits']['hits']
    raw_hits = []
    for hit in current_hits:
        if len(hit['_source']['question']) == 0 or len(hit['_source']['answer']) == 0:
            continue
        raw_hit = {
            "score": hit['_score'],
            "id": hit['_id'],
            "field_search": field_search,
            "question": hit['_source']['question'],
            "answer": hit['_source']['answer'],
            "relate_q_q": 0
        }
        print(raw_hit)
        raw_hits.append(raw_hit)

    raw_result = {
        "total": results['total'],
        "total_current": len(results['hits']),
        "max_score": results['max_score'],
        "origin_question": query,
        "hits": raw_hits
    }
    print(raw_result)
    with open('search_results.json', 'w') as outfile:
        json.dump(raw_result, outfile)


def cover_judged():
    with open('./judged_1.json', 'r') as f:
        data = json.load(f)
        hits_cover = []
        data["origin_question"] = data['hits'][0]['origin_question']
        for hit in data['hits']:
            hits_cover.append({
                "score": hit['score'],
                "id": hit['id'],
                "field_search": hit['field_search'],
                "question": hit['question_field']['question'],
                "answer": hit['answer_field']['answer'],
                "relate_q_q": hit['question_field']['related']
            })
        data_dunp = {
            "total": data['total'],
            "total_current": data['total_current'],
            "max_score": data['max_score'],
            "origin_question": data['hits'][0]['origin_question'],
            "hits": hits_cover
        }
        print(data)
        with open('judged_1_cover.json', 'w') as outfile:
            json.dump(data_dunp, outfile)



def raw_query_pool():
    with open('./query_pool.json') as f:
        queries = json.load(f)
        print("Current queries len: ", len(queries))
        print("\n")
        arr_id = [query['id'] for query in queries]
        arr_id_checked = list(arr_id)

        arr_question_source = []
        with jsonlines.open(PATH_QUESTION_ANSWER) as reader:
            for qa in reader:
                if not is_valid_qa(qa):
                    continue
                arr_question_source.append(qa)
            print(random.choice(arr_question_source))

        user_judge = ''

        while ((len(arr_id) != 100) and (user_judge != '0')):
            qa_checking = random.choice(arr_question_source)
            if qa_checking['id_cmt'] in arr_id_checked:
                continue
            arr_id_checked.append(qa_checking['id_cmt'])
            # print("Question: %(question)s\n" %qa_checking)
            # print('Input your jugde for quenstion: ')
            user_judge = input(qa_checking['question'] + '\n')
            if user_judge != '1':
                print("Collecting next question...\n")
                continue
            print("Add to query...\n")
            arr_id.append(qa_checking['id_cmt'])
            queries.append({
                'id': qa_checking['id_cmt'],
                'question': qa_checking['question'],
                'searched': 0
            })
            print("Current queries len: ", len(queries))
            print("\n")

        with open('./query_pool.json', 'w') as outfile:
            json.dump(queries, outfile)





query = 'Samsung j6 với ip6 loại nào dùng được hơn ạ'
# get_search_result(query=query)
raw_query_pool()

