import jsonlines
import convenion
from elasticsearch import Elasticsearch
import json
import random
import glob
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/iphone-6-32gb-gold.jl'
# PATH_QUESTION_ANSWER = '/Users/tienthanh/Projects/ML/datapool/tgdd_qa/QA-example.jl'
PATH_QUESTION_ANSWER_INDEXER = './elastic/qa_indexer.jl'

URL_SEARCH = 'http://127.0.0.1:9200/qa_tgdd/_search'


def raw_index_file():
    with jsonlines.open(PATH_QUESTION_ANSWER_INDEXER, mode='w') as writer:
        with jsonlines.open(PATH_QUESTION_ANSWER) as reader:
            for qa in reader:
                if not convenion.is_valid_qa(qa):
                    continue
                id_doc = qa['id_cmt']
                question = qa['question']
                answer = qa['answer']
                question_custom = convenion.customize_string(question)
                answer_custom = convenion.customize_string(answer)
                question_removed_stopword = convenion.customize_and_remove_stopword(question)
                answer_removed_stopword = convenion.customize_and_remove_stopword(answer)
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


def get_search_result(query_obj, page=0, size=10, field_search="question", **kwargs):
    es = Elasticsearch()
    body = {
        "query": {
            "match": {
                field_search: query_obj['question']
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
        "id_query": query_obj['id'],
        "total": results['total'],
        "total_current": len(results['hits']),
        "max_score": results['max_score'],
        "origin_question": query_obj['question'],
        "hits": raw_hits
    }

    return raw_result
    # with open('search_results_exp.json', 'w') as outfile:
    #     json.dump(raw_result, outfile)


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
                if not convenion.is_valid_qa(qa):
                    continue
                arr_question_source.append(qa)
            print(random.choice(arr_question_source))

        user_judge = ''

        while (len(arr_id) != 100) and (user_judge != '0'):
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


def search_by_query_pool():
    with open('./elastic/query_pool.json') as f:
        queries = json.load(f)
        for query_obj in queries:
            if query_obj['searched'] != 0:
                continue
            raw_result = get_search_result(query_obj)
            path = './elastic/search_result/' + str(query_obj['id']) + '.json'
            with open(path, 'w') as outfile:
                json.dump(raw_result, outfile)


def statistic_search_result():
    judged_results_path = glob.glob("./elastic/judged/tmp/*.json")
    count_questions = len(judged_results_path)
    total_pair = 0
    total_good = 0
    total_useful = 0
    notyet_judged = 0
    total_bad = 0

    total_pair_2 = 0
    total_good_2 = 0
    total_useful_2 = 0
    notyet_judged_2 = 0
    total_bad_2 = 0

    for path in judged_results_path:
        with open(path, 'r') as f:
            # print(path)
            judged_result = json.load(f)
            # print(len(judged_result['hits']))
            # total_pair += len(judged_result['hits'])
            notyet_judged += len([question for question in judged_result['hits'] if question['relate_q_q'] == 0])
            total_good += len([question for question in judged_result['hits'] if question['relate_q_q'] == 2])
            total_useful += len([question for question in judged_result['hits'] if question['relate_q_q'] == 1])
            total_bad += len([question for question in judged_result['hits'] if question['relate_q_q'] == -1])
            # if notyet_judged > 0:
            #     print(path)
            #     break
            half_hits = judged_result['hits'][:len(judged_result['hits'])//2]
            notyet_judged_2 += len([question for question in half_hits if question['relate_q_q'] == 0])
            total_good_2 += len([question for question in half_hits if question['relate_q_q'] == 2])
            total_useful_2 += len([question for question in half_hits if question['relate_q_q'] == 1])
            total_bad_2 += len([question for question in half_hits if question['relate_q_q'] == -1])

    # print('notyet_judged: ', notyet_judged)
    print('total_question: ', count_questions)
    total_pair = total_good + total_useful + total_bad
    print('total_pair: ', total_pair)
    print('total_good: %d - %f' % (total_good, (total_good * 100 / total_pair)))
    print('total_useful: %d - %f' % (total_useful, (total_useful * 100 / total_pair)))
    print('total_bad: %d - %f' % (total_bad, (total_bad * 100 / total_pair)))
    print('------------------------------')
    total_pair_2 = total_good_2 + total_useful_2 + total_bad_2
    print('total_pair_half: ', total_pair_2)
    print('total_good_half: %d - %f' % (total_good_2, (total_good_2 * 100 / total_pair_2)))
    print('total_useful_half: %d - %f' % (total_useful_2, (total_useful_2 * 100 / total_pair_2)))
    print('total_bad_half: %d - %f' % (total_bad_2, (total_bad_2 * 100 / total_pair_2)))


def caculate_AP(path):
    with open(path, 'r') as f:
        search_result = json.load(f)
        hits = search_result['hits']
        arr_denote = []
        for hit in hits:
            if hit['relate_q_q'] == 3:
                continue
            if hit['relate_q_q'] == 2 or hit['relate_q_q'] == 1:
                arr_denote.append(1)
            if hit['relate_q_q'] == -1:
                arr_denote.append(0)
        # if convenion.caculate_AP(arr_denote) < 0.5:
        #     print('Path: ', path)
        return convenion.caculate_AP(arr_denote)


def caculate_mAP():
    paths = glob.glob("./elastic/judged/tmp/*.json")
    arr_AP = []
    for path in paths:
        arr_AP.append(caculate_AP(path))
    print(arr_AP)
    nparr_AP = np.array(arr_AP)
    print('mAP: ', nparr_AP.mean())
    print('max AP: ', nparr_AP.max())
    print('min AP: ', nparr_AP.min())
    print('radian AP: ', np.median(nparr_AP))

    print('sort: ', np.sort(nparr_AP))
    dfarr_ap = DataFrame()

statistic_search_result()
caculate_mAP()
