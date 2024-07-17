from elasticsearch import Elasticsearch, helpers, exceptions
from elasticsearch.helpers import scan
import pandas as pd
# from dotenv import load_dotenv
import streamlit as st
import os

def get_elasticsearch_client():
    # load_dotenv()

    try:
        ELASTIC_USER = st.secrets["ELASTIC_USER"]
        ELASTIC_PASSWORD = st.secrets["ELASTIC_PASSWORD"]
        ELASTIC_CLOUD_ID = st.secrets["ELASTIC_CLOUD_ID"]

        client = Elasticsearch(
            cloud_id=ELASTIC_CLOUD_ID,
            http_auth=(
                ELASTIC_USER, 
                ELASTIC_PASSWORD
            )
        )
        return client

    except exceptions.AuthenticationException as e:
        print("Authentication failed:", str(e))

def extract_elasticsearch_data(start_date, end_date, corp_code = None, sasb_code = None, entity_relv=0.1, sasb_relv=0.1):
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "date": {
                                "gte":start_date,
                                "lte":end_date
                            }
                        }
                    },
                    {
                        "nested": {
                            "path": "entity_tags",
                            "query": {
                                "bool": {
                                    "must": [
                                        {
                                            "exists": {
                                                "field": "entity_tags.comp_id" 
                                                }
                                        },
                                        {
                                            "range": {
                                                "entity_tags.relevance": {"gte": entity_relv}
                                            }
                                        }
                                    ]
                                }
                            }   
                        }
                    },
                    {
                        "nested": {
                            "path": "sasb_tags",
                            "query": {
                                "bool": {
                                    "must": [
                                        {
                                            "range": {
                                                "sasb_tags.relevance":{"gte": sasb_relv}
                                            }
                                        },
                                    ]
                                }
                            }
                        }
                    }
                ],
                "must_not": [ 
                      {
                        "terms": {
                            "category": ["opinion", "living_culture"]
                        }
                      }
                    ]
                }
            },
        "sort": [
            {"date": {"order": "asc"}}
        ]
    }
    
    if corp_code != None:
        query['query']['bool']['must'][1]['nested']['query']['bool']['must'][0] = {'match': {'entity_tags.comp_id': corp_code}}
    if sasb_code != None:
        query['query']['bool']['must'][2]['nested']['query']['bool']['must'].insert(0,{"match": {"sasb_tags.id":sasb_code}})
        
    client = get_elasticsearch_client()
    res = scan(client,query=query,index="esg-lens",scroll='2m',size=10000)
    items = [hit['_source'] for hit in res]   
    df = pd.DataFrame(items)
    return df

def calculate_entity_relevance(_ratio, _sent_i):
    # define title and topline
    _title = [0] # title itself
    _topline = [1, 2, 3] # first 3 sentence
    w = 1
    if any(i in _sent_i for i in _title):
        w += 0.7
    if any(i in _sent_i for i in _topline):
        w += 0.3
    relevance = _ratio * w
    return relevance

def extract_event(df, corp_code=None, sasb_code=None, intersect_relevance_threshold=0.13):
    df = df[~df.summary.str.contains('목표가|목표주가')].copy()
    df = df[~df.title.str.contains('목표가|목표주가')].copy()

    explode_entity = df.explode('entity_tags')
    if corp_code != None:
        explode_entity = explode_entity[explode_entity['entity_tags'].apply(lambda x: x.get('comp_id') == corp_code)]
        entity_modify = explode_entity['entity_tags'].to_dict()
        df['entity_tags'] = df.index.map(entity_modify)
        del entity_modify
        explode_df = df.explode('sasb_tags')
        explode_df = explode_df[~explode_df.entity_tags.isna()]
    else:
        explode_entity = explode_entity[explode_entity['entity_tags'].apply(lambda x: x.get('comp_id') != None)]
        explode_df = explode_entity.explode('sasb_tags')
        explode_df = explode_df[~explode_df.entity_tags.isna()]
        
    if sasb_code != None:
        explode_df = explode_df[explode_df['sasb_tags'].apply(lambda x: x.get('id') == sasb_code)]
    
    explode_df['intersect_index'] = explode_df.apply(lambda row: set(row['entity_tags']['parent_sent_i']).intersection(set(row['sasb_tags']['negative_sent_i'])),axis=1)
    explode_df_final = explode_df[explode_df.intersect_index != set()].copy()
    if len(explode_df_final) <1:
        print('No negative sentence matched')
        return None
    
    explode_df_final['intersect_ratio'] = explode_df_final.apply(lambda row : len(row['intersect_index']) / round(row['sasb_tags']['count']/row['sasb_tags']['nonfilter_ratio']),axis=1).tolist()
    explode_df_final['intersect_relv'] = explode_df_final.apply(lambda row : calculate_entity_relevance(row['intersect_ratio'],row['intersect_index']),axis=1)
    explode_df_final['sasb_tag_label'] = explode_df_final['sasb_tags'].apply(lambda x: x['label'])
    explode_df_final['sasb_tag_id'] = explode_df_final['sasb_tags'].apply(lambda x: x['id'])
    explode_df_final.sort_values('intersect_relv',ascending=False,inplace=True)

    explode_df_final['week'] = pd.to_datetime(explode_df_final.date).dt.isocalendar().year.astype(str) + '-' + pd.to_datetime(explode_df_final.date).dt.isocalendar().week.astype(str)
    result = explode_df_final.groupby(['date','sasb_tag_label']).head(1).copy().sort_values('intersect_relv',ascending=False)
    result = result[result['intersect_relv']>=intersect_relevance_threshold]

    result['sasb_tags_origin'] = df.loc[result.index,'sasb_tags'].apply(lambda lst: [{v['label']:v['nonfilter_ratio']} for v in lst])
    result['sasb_tags_origin'] = result['sasb_tags_origin'].apply(lambda data :{key: value for item in data for key, value in item.items()})
    result['corp_code'] = result['entity_tags'].apply(lambda x: x['comp_id'])
    result['corp_name'] = result['entity_tags'].apply(lambda x: x['lookup_name'])
    
    result['date'] = pd.to_datetime(result['date'])
    result['date'] = result['date'].dt.strftime('%Y-%m-%d')
    ############################# Add English Name Change summary columns name
    # result = result[['time', 'date', 'week', 'corp_code','corp_name','sasb_tag_id','sasb_tag_label','title','summary','original_link','press','image_','sasb_tags_origin','intersect_relv']].reset_index(drop=True)
    result = result[['date','corp_name','sasb_tag_label','title','original_link']].reset_index(drop=True)
    result = result.head(36)
    
    del explode_df,explode_df_final

    if result.empty:
        return None
    else:
        return result


def make_timeline_data(start_date=None, end_date=None, corp_code=None, sasb_code =None, intersect_relevance_threshold=0.1):
    df = extract_elasticsearch_data(start_date = start_date, end_date = end_date, corp_code = corp_code, sasb_code = sasb_code ,entity_relv = 0.1, sasb_relv = 0.1)

    alarming_message = "No specific events have been extracted with present conditions."

    if df.empty:
        alarming_message = "No specific events have been extracted with present conditions."
        return alarming_message, alarming_message
        
    else:
        company_event_df = extract_event(df = df, corp_code = corp_code, sasb_code = sasb_code, intersect_relevance_threshold = intersect_relevance_threshold)

        if (company_event_df is None):
            return df, alarming_message
        
        else:
            company_event_df.sort_values('date', ascending=False, inplace=True)
            return df, company_event_df
