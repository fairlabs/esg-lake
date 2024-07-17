from sqlalchemy import create_engine
import re
import pandas as pd
from typing import Tuple
import plotly.express as px
import plotly.graph_objects as go
import re
from sqlalchemy import create_engine
import time
# from dotenv import load_dotenv
import streamlit as st
import os

from exposure_pct_class import get_data_from_mysql




class get_data_from_redshift():
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

        # load_dotenv()
        # self.redshift_user = os.getenv("redshift_user")
        # self.redshift_pass = os.getenv("redshift_pass")
        # self.redshift_endpoint = os.getenv("redshift_endpoint")
        # self.redshift_port = int(os.getenv("redshift_port"))
        # self.redshift_dbname = os.getenv("redshift_dbname")

        self.mysql_host = st.secrets["mysql_host"]
        self.mysql_pass = st.secrets["mysql_pass"]
        self.mysql_port = int(st.secrets["mysql_port"])
        self.mysql_username = st.secrets["mysql_username"]
        self.mysql_database = st.secrets["mysql_database"]
        
    def get_redshift_con(self):
        
        engine_string = "postgresql+psycopg2://%s:%s@%s:%d/%s" \
                        % (self.redshift_user, self.redshift_pass, self.redshift_endpoint, self.redshift_port, self.redshift_dbname)
        engine = create_engine(engine_string)
        redshift_con = engine.connect()
        
        return redshift_con

    def get_companies_index_data(self, table_name):
        """
        table_name = 'exposure' | 'attention' | 'risk'
        """
        
        start_time = time.time()
#         print(f"Started retrieving {table_name} data from redshift...", end='\r')
        print("\n[Table Name]", table_name)

        conn = self.get_redshift_con()

        if table_name == 'attention':

            query = f"""
                        SELECT date,corp_code,
                            pos_attention_110,neg_attention_110,neu_attention_110,pos_attention_120,neg_attention_120,neu_attention_120,pos_attention_130,neg_attention_130,neu_attention_130,pos_attention_140,neg_attention_140,neu_attention_140,pos_attention_150,neg_attention_150,neu_attention_150,pos_attention_160,neg_attention_160,neu_attention_160,
                            pos_attention_210,neg_attention_210,neu_attention_210,pos_attention_220,neg_attention_220,neu_attention_220,pos_attention_230,neg_attention_230,neu_attention_230,pos_attention_240,neg_attention_240,neu_attention_240,pos_attention_250,neg_attention_250,neu_attention_250,pos_attention_260,neg_attention_260,neu_attention_260,pos_attention_270,neg_attention_270,neu_attention_270,
                            pos_attention_310,neg_attention_310,neu_attention_310,pos_attention_320,neg_attention_320,neu_attention_320,pos_attention_330,neg_attention_330,neu_attention_330,
                            pos_attention_410,neg_attention_410,neu_attention_410,pos_attention_420,neg_attention_420,neu_attention_420,pos_attention_430,neg_attention_430,neu_attention_430,pos_attention_440,neg_attention_440,neu_attention_440,pos_attention_450,neg_attention_450,neu_attention_450,
                            pos_attention_510,neg_attention_510,neu_attention_510,pos_attention_520,neg_attention_520,neu_attention_520,pos_attention_530,neg_attention_530,neu_attention_530,pos_attention_540,neg_attention_540,neu_attention_540,pos_attention_550,neg_attention_550,neu_attention_550,
                            pos_attention_raw, pos_attention_sasb, neu_attention_raw, neu_attention_sasb, neg_attention_raw, neg_attention_sasb
                        FROM esg_{table_name}_v2
                        WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}'
                    ;"""
        elif table_name == 'exposure':

            query = f"""
                        SELECT date, corp_code,
                            neg_exposure_110, neg_exposure_120, neg_exposure_130, neg_exposure_140, neg_exposure_150, neg_exposure_160,
                                neg_exposure_210, neg_exposure_220, neg_exposure_230, neg_exposure_240, neg_exposure_250, neg_exposure_260, neg_exposure_270,
                                neg_exposure_310, neg_exposure_320, neg_exposure_330,
                                neg_exposure_410, neg_exposure_420, neg_exposure_430, neg_exposure_440, neg_exposure_450,
                                neg_exposure_510, neg_exposure_520, neg_exposure_530, neg_exposure_540, neg_exposure_550,
                                neg_exposure_1, neg_exposure_2, neg_exposure_3, neg_exposure_4, neg_exposure_5,
                                neg_exposure_sasb, neg_exposure_raw
                        FROM esg_{table_name}_v2
                        WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}'
                    ;"""
            
        # elif table_name == 'exposure':
        #     query = f"""
        #                 SELECT date,corp_code,
        #                 pos_exposure_110,neg_exposure_110,neu_exposure_110,pos_exposure_120,neg_exposure_120,neu_exposure_120,pos_exposure_130,neg_exposure_130,neu_exposure_130,pos_exposure_140,neg_exposure_140,neu_exposure_140,pos_exposure_150,neg_exposure_150,neu_exposure_150,pos_exposure_160,neg_exposure_160,neu_exposure_160,
        #                 pos_exposure_210,neg_exposure_210,neu_exposure_210,pos_exposure_220,neg_exposure_220,neu_exposure_220,pos_exposure_230,neg_exposure_230,neu_exposure_230,pos_exposure_240,neg_exposure_240,neu_exposure_240,pos_exposure_250,neg_exposure_250,neu_exposure_250,pos_exposure_260,neg_exposure_260,neu_exposure_260,pos_exposure_270,neg_exposure_270,neu_exposure_270,
        #                 pos_exposure_310,neg_exposure_310,neu_exposure_310,pos_exposure_320,neg_exposure_320,neu_exposure_320,pos_exposure_330,neg_exposure_330,neu_exposure_330,
        #                 pos_exposure_410,neg_exposure_410,neu_exposure_410,pos_exposure_420,neg_exposure_420,neu_exposure_420,pos_exposure_430,neg_exposure_430,neu_exposure_430,pos_exposure_440,neg_exposure_440,neu_exposure_440,pos_exposure_450,neg_exposure_450,neu_exposure_450,
        #                 pos_exposure_510,neg_exposure_510,neu_exposure_510,pos_exposure_520,neg_exposure_520,neu_exposure_520,pos_exposure_530,neg_exposure_530,neu_exposure_530,pos_exposure_540,neg_exposure_540,neu_exposure_540,pos_exposure_550,neg_exposure_550,neu_exposure_550,
        #                 pos_exposure_raw, pos_exposure_sasb, neu_exposure_raw, neu_exposure_sasb, neg_exposure_raw, neg_exposure_sasb
        #                 FROM esg_{table_name}_v2
        #                 WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}'
        #             ;"""

        elif table_name == 'risk':
            
            query = f"""
                        SELECT date,corp_code,
                            nfr_110_pulse,nfr_110_pulse_rsi,nfr_110_score,nfr_120_pulse,nfr_120_pulse_rsi,nfr_120_score,nfr_130_pulse,nfr_130_pulse_rsi,nfr_130_score,nfr_140_pulse,nfr_140_pulse_rsi,nfr_140_score,nfr_150_pulse,nfr_150_pulse_rsi,nfr_150_score,nfr_160_pulse,nfr_160_pulse_rsi,nfr_160_score,
                            nfr_210_pulse,nfr_210_pulse_rsi,nfr_210_score,nfr_220_pulse,nfr_220_pulse_rsi,nfr_220_score,nfr_230_pulse,nfr_230_pulse_rsi,nfr_230_score,nfr_240_pulse,nfr_240_pulse_rsi,nfr_240_score,nfr_250_pulse,nfr_250_pulse_rsi,nfr_250_score,nfr_260_pulse,nfr_260_pulse_rsi,nfr_260_score,nfr_270_pulse,nfr_270_pulse_rsi,nfr_270_score,
                            nfr_310_pulse,nfr_310_pulse_rsi,nfr_310_score,nfr_320_pulse,nfr_320_pulse_rsi,nfr_320_score,nfr_330_pulse,nfr_330_pulse_rsi,nfr_330_score,
                            nfr_410_pulse,nfr_410_pulse_rsi,nfr_410_score,nfr_420_pulse,nfr_420_pulse_rsi,nfr_420_score,nfr_430_pulse,nfr_430_pulse_rsi,nfr_430_score,nfr_440_pulse,nfr_440_pulse_rsi,nfr_440_score,nfr_450_pulse,nfr_450_pulse_rsi,nfr_450_score,
                            nfr_510_pulse,nfr_510_pulse_rsi,nfr_510_score,nfr_520_pulse,nfr_520_pulse_rsi,nfr_520_score,nfr_530_pulse,nfr_530_pulse_rsi,nfr_530_score,nfr_540_pulse,nfr_540_pulse_rsi,nfr_540_score,nfr_550_pulse,nfr_550_pulse_rsi,nfr_550_score,
                            nfr_raw_pulse, nfr_sasb_pulse, nfr_raw_pulse_rsi, nfr_sasb_pulse_rsi
                        FROM esg_{table_name}_v2
                        WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}'
                    ;"""
            
        elif table_name == 'article_count':
            query = f"""
                        SELECT date,corp_code,
                        article_counts_110,article_counts_120,article_counts_130,article_counts_140,article_counts_150,article_counts_160,
                        article_counts_210,article_counts_220,article_counts_230,article_counts_240,article_counts_250,article_counts_260,article_counts_270,
                        article_counts_310,article_counts_320,article_counts_330,
                        article_counts_410,article_counts_420,article_counts_430,article_counts_440,article_counts_450,
                        article_counts_510,article_counts_520,article_counts_530,article_counts_540,article_counts_550,
                        article_counts_sasb,article_counts_raw
                        FROM esg_attention_v2
                        WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}'
                    ;"""
            
        else:
            print("[Error] table_name is not correct. Please check again the letters.")
        
        res = conn.execute(query)
        cols = res.keys()
        data = res.fetchall()
        res_df = pd.DataFrame(data, columns=cols)
        
        
        # Record the end time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time_seconds = end_time - start_time
        # Convert elapsed time to minutes and seconds
        elapsed_minutes = int(elapsed_time_seconds // 60)
        elapsed_seconds = int(elapsed_time_seconds % 60)
        # Format the elapsed time as "minute:second"
        elapsed_time_formatted = "{:02d}:{:02d}".format(elapsed_minutes, elapsed_seconds)

        print("Elapsed time:", elapsed_time_formatted, "\n")
        print(f"Finished retrieving {table_name} from reshift...", end='\r')

        return res_df
    

class data_loader(get_data_from_mysql):
    def __init__(self, start_date : str, end_date : str):
        super().__init__()
        self.company_data = super().company_gic_info('company_info')
        self.company_meta_info_df = self.company_data[['corp_code', 'stock_code', 'corp_name_ko_clean', 'sics_industry_name', 'sics_sector_name']]
        self.company_meta_info_df.rename(columns={'index' : 'rank', 'sics_industry_name' : 'industry', 'sics_sector_name' : 'sector'}, inplace=True)
        self.gic_code_to_name_dict = super().gic_info_mapping_dict()['general_issue_name_en']
        
        self.start_date = start_date
        self.end_date = end_date
        
    def get_raw_df(self) -> dict:
        retrieving_data_obj = get_data_from_redshift(self.start_date, self.end_date)
        
        full_risk_df = retrieving_data_obj.get_companies_index_data('risk')
        full_exposure_df = retrieving_data_obj.get_companies_index_data('exposure')
#         full_attention_df = retrieving_data_obj.get_companies_index_data('attention')
        
        full_risk_df['date'] = pd.to_datetime(full_risk_df.date)
        full_exposure_df['date'] = pd.to_datetime(full_exposure_df.date)
#         full_attention_df['date'] = pd.to_datetime(full_attention_df.date)
        
        
        pulse_df = full_risk_df.filter(regex=r'date|corp_code|nfr_\d{3}_pulse$|nfr_raw_pulse$|nfr_sasb_pulse$')
        score_df = full_risk_df.filter(regex=r'date|corp_code|nfr_\d{3}_score$|nfr_raw_score$|nfr_sasb_score$')
        pulse_rsi_df = full_risk_df.filter(regex=r'date|corp_code|nfr_\d{3}_pulse_rsi$|nfr_raw_pulse_rsi$|nfr_sasb_pulse_rsi$')

        neg_exposure_df = full_exposure_df.filter(regex=r'date|corp_code|neg_exposure_\d{3}$|neg_exposure_raw$|neg_exposure_sasb$')
#         pos_exposure_df = full_exposure_df.filter(regex=r'date|corp_code|pos_exposure_\d{3}$|pos_exposure_raw$|pos_exposure_sasb$')
#         neu_exposure_df = full_exposure_df.filter(regex=r'date|corp_code|neu_exposure_\d{3}$|neu_exposure_raw$|neu_exposure_sasb$')
#         ttl_exposure_df = self.get_total_exposure_attention_df(neg_exposure_df, pos_exposure_df, neu_exposure_df, 'exposure')

#         neg_attention_df = full_attention_df.filter(regex=r'date|corp_code|neg_attention_\d{3}$|neg_attention_raw$|neg_attention_sasb$')
#         pos_attention_df = full_attention_df.filter(regex=r'date|corp_code|pos_attention_\d{3}$|pos_attention_raw$|pos_attention_sasb$')
#         neu_attention_df = full_attention_df.filter(regex=r'date|corp_code|neu_attention_\d{3}$|neu_attention_raw$|neu_attention_sasb$')
#         ttl_attention_df = self.get_total_exposure_attention_df(neg_attention_df, pos_attention_df, neu_attention_df, 'attention')

        input_df_dict={
            
            'pulse' : pulse_df,
            'score' : score_df,
            'pulse_rsi' : pulse_rsi_df,

            'neg_exposure' : neg_exposure_df,
#             'pos_exposure' : pos_exposure_df,
#             'neu_exposure' : neu_exposure_df,
#             'ttl_exposure' : ttl_exposure_df,

#             'neg_attention': neg_attention_df,
#             'pos_attention' : pos_attention_df,
#             'neu_attention' : neu_attention_df,
#             'ttl_attention' : ttl_attention_df
        }
        
        return input_df_dict
    
    def get_total_exposure_attention_df(self, neg_df, pos_df, neu_df, exposure_or_attention) -> pd.DataFrame:
        neg_df = neg_df[neg_df.columns[~neg_df.columns.str.contains(r'raw|sasb', regex=True)]]
        pos_df = pos_df[pos_df.columns[~pos_df.columns.str.contains(r'raw|sasb', regex=True)]]
        neu_df = neu_df[neu_df.columns[~neu_df.columns.str.contains(r'raw|sasb', regex=True)]]
            
        df = pd.merge(neg_df, pos_df, how='left', on=['date', 'corp_code'])
        df = pd.merge(df,  neu_df, how='left', on=['date', 'corp_code'])

        temp_gic_ls = list(self.gic_code_to_dict.keys())

        for temp_gic in temp_gic_ls:
            neg_col = f'neg_{exposure_or_attention}_{temp_gic}'
            pos_col = f'pos_{exposure_or_attention}_{temp_gic}'
            neu_col = f'neu_{exposure_or_attention}_{temp_gic}'
            total_col = f'ttl_{exposure_or_attention}_{temp_gic}'

            df[total_col] = df[neg_col] + df[pos_col] + df[neu_col]

        res_df = df.filter(regex=fr'date|corp_code|ttl_{exposure_or_attention}_\d{{3}}$')

        return res_df
    

class get_dataset_for_graph(data_loader):
    
    def __init__(self, period_sort : str, y_axis_data : str, y_column_name : str, bubble_size_data : str, bubble_size_column_name : str, input_df_dict : dict, company_meta_info_df):
        '''
        - period_sort:  ['year', 'month', 'year_month']
        '''
        self.period_sort = period_sort
        self.y_axis_data = y_axis_data
        self.y_column_name = y_column_name
        self.bubble_size_data = bubble_size_data
        self.bubble_size_column_name = bubble_size_column_name

        self.input_df_dict = input_df_dict
        self.company_meta_info_df = company_meta_info_df
        self.gic_code_to_name_dict = get_data_from_mysql().gic_info_mapping_dict()['general_issue_name_en']
    
        
    def set_the_period(self, period_sort : str, df : pd.DataFrame) -> pd.DataFrame:
        """ Group the date by the 'period_sort' -> year, month, year_month
        
        [Example] if period_sort == 'year_month':
        
            |    | date       | corp_code | neg_exposure_110 | neg_exposure_120 | ... | neg_exposure_550 | year_month |
            |----|------------|-----------|------------------|------------------|-----|------------------|------------|
            | 0  | 2023-01-01 | 00972503  | 0.0              | 0.0              | ... | 0.0              | 2023-01    |
            | 1  | 2023-01-01 | 00351630  | 0.0              | 0.0              | ... | 0.0              | 2023-01    |
            |... |     ...    |    ...    | ...              | ...              | ... | ...              |   ...      |
            |999 | 2023-04-23 | 01204056  | 0.0              | 0.0              | ... | 0.0              | 2023-04    |
            ...
        """
        
        if period_sort == 'whole_period':
            df['whole_period'] = 'whole_period'

        elif period_sort == 'year':
            df['year'] = df.date.dt.year.astype(str)

        elif period_sort == 'month':
            df['month'] = df.date.dt.month.astype(str)

        elif period_sort == 'year_month':
            df['year_month'] = df.date.dt.to_period('M').astype(str)
        
        return df
    
        
    def calculated_df(self, period_sort : str, index_name : str) -> pd.DataFrame:
        """
        pulse, score, rsi -> the 'last' value in the period
        exposure, attention -> 'sum' of values in the period
        
        [Example] period_sort, 'year_month' in here, already set as index.
        
            | year_month | corp_code | neg_exposure_110 | neg_exposure_120 | ... | neg_exposure_550 |
            |------------|-----------|------------------|------------------|-----|------------------|
            | 2023-01    | 00972503  | 0.0              | 0.0              | ... | 0.0              |
            | 2023-01    | 00351630  | 0.0              | 0.0              | ... | 0.0              |
            | 2023-01    | 00975290  | 0.0              | 0.0              | ... | 0.0              |
            | 2023-01    | 00351995  | 0.0              | ...              | ... | ...              |

        """

        df = self.input_df_dict[index_name].copy()
        df = self.set_the_period(self.period_sort, df)
        df.drop('date', axis=1, inplace=True)

        risk_table_index = ['pulse', 'score', 'rsi']
        exposure_attention_table_index = ['exposure', 'attention']
        
        if any([check_index in index_name for check_index in risk_table_index]):
            res_df = df.groupby([period_sort, 'corp_code']).last().reset_index()
        
        elif any([check_index in index_name for check_index in exposure_attention_table_index]):
            res_df = df.groupby([period_sort, 'corp_code']).sum().reset_index()
        
        return res_df
    
    
    def integrate_bulk_df(self, bubble_size_data : str, bubble_size_column_name : str) -> pd.DataFrame:
        '''
        If you set the marcap data with korean won as bubble_size element later, divide 100 million won.
        
        [Result]
            |        | year_month | corp_code | stock_code | corp_name_ko_clean |               industry                |     sector      | nfr_310_pulse | nfr_310_pulse_rsi |
            |--------|------------|-----------|------------|--------------------|---------------------------------------|-----------------|---------------|-------------------|
            | 0      |   2023-01  |  00100601 |   114190   |      강원에너지       | Engineering & Construction Services   | Infrastructure  |           1.0 |               0.0 |
            | 1      |   2023-02  |  00100601 |   114190   |      강원에너지       | Engineering & Construction Services   | Infrastructure  |           1.0 |               0.0 |
            | 2      |   2023-03  |  00100601 |   114190   |      강원에너지       | Engineering & Construction Services   | Infrastructure  |           1.0 |               0.0 |
            |   ...  |     ...    |    ...    |    ...     |         ...        |                  ...                  |       ...       |        ...    |          ...      |
            | 620287 |   2024-03  |  01705777 |   450410   |    엔에이치스팩28호    | Asset Management & Custody Activities |    Financials   |           1.0 |               0.0 |
            | 620287 |   2024-04  |  01705777 |   450410   |    엔에이치스팩28호    | Asset Management & Custody Activities |    Financials   |           1.0 |               0.0 |

        '''
        
        # 1. Get the calculated df
        y_axis_bulk_data = self.calculated_df(self.period_sort, self.y_axis_data)[[self.period_sort, 'corp_code', self.y_column_name]]
        bubble_size_bulk_data = self.calculated_df(self.period_sort, bubble_size_data)[[self.period_sort, 'corp_code', bubble_size_column_name]]
        

        # 2. Merge with meta_info -> sector, industry 
            ### Merge -> company_meta_info + y_axis_data + bubble_size_data
        y_bulk_meta_df = pd.merge(y_axis_bulk_data, self.company_meta_info_df, how='right', on='corp_code')
        all_merged_df = pd.merge(y_bulk_meta_df, bubble_size_bulk_data, how='left', on=[f"{self.period_sort}", 'corp_code'])
        
            ### just arranging the order of the columns
        meta_info_columns_ls = self.company_meta_info_df.columns.tolist()
        
        arranged_columns_ls = [self.period_sort]
        arranged_columns_ls.extend(meta_info_columns_ls)
        arranged_columns_ls.extend([self.y_column_name, bubble_size_column_name])
        
        all_merged_df = all_merged_df[arranged_columns_ls]
        all_merged_df.dropna(subset=['corp_name_ko_clean'], inplace=True)

        return all_merged_df
    
    
    def get_xtick_order_by_exposure_or_attention(self, df : pd.DataFrame, column_name : str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        [Result]
            |   | year_month |                          industry                         | neg_exposure_310 | industry_rank |
            |---|------------|----------------------------------------------------------|------------------|---------------|
            | 0 | 2023-01    | Advertising & Marketing                                   |          0.024688|             46|
            | 1 | 2023-01    | Aerospace & Defence                                       |          0.930481|             36|
            | 2 | 2023-01    | Agricultural Products                                     |          0.000000|             51|
            | 3 | 2023-01    | Air Freight & Logistics                                   |       6305.931641|              1|
            |...|...         | ...                                                       | ...              | ...           |
            |174| 2024-04    | Technology & Communications                               |        902.184983|              3|
            |175| 2024-04    | Transportation                                            |       1261.503494|              2|

        """
                
        # Sector Rank
        sector_rank_df = df.groupby([self.period_sort, 'sector'])[column_name].sum().reset_index()
        sector_rank_df['sector_rank'] = sector_rank_df.groupby(self.period_sort)[column_name].rank(method='first', ascending=False).astype('int')
        sector_rank_df.drop([column_name], axis=1, inplace=True)
    

        # Industry Rank
        industry_rank_df = df.groupby([self.period_sort, 'industry'])[column_name].sum().reset_index()
        industry_rank_df['industry_rank'] = industry_rank_df.groupby(self.period_sort)[column_name].rank(method='first', ascending=False).astype('int')
        industry_rank_df.drop([column_name], axis=1, inplace=True)

        return sector_rank_df, industry_rank_df
    
    
    def get_company_bulk(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        [Result] -> res_df
            |     | year_month | corp_code | stock_code | corp_name_ko_clean |               industry              |          sector           | nfr_310_pulse   | nfr_310_pulse_rsi | sector_rank | industry_rank |
            |-----|------------|-----------|------------|--------------------|-------------------------------------|---------------------------|-----------------|-------------------|-------------|---------------|
            |   0 | 2023-01    | 00100601  | 114190     | 강원에너지            | Engineering & Construction Services | Infrastructure            |        1.000000 |           0.000000 |           8 |            14 |
            |   1 | 2023-01    | 00186939  | 026150     | 특수건설             | Engineering & Construction Services | Infrastructure            |        1.000000 |           0.000000 |           8 |            14 |
            |   2 | 2023-01    | 00608440  | 340570     | 티앤엘               | Medical Equipment & Supplies        | Health Care               |        2.850474 |           0.095752 |           9 |            35 |
            | ... | ...        | ...       | ...        | ...                | ...                                 | ...                       | ...             | ...                 | ...         | ...           |
            |38765| 2024-04    | 00113492  | 004540     | 깨끗한나라            | Containers & Packaging              | Resource Transformation   |        1.000000 |           0.001319 |           5 |            48 |
            |38766| 2024-04    | 01384477  | 323230     | 엠에프엠코리아         | Asset Management & Custody Activities | Financials               |        1.000000 |           0.000000 |           6 |            39 |
        
        
        [Result] -> sector/industry_rank_df
            |     | year_month |            industry            | industry_rank |
            |-----|------------|--------------------------------|---------------|
            |   0 | 2023-01    | Advertising & Marketing        |            46 |
            |   1 | 2023-01    | Aerospace & Defence            |            36 |
            |   2 | 2023-01    | Agricultural Products          |            51 |
            | ... | ...        | ...                            |           ... |
            |1069 | 2024-04    | Toys & Sporting Goods          |            30 |
            |1070 | 2024-04    | Waste Management               |            66 |
        """
        
        # Get the integrated df for 'y variable' and 'bubble size variabe'
        integrated_df_temp = self.integrate_bulk_df(self.bubble_size_data, self.bubble_size_column_name)

        if ('exposure' in self.bubble_size_column_name) or ('attention') in self.bubble_size_column_name:
            sector_rank_df, industry_rank_df = self.get_xtick_order_by_exposure_or_attention(integrated_df_temp, self.bubble_size_column_name)

        else:
            ### If the bubble_size_data does not belong to 'exposure' or 'attention', just set the order of x_ticks of sector or industry by 'neg_exposure_sum' ###
            temp_gic = self.bubble_size_column_name.split('_')[1]
            temp_column_name = f"neg_exposure_{temp_gic}"

            temp_neg_exposure_df = self.integrate_bulk_df('neg_exposure', temp_column_name)
            sector_rank_df, industry_rank_df = self.get_xtick_order_by_exposure_or_attention(temp_neg_exposure_df, temp_column_name)

        res_df = pd.merge(integrated_df_temp, sector_rank_df, how='left', on=[self.period_sort, 'sector'])
        res_df = pd.merge(res_df, industry_rank_df, how='left', on=[self.period_sort, 'industry'])
        res_df = res_df.sort_values(self.period_sort).reset_index(drop=True)
        res_df = round(res_df, 2)
        
        if (self.y_axis_data == 'pulse') & (self.bubble_size_data == 'pulse_rsi'):
            '''
            Emerging Risk - Data-Driven in platform
            '''
            pulse_cond = (res_df[self.y_column_name] >= 40)
            pulse_rsi_cond = (res_df[self.bubble_size_column_name] >= 40)
            res_df = res_df[pulse_cond & pulse_rsi_cond]
        else:
            pass

        return res_df, sector_rank_df, industry_rank_df
    
    ###############################################################################################################################################################################################################
    ###############################################################################################################################################################################################################
    ####################################################################################        Graph       #######################################################################################################
    ###############################################################################################################################################################################################################
    
    def snapshot_with_gic_in_a_period(self, ind_or_sec, specific_period):
        """
        - period_sort : specific_period
        -    'year'   :    '2024'
        -    'month'  :     '04'
        - 'year_month' :   '2024-04'
        
        [Example]
            snapshot_with_gic_in_a_period('industry', '2024-04')
       """

        company_bulk_data, _, _ = self.get_company_bulk()
        company_bulk_data = company_bulk_data[company_bulk_data[self.period_sort]==specific_period]
        
        if ("rsi" in self.bubble_size_column_name) & (company_bulk_data.empty):
            print(self.gic_code_to_name_dict[int(self.bubble_size_column_name.split("_")[1])])
            gic_name = self.gic_code_to_name_dict[int(self.bubble_size_column_name.split("_")[1])]
            error_message = f"There is no any company with significant momentum in '{gic_name}'"
            print(error_message)
            return error_message
            
        else:
            # [Conditon] Mandatory disclosure companies of which total asset is over 2 trillion won.
    #         company_bulk_data = pd.merge(company_bulk_data, marcap_df, how='left', on='stock_code')
    #         company_bulk_data.dropna(inplace=True)        

            column_name_sector_rank = company_bulk_data.columns[company_bulk_data.columns.str.contains('sector_')][0]
            column_name_industry_rank = company_bulk_data.columns[company_bulk_data.columns.str.contains('industry_')][0]

        #     scale_value = (company_bulk_data['Marcap'].max() - company_bulk_data['Marcap'].min()) / 100
        #     company_bulk_data['Marcap_scale']  = company_bulk_data['Marcap'] / scale_value + 1

            print(f"Total number of companies : {len(company_bulk_data)}")

            if ind_or_sec == 'sector':        
                df = company_bulk_data.sort_values(column_name_sector_rank)
                x = df.sector
                
            elif ind_or_sec == 'industry':
                df = company_bulk_data.sort_values(column_name_industry_rank)
        #         df = df[df['industry_sum_rank'] <= 20]
                x = df.industry
            
            ### Generate Graph ###
            
            if (self.y_axis_data == 'pulse') & (self.bubble_size_data=='pulse_rsi'):
                # width, height = 1500, 800
                width, height = 2000, 800
    #             color_continuous_scale = [[0, 'rgba(255, 69, 0, 0.2)'], [1, 'rgba(255, 69, 0, 1)']]
                color_continuous_scale = [[0, 'rgba(255,179,0, 0.2)'], [1, 'rgba(255,8,0, 1)']]
                bubble_size_max = 20
            else:
                width, height, color_continuous_scale, bubble_size_max = 2000, 800, None, 40
            
            
            fig = px.scatter(df,
                            x=x,
                            y=df[self.y_column_name],
                            color=self.bubble_size_column_name,
                            color_continuous_scale=color_continuous_scale, # Color setting, rsi -> one color
                            size=self.bubble_size_column_name,
                            size_max=bubble_size_max,
                            custom_data=['corp_name_ko_clean', 'sector', 'industry', self.y_column_name, self.bubble_size_column_name],
    #                          labels=dict(sector="Sector by rank<br> (left edge is the hisghest rank)", esg_code=f"Risk score in {year}"),
                            width=width, height=height,
                            template= 'plotly_white',
                            hover_name='corp_name_ko_clean',
                            )
            fig.update_traces(hovertemplate = "<b><i>corp_name</i>: %{customdata[0]}</b>"  +
                                            "<br><b><i>sector</i>: %{customdata[1]}</b>" +
                                            "<br><b><i>industry</i>: %{customdata[2]}</b>" +
                                            "<br>" + 
                                            f"<br><b><i>{self.y_column_name}</i>: %{{customdata[3]}}</b>" +
                                            f"<br><b><i>{self.bubble_size_column_name}</i>: %{{customdata[4]}}</b>"
                                            )
            
            ### Setting title and else
            try:
                gic_num = int(re.search('\d+', self.y_column_name).group())
                gic_name = self.gic_code_to_name_dict[gic_num]

            except:
                gic_name =self.y_column_name


            fig.update_layout(
                            title={
                                'text': f"<b>Monitoring table for '{gic_name}' by {self.y_axis_data} and {self.bubble_size_data}</b><br>",
                                'y':0.97,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'})

            fig.update_xaxes(tickangle=90,
                            showline=True, linewidth=1, linecolor='black', mirror=True,
                            title_text = f"Industry<br>",
                            showgrid=True)

            fig.update_yaxes(showspikes=True, spikecolor="skyblue", spikesnap="cursor", spikemode="across",
                            title_text = f"{self.y_column_name}</br>",
                            showgrid=False)


        #     fig.update_yaxes(type="log")
            return fig
            # fig.show()

    def snapshot_with_gic_in_a_period_slider(self, ind_or_sec):
        """
        period_sort : specific_period
           'year'   :    '2024'
           'month'  :     '04'
       'year_month' :   '2024-04'
       """
        
        company_bulk_data, _, _ = self.get_company_bulk()
#         company_bulk_data_latest = company_bulk_data[company_bulk_data[self.period_sort]==specific_period]
        
        # Mandatory disclosure companies of which total asset is over 2 trillion won.
        company_bulk_data = pd.merge(company_bulk_data, marcap_df, how='left', on='stock_code')
        company_bulk_data.dropna(inplace=True)
        display(company_bulk_data)
    
        column_name_sector_rank = company_bulk_data.columns[company_bulk_data.columns.str.contains('sector_')][0]
        column_name_industry_rank = company_bulk_data.columns[company_bulk_data.columns.str.contains('industry_')][0]
        

#         print(len(company_bulk_data))

        if ind_or_sec == 'sector':
            df = company_bulk_data.sort_values(column_name_sector_rank)  # Update 'sector_rank_column_name' as needed
            x = df['sector']
        elif ind_or_sec == 'industry':
            df = company_bulk_data.sort_values(column_name_industry_rank)  # Update 'industry_rank_column_name' as needed
            x = df['industry']

        # Extracting the custom data fields for hover
        custom_data = df[['corp_name_ko_clean', 'sector', 'industry', self.y_column_name, self.bubble_size_column_name]]

        # Creating the scatter plot
        fig = go.Figure()
        
        for step in range(company_bulk_data[self.period_sort].nunique()):
            fig.add_trace(
                go.Scatter(
                    x = x,
                    y = df[self.y_column_name],
                    visible=False,
                    mode='markers',
                    marker=dict(
                        size=df[self.bubble_size_column_name],
                        sizemode='diameter',
                        sizeref=200,
                        color=df[self.bubble_size_column_name],
                        showscale=True
                    ),
                    customdata=custom_data,
                    hovertemplate=(
                        "<b><i>corp_name</i>: %{customdata[0]}</b><br>" +
                        "<b><i>sector</i>: %{customdata[1]}</b><br>" +
                        "<b><i>industry</i>: %{customdata[2]}</b><br>" +
                        f"<b><i>{self.y_column_name}</i>: %{{customdata[3]}}</b><br>" +
                        f"<b><i>{self.bubble_size_column_name}</i>: %{{customdata[4]}}</b>"
                    ),
                    hoverlabel=dict(namelength=-1),
                    text=custom_data['corp_name_ko_clean']
                ))
            fig.update_layout(width=2000, height=800)
            
        print(len(fig.data))
        fig.data[-1].visible = True

        
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "Slider switched to step: " + str(i)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

            
        # 최종 슬라이더 데이터 업데이트
        fig.update_layout(
        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 50},
            steps=steps
        )])


        
        ### Setting title and else
        try:
            gic_num = int(re.search('\d+', self.y_column_name).group())
            gic_name = self.gic_code_to_dict[gic_num]

        except:
            gic_name =self.y_column_name

        if 'rsi' in self.bubble_size_data :
            fig.update_layout(
                        title={
                            # 'text': f"<b>Monitoring chart in {gic_name} {self.y_axis_data} and {self.bubble_size_data}</b><br>",
                            # 'text': f"<b>Monitoring chart in {gic_name} by pulse and {self.bubble_size_data.split("_")[:-1]}</b><br>",
                            'y':0.97,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})
        else:
            fig.update_layout(
                            title={
                                # 'text': f"<b>Monitoring chart in {gic_name} {self.y_axis_data} and {self.bubble_size_data}</b><br>",
                                # 'text': f"<b>Monitoring chart in {gic_name} by pulse and {self.bubble_size_data.split("_")[:-1]}</b><br>",
                                'y':0.97,
                                'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'})

        fig.update_xaxes(tickangle=90,
                         showline=True, linewidth=1, linecolor='black', mirror=True,
                        title_text = f"Industry<br>",
                        showgrid=True)

        fig.update_yaxes(showspikes=True, spikecolor="skyblue", spikesnap="cursor", spikemode="across",
                        title_text = f"{self.y_column_name}</br>",
                        showgrid=False)

        # Show the plot
        fig.show()
        