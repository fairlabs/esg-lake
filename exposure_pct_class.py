import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

import re
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class get_data_from_mysql():
    
    def __init__(self):
        # load_dotenv()
        # self.mysql_host = os.getenv("mysql_host")
        # self.mysql_pass = os.getenv("mysql_pass")
        # self.mysql_port = int(os.getenv("mysql_port"))
        # self.mysql_username = os.getenv("mysql_username")
        # self.mysql_database = os.getenv("mysql_database")

        self.mysql_host = st.secrets["mysql_host"]
        self.mysql_pass = st.secrets["mysql_pass"]
        self.mysql_port = int(st.secrets["mysql_port"])
        self.mysql_username = st.secrets["mysql_username"]
        self.mysql_database = st.secrets["mysql_database"]

    def company_gic_info(self, table_name = 'company_info'):
        '''
        table_name = 'company_info' | 'gic_info'
        '''
        db_connection = mysql.connector.connect(
            host = self.mysql_host,
            port = self.mysql_port,
            username = self.mysql_username,
            database = self.mysql_database,
            password = self.mysql_pass
        )
        connection = db_connection
        
        if table_name == 'company_info':
            sql = f"""
            SELECT *
            FROM {self.mysql_database}.company_info
            """
            
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql)
                    result = pd.DataFrame(cursor.fetchall(),columns = cursor.column_names)
                    
            ### Simply modifying the name of the industries
            result.loc[result.sics_industry_name=='E-commerce', 'sics_industry_name'] = 'E-Commerce'
            result.loc[result.sics_industry_name=='Aerospace & Defense', 'sics_industry_name'] = 'Aerospace & Defence'
            result.loc[result.sics_industry_name=='Oil & Gas â€“ Refining & Marketing', 'sics_industry_name'] = 'Oil & Gas – Refining & Marketing'
                    
        elif table_name == 'gic_info':
            sql = f"""
            SELECT *
            FROM {self.mysql_database}.general_issue
            """
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql)
                    result = pd.DataFrame(cursor.fetchall(),columns = cursor.column_names)
                    
        connection.close
        
        return result

    def company_info_mapping_dict(self):
        company_info = self.company_gic_info('company_info')
        company_info_dict  = company_info.set_index('corp_code')[['corp_name_en_clean','sics_industry_name']].to_dict()
        corp_code_to_name = {k:v for k,v in company_info_dict['corp_name_en_clean'].items()}
        return corp_code_to_name,company_info_dict

    def gic_info_mapping_dict(self):
        gic_info = self.company_gic_info('gic_info')
#         gic_info.general_issue_code = gic_info.general_issue_code.astype(str)
        gic_info_dict = gic_info.set_index('general_issue_code')[['general_issue_name_en','general_issue_name_ko','tax_code']].to_dict()

        return gic_info_dict
    
    def get_sasb_base_gic_df(self):
        sasb_base_gic_df = pd.read_csv('https://github.com/fairlabs/esg-lake/blob/main/materials/sasb_base_gic_file.csv')
        # sasb_base_gic_df = pd.read_csv('./materials/sasb_base_gic_file.csv')
        return sasb_base_gic_df
    

class get_data_from_redshift():
    """
    Example of 'corp_codes_str' is below. It is available to both multiple or single companies.
    
    corp_codes_str = '00149655, 00126380, 00159193, ...'
    corp_codes_str = '00149655'
    """
        
    def __init__(self, start_date, end_date, corp_codes_str):
        self.start_date = start_date
        self.end_date = end_date
        self.corp_codes_str = corp_codes_str

        # load_dotenv()
        # self.redshift_user = os.getenv("redshift_user")
        # self.redshift_pass = os.getenv("redshift_pass")
        # self.redshift_endpoint = os.getenv("redshift_endpoint")
        # self.redshift_port = int(os.getenv("redshift_port"))
        # self.redshift_dbname = os.getenv("redshift_dbname")

        self.redshift_user = st.secrets["redshift_user"]
        self.redshift_pass = st.secrets["redshift_pass"]
        self.redshift_endpoint = st.secrets["redshift_endpoint"]
        self.redshift_port = int(st.secrets["redshift_port"])
        self.redshift_dbname = st.secrets["redshift_dbname"]


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
        
        print(f"Started retrieving {table_name} data from redshift...", end='\r')

        conn = self.get_redshift_con()

        if table_name == 'attention':

            query = f"""
                        SELECT *
                        FROM esg_{table_name}_v2
                        WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}' AND corp_code IN ('{self.corp_codes_str}')
                    ;"""
            
        elif table_name == 'exposure':
            query = f"""
                        SELECT *
                        FROM esg_{table_name}_v2
                        WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}' AND corp_code IN ('{self.corp_codes_str}')
                    ;"""

        elif table_name == 'risk':
            query = f"""
                        SELECT *
                        FROM esg_{table_name}_v2
                        WHERE date BETWEEN '{self.start_date}' AND '{self.end_date}' AND corp_code IN ('{self.corp_codes_str}')
                    ;"""
            
        else:
            print("[Error] table_name is not correct. Please check again the letters.")

        res = conn.execute(query)
        cols = res.keys()
        data = res.fetchall()
        res_df = pd.DataFrame(data, columns=cols)

        print(f"Finished retrieving {table_name} from reshift...", end='\r')

        return res_df
    

class group_risk_distribution(get_data_from_mysql):
    
    def __init__(self, industry_or_sector, name_of_industry_or_sector, start_date, end_date):
        """
        Example:
        
        - industry_or_sector = 'industry'
        - name_of_industry_or_sector = 'Hardware'
        - start_date = '2024-01-01'
        - end_date = '2024-03-31'
        """
        # Select 'industry' or 'sector'
        self.industry_or_sector = industry_or_sector
        self.name_of_industry_or_sector = name_of_industry_or_sector
        self.start_date = start_date
        self.end_date = end_date
        
        # get company_data, gic_full_info
        super().__init__()
        self.company_data = super().company_gic_info('company_info')
        self.gic_full_info = super().company_gic_info('gic_info')
        self.corp_name_to_corp_code = self.company_data[['corp_name_ko_clean', 'corp_code']].set_index('corp_name_ko_clean').to_dict()['corp_code']
        self.corp_code_to_name_dict, _ = super().company_info_mapping_dict()
        self.gic_dict = super().gic_info_mapping_dict()
        self.sasb_base_gic_df = super().get_sasb_base_gic_df()

    
    def get_corp_codes(self):
        """
        Get 'corp_codes' in the selected 'industry' or 'sector' by a list.
        """
        
        if self.industry_or_sector in ('industry', 'sector'):
            # Check if the name of industry/sector is correct
            total_list_industry_or_sector = self.company_data[f"sics_{self.industry_or_sector}_name"].unique().tolist()

            if self.name_of_industry_or_sector not in total_list_industry_or_sector:
                print(f"#########[Notion]#########")
                print(f"The '{self.name_of_industry_or_sector}' is not in the {self.industry_or_sector} list. \nPlease check again in the table below\n")

                # pd.set_option('display.max_rows', 70)
                sector_industry_info_df = self.company_data.groupby(['sics_sector_name', 'sics_industry_name']).count().reset_index(level=1)[['sics_industry_name']]
                # display(sector_industry_info_df)

                return

            # Extract exact corp_codes in the industry/sector 
            corp_code_ls = self.company_data[self.company_data[f"sics_{self.industry_or_sector}_name"] == self.name_of_industry_or_sector]['corp_code'].tolist()


            print(f"Completed in getting all corp_codes in [{self.name_of_industry_or_sector}]\n")

            print(f"#########[Information]#########")
            print(f"Category : {self.industry_or_sector}")
            print(f"Name : {self.name_of_industry_or_sector}")
            print(f"From : {self.start_date}")
            print(f"End : {self.end_date}\n")
        
        elif self.industry_or_sector == 'portpolio':
            corp_code_ls = list(map(lambda x : self.corp_name_to_corp_code[x], self.name_of_industry_or_sector))
            
        return corp_code_ls
    
    def get_sasb_base_gic(self, corp_codes_ls):
        """
        Get sasb_gic list for the selected 'industry' or 'sector'.
        """
        if self.industry_or_sector in ('industry', 'sector'):
            # Extract sasb_base_gic_code
            sasb_base_gic_code_ls = sorted(list(set(self.sasb_base_gic_df[self.sasb_base_gic_df[self.industry_or_sector] == self.name_of_industry_or_sector]['gic'])))        
            sasb_base_gic_name_ls = [self.gic_full_info[self.gic_full_info['general_issue_code'] == x]['general_issue_name_en'].iloc[0] for x in sasb_base_gic_code_ls]
            
        elif self.industry_or_sector == 'portpolio':
            portpolio_industry_ls = self.company_data[self.company_data.corp_code.isin(corp_codes_ls)]['sics_industry_name'].unique().tolist()
            
            sasb_base_gic_code_ls = sorted(list(set(self.sasb_base_gic_df[self.sasb_base_gic_df['industry'].isin(portpolio_industry_ls)]['gic'])))        
            sasb_base_gic_name_ls = [self.gic_full_info[self.gic_full_info['general_issue_code'] == x]['general_issue_name_en'].iloc[0] for x in sasb_base_gic_code_ls]
        
        print(f"#########[sasb_base_gic_name_ls]#########")
        [print(i+1, gic) for i, gic in enumerate(sasb_base_gic_name_ls)]
        print("")
        
        return sasb_base_gic_name_ls
    
    def get_industry_sector_index_df(self, index_name='neg_exposure'):
        """
        Get index_df such as below.
        
        index_name = ['neg_exposure', 'pos_exposure', 'neg_attention', 'pos_attention']
        """
        
        corp_codes_ls = self.get_corp_codes()
        corp_codes_str = "','".join(corp_codes_ls)

        sasb_base_gic_ls = self.get_sasb_base_gic(corp_codes_ls)
        
        group_index_df_obj = get_data_from_redshift(self.start_date, self.end_date, corp_codes_str)
        
        if 'exposure' in index_name:
            extract_column_format = index_name + '_\d{3}$'
            
            df = group_index_df_obj.get_companies_index_data('exposure')
            df = df.filter(regex=r'date|corp_code|{}'.format(extract_column_format))
#             df = df.filter(regex=r'{}'.format(extract_column_format))

            df = self.get_sasb_tag_names(df)
        
        
        elif 'attention' in index_name:
            extract_column_format = index_name + '_\d{3}$'
            
            df = group_index_df_obj.get_companies_index_data('attention')
            df = df.filter(regex=r'date|corp_code|{}'.format(extract_column_format))
#             df = df.filter(regex=r'{}'.format(extract_column_format))

            df = self.get_sasb_tag_names(df)

    
        elif 'pulse' in index_name:
            extract_column_format = 'nfr_\d{3}_pulse$'
            df = group_index_df_obj.get_companies_index_data('risk')
#             df = df.filter(regex=r'date|corp_code|{}'.format(extract_column_format))
            df = df.filter(regex=r'date|corp_code|nfr_raw_pulse$|nfr_sasb_pulse$|nfr_\d_pulse$|nfr_\d{3}_pulse$')
            df = self.get_sasb_tag_names(df)
        
        
        return corp_codes_ls, sasb_base_gic_ls, df
    
    
    def get_sasb_tag_names(self, df):
        
        df['corp_name_en_clean'] = df['corp_code'].apply(lambda x : self.corp_code_to_name_dict.get(x,x))
#         df.rename(columns={"corp_code":"corp_name_en_clean"}, inplace=True)
        df.set_index('date', inplace=True)
        
        gic_code_to_name_dict = self.gic_dict['general_issue_name_en']
        
        gic_code_to_name_dict['raw'] = 'Overall'
        gic_code_to_name_dict['sasb'] = 'Overall(SASB)'
        gic_code_to_name_dict[1] = 'Environmental'
        gic_code_to_name_dict[2] = 'Social Capital'
        gic_code_to_name_dict[3] = 'Human Capital'
        gic_code_to_name_dict[4] = 'Business Model and Innovation'
        gic_code_to_name_dict[5] = 'Leadership and Governance'
        
#         print([gic_code_to_name_dict.get(int(re.search('\d+', node).group()), node) for node in df.columns[1:] if re.search('\d+', node) is not None])
#         df.columns = [df.columns[0]] + [gic_code_to_name_dict.get(int(re.search('\d+', node).group()), node) for node in df.columns[1:]]

        columns = df.columns
        new_columns = [columns[0]]  # Keep the first column as is
        substrings = ['raw', 'sasb']
        
        for node in columns[1:]:
            match = re.search(r'\d+', node)
            if match:
                # If a digit is found in the node name, use it to get the name from the dictionary
                gic_code = int(match.group())
                new_columns.append(gic_code_to_name_dict.get(gic_code, node))
            elif any(sub in node for sub in substrings):
                # If no digit but an underscore is found, use the part after the underscore
                gic_code_part = node.split("_")[1]
                new_columns.append(gic_code_to_name_dict.get(gic_code_part, node))
            else:
                # Otherwise, keep the original node name
                new_columns.append(node)
                
        df.columns = new_columns
        
        return df
    

class create_feature(get_data_from_mysql):
    def __init__(self, corp_code_ls, sasb_base_gic_ls, df):
        super().__init__()
        self.gic_full_info = super().company_gic_info('gic_info')
        self.corp_codes = corp_code_ls
        self.df = df.drop('corp_code', axis=1)
        self.sasb_base_gic_ls = sasb_base_gic_ls
#         self.sasb_base_gic_df = super().get_sasb_base_gic_ls()
    
    def cal_proportion_in_df(self, df):
        '''
        Just calculate the proportion as the last process.
        - Let the last column as proportion values in total.
        '''
        
        last_column = df.columns[-1]
        total_sum = df[last_column].sum()
        
        df[last_column] = round(df[last_column].div(total_sum) * 100, 2)
        df.rename(columns={last_column : 'proportion'}, inplace=True)
        
        return df
    
    def get_proportion_df(self):
        '''
        Get proportion of 'exposure' | 'attention' calculated by sum in the selected period.
        '''
        
        base_df = self.df
        
        temp_df = base_df.groupby('corp_name_en_clean').sum().stack().reset_index()
        temp_df.columns = ['corp_nm', 'gic_nm', 'sum_value']
        
        gic_grouped_df = temp_df.groupby('gic_nm')['sum_value'].sum().sort_values(ascending=False).reset_index()
                
        result_df = self.cal_proportion_in_df(gic_grouped_df)
        
        print(f"#########[sasb_base_gic_name_ls]#########")
        [print(i+1, gic) for i, gic in enumerate(self.sasb_base_gic_ls)]

        # result_df = result_df.style.applymap(self.draw_color, sasb_code_ls=self.sasb_base_gic_ls)
        
        return result_df
    
    def draw_color(self, x, sasb_code_ls):
        '''
        Represent the sasb_base_gic
        '''
        
        if x in sasb_code_ls:
#             color = f"background-color : rgb(255, 155, 80)"
            color = f"background-color : rgb(255, 187, 92)"
            return color
        else:
            return ""
    
    def proportion_companies_all_gic(self):
        '''
        Proportion of companies in all gic (raw)
        '''
        
        temp_df = self.df.groupby('corp_name_en_clean').sum().stack().reset_index()
        temp_df.columns = ['corp_nm', 'gic_nm', 'sum_value']

        comp_rank_in_all_gic_df = temp_df.groupby('corp_nm')['sum_value'].sum().sort_values(ascending=False).reset_index()
        result_df = self.cal_proportion_in_df(comp_rank_in_all_gic_df)

        
#         ======================================================================
        # if len(result_df) > 5:
        #     temp = result_df.iloc[:5]

        #     others_proportion = round(result_df.iloc[5:]['proportion'].sum(), 2)
        #     others_data = {'corp_nm' : 'others',
        #                   'proportion' : others_proportion}

        #     temp =temp.append(others_data, ignore_index=True)
        #     result_df = temp

        if len(result_df) > 5:
            temp = result_df.iloc[:5].copy()  # Use .copy() to avoid SettingWithCopyWarning

            others_proportion = round(result_df.iloc[5:]['proportion'].sum(), 2)
            others_data = pd.DataFrame({'corp_nm': ['others'], 'proportion': [others_proportion]})
        
            temp = pd.concat([temp, others_data], ignore_index=True)
            result_df = temp
        else:
            result_df = result_df[result_df.proportion != 0]
#         ======================================================================
        
        
        self.get_pie_chart(result_df)
        plt.title("[Proportion of companies' in all gic]")
        
        # plt.savefig('./total_gic_companies_pie_Chart.png', dpi=500, bbox_inches='tight')  # Specify the file name, DPI, and bbox_inches='tight' to avoid trimming the plot
        plt.show()
        
        return result_df
    
    def proportion_companies_one_gic(self, selected_gic_nm):
        '''
        Proportion of companies when look through one specific gic
        '''
        
        temp_df = self.df.groupby('corp_name_en_clean').sum().stack().reset_index()
        temp_df.columns = ['corp_nm', 'gic_nm', 'sum_value']
        
        comp_rank_in_one_gic_df = temp_df[temp_df['gic_nm'] == selected_gic_nm].sort_values('sum_value', ascending=False)
        result_df = self.cal_proportion_in_df(comp_rank_in_one_gic_df)
        
        
#         ======================================================================
        # if len(result_df) > 5:
        #     temp = result_df.iloc[:5]

        #     others_proportion = round(result_df.iloc[5:]['proportion'].sum(), 2)
        #     others_data = {'corp_nm' : 'others',
        #                   'proportion' : others_proportion}

        #     temp =temp.append(others_data, ignore_index=True)
        #     result_df = temp

        if len(result_df) > 5:
            temp = result_df.iloc[:5].copy()  # Use .copy() to avoid SettingWithCopyWarning

            others_proportion = round(result_df.iloc[5:]['proportion'].sum(), 2)
            others_data = pd.DataFrame({'corp_nm': ['others'], 'proportion': [others_proportion]})
        
            temp = pd.concat([temp, others_data], ignore_index=True)
            result_df = temp
        else:
            result_df = result_df[result_df.proportion != 0]
#         ======================================================================


        # self.get_pie_chart(result_df)
        plt.title(f"[Proportion of companies' in '{selected_gic_nm}']")
        
        # plt.savefig(f'./{selected_gic_nm}_companies_pie_Chart.png', dpi=500, bbox_inches='tight')
        # plt.show()
        
        return result_df
    
    def get_pie_chart(self, df):
        '''
        Make a pie chart by proportions the df
        '''
        colors = sns.color_palette('hls',10)
        
        wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}
        
        plt.pie(df.proportion,
                labels = df.corp_nm,
                autopct='%.1f%%', startangle=260, counterclock=False, wedgeprops=wedgeprops,
               colors = colors)
