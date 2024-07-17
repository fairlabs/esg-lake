import streamlit as st
import numpy as np
import pandas as pd
import datetime
import calendar

import time
import matplotlib.pyplot as plt
import mysql.connector
import re
import plotly.express as px
import plotly.graph_objects as go


import bubble_chart



def date_range(min_start_date : str, timedelta_days_for_default_start_date : int):
    """
    min_start_date : Set the minimun date that can be selected, e.g. '2023-01-01' for the timeline
    timedelta_days_for_default_start_date : the default setting date from today, e.g. 90 means 3 months ago for the defarult

    end_date is automatically set as the latest date
    """
    today = datetime.datetime.now() - datetime.timedelta(days=1)
    start_date = datetime.datetime.strptime(min_start_date, '%Y-%m-%d').date()

    end_date = today.date()
    three_months_ago = today - datetime.timedelta(days=timedelta_days_for_default_start_date)
    default_start = three_months_ago.date()
    date_range = st.date_input("Select Date Range", (default_start, end_date), start_date, end_date, format="MM.DD.YYYY")

    return date_range[0], date_range[1]

def generate_date_lists(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    years = date_range.year.unique()
    year_months = date_range.to_period('M').unique()
    
    # Convert to strings
    years_str = [str(year) for year in years]
    year_months_str = [period.strftime('%Y-%m') for period in year_months]
    
    return years_str, year_months_str


st.set_page_config(
    page_title="Momentum",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

temp_company_data = bubble_chart.get_data_from_mysql().company_gic_info()
gic_info_dict = bubble_chart.get_data_from_mysql().gic_info_mapping_dict()
gic_list = list(gic_info_dict['general_issue_name_en'].values())
gic_name_to_code_dict = {v:k for k, v in gic_info_dict['general_issue_name_en'].items()}

@st.cache_data
def load_data(start_date, end_date):
    data_loader_obj = bubble_chart.data_loader(start_date, end_date)
    company_meta_info_df = data_loader_obj.company_meta_info_df
    input_df_dict = data_loader_obj.get_raw_df()
    return company_meta_info_df, input_df_dict


if 'state' not in st.session_state:
    st.session_state.state={
        "bubble_chart_start_date" : '2024-04-01',
        "bubble_chart_end_date" : '2024-06-30',
        
        "previous_bubble_chart_gic_selection" : None,
        "changed_bubble_chart_gic_selection" : None,

        "previous_bubble_chart_period_type" : None,
        "changed_bubble_chart_period_type" : None,

        "previous_bubble_chart_period_selected" : None,
        "changed_bubble_chart_period_selected" : None,

        "previous_bubble_chart_group_classification" : None,
        "changed_bubble_chart_group_classification" : None,
    }


# st.session_state.state['input_df_dict'] = None
# Top container for date range selection

# date_range_changed_condition = st.session_state.state["previous_bubble_chart_date_range"] != st.session_state.state["changed_bubble_chart_date_range"]
# gic_selection_changed_condition = st.session_state.state["previous_bubble_chart_gic_selection"] != st.session_state.state["changed_bubble_chart_gic_selection"]
# period_type_changed_condition = st.session_state.state["previous_bubble_chart_period_type"] != st.session_state.state["changed_bubble_chart_period_type"]

# period_selection_changed_condition = st.session_state.state["previous_bubble_chart_period_selected"] != st.session_state.state["changed_bubble_chart_period_selected"]
# group_classification_changed_condition = st.session_state.state["previous_bubble_chart_group_classification"] != st.session_state.state["changed_bubble_chart_group_classification"]


with st.container():
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.session_state.state['bubble_chart_start_date']
        end_date = st.session_state.state['bubble_chart_end_date']

        st.write(f"Date Range: {start_date} ~ {end_date}")

        years_str, year_months_str = generate_date_lists(start_date, end_date)
    
        company_meta_info_df, input_df_dict = load_data(start_date, end_date)
            
        
    with col2:
        gic_selection_option = st.selectbox("Select GIC for overview", gic_list)
        gic_code_selected = gic_name_to_code_dict[gic_selection_option]

        st.session_state.state['changed_bubble_chart_gic_selection'] = gic_code_selected
        gic_selection_changed_condition = st.session_state.state["previous_bubble_chart_gic_selection"] != st.session_state.state["changed_bubble_chart_gic_selection"]
        st.session_state.state['previous_bubble_chart_gic_selection'] = gic_code_selected


    col_3, col_4, col_5 = st.columns(3)
    with col_3:
        period_type_selection_option = st.selectbox("Select Period Type", ['whole_period', 'year_month'])

    with col_4:
        if period_type_selection_option == 'whole_period':
            period_selection_option = st.selectbox("Select the Period", ['whole_period'])
            standard_date_momentum_chart = end_date

        # elif period_type_selection_option == 'year':
        #     period_selection_option = st.selectbox("Select the Period", years_str)

        #     filtered_dates = pd.date_range(start=start_date, end=end_date, freq='D').to_series()
        #     filtered_dates = filtered_dates[filtered_dates.dt.year == int(period_selection_option)]
        #     standard_date_momentum_chart = filtered_dates.max().strftime("%Y-%m-%d")

        elif period_type_selection_option == 'year_month':
            period_selection_option = st.selectbox("Select the Period", year_months_str)

            year, month = map(int, period_selection_option.split('-'))
            filtered_dates = pd.date_range(start=start_date, end=end_date, freq='D').to_series()
            filtered_dates = filtered_dates[(filtered_dates.dt.year == year) & (filtered_dates.dt.month == month)]
            standard_date_momentum_chart = filtered_dates.max().strftime("%Y-%m-%d")

    with col_5:
        group_type_selection_option = st.selectbox("Select Sector / Industry", ['sector', 'industry'])


# dataset_obj = bubble_chart.get_dataset_for_graph(period_type_selection_option, 'pulse', f'nfr_{gic_code_selected}_pulse', 'neg_exposure', f'neg_exposure_{gic_code_selected}', input_df_dict, company_meta_info_df)
# pulse_exposure_bubble_fig = dataset_obj.snapshot_with_gic_in_a_period(group_type_selection_option, period_type_selection_option)

with st.container():
    dataset_obj = bubble_chart.get_dataset_for_graph(period_type_selection_option, 'pulse', f'nfr_{gic_code_selected}_pulse', 'neg_exposure', f'neg_exposure_{gic_code_selected}', input_df_dict, company_meta_info_df)
    pulse_exposure_bubble_fig = dataset_obj.snapshot_with_gic_in_a_period(group_type_selection_option, period_selection_option)
    st.plotly_chart(pulse_exposure_bubble_fig)


    st.write(f"Standarad Date : {standard_date_momentum_chart}")

    dataset_obj = bubble_chart.get_dataset_for_graph(period_type_selection_option, 'pulse', f'nfr_{gic_code_selected}_pulse', 'pulse_rsi', f'nfr_{gic_code_selected}_pulse_rsi', input_df_dict, company_meta_info_df)
    pulse_exposure_bubble_fig = dataset_obj.snapshot_with_gic_in_a_period(group_type_selection_option, period_selection_option)

    if isinstance(pulse_exposure_bubble_fig, str):
        st.write(pulse_exposure_bubble_fig)
    else:
        st.plotly_chart(pulse_exposure_bubble_fig)
