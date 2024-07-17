import streamlit as st
import numpy as np
import pandas as pd
import datetime

import time
import matplotlib.pyplot as plt
import mysql.connector
import re
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')


import exposure_pct_class
import timeline_extract
# from utils import initialize_state



st.set_page_config(
    page_title="DashBoard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
    

# def create_dashboard_page():
def get_basic_info():
    """
    Just getting basic info
    """

    ## Basic Data Retrieval
    # company / gic data
    temp_company_data = exposure_pct_class.get_data_from_mysql().company_gic_info()
    gic_info_dict = exposure_pct_class.get_data_from_mysql().gic_info_mapping_dict()
    gic_name_to_code_dict = {v:k for k, v in gic_info_dict['general_issue_name_en'].items()}

    # Seector List
    sector_list = sorted(temp_company_data.sics_sector_name.unique().tolist())

    # Industry List
    industry_list = sorted(temp_company_data.sics_industry_name.unique().tolist())

    group_name_option =None
    
    return temp_company_data, gic_info_dict, gic_name_to_code_dict, sector_list, industry_list, group_name_option


def compute_and_diaplay_feature_1(group_classification_option, group_name_option, start_date, end_date):
    """
    First Feature, proportion of neg_exposure in a group
    """
    st.subheader(f"Top 10 Controversies")
    group_pct_obj = exposure_pct_class.group_risk_distribution(group_classification_option, group_name_option, start_date, end_date)
    corp_code_ls, sasb_gic_ls, temp_df = group_pct_obj.get_industry_sector_index_df('neg_exposure')
    feature_obj = exposure_pct_class.create_feature(corp_code_ls, sasb_gic_ls, temp_df)

    total_proportion_df = feature_obj.get_proportion_df().iloc[:10]

    custom_color_scale = ['#00FF00', '#FFFF00', '#FF0000']
    # custom_color_scale = ['#28A745', '#FFC107', '#DC3545']
    fig_barh = px.bar(total_proportion_df.sort_values('proportion', ascending=True), y='gic_nm', x='proportion', orientation='h',
                    # title='Top 10 Exposures',
                    labels={'proportion': 'Percentage', 'gic_nm': 'Topic'},
                    color='proportion',
                    # color_continuous_scale=px.colors.sequential.Viridis,
                    color_continuous_scale=custom_color_scale)


    proportion_companies_all_gic = feature_obj.proportion_companies_all_gic()
    custom_color_scale_pie_chart = ['#00FF00', '#FFFF00', '#FF0000']
    fig_pie_all_gic = px.pie(proportion_companies_all_gic, values='proportion', names='corp_nm',
                            color_discrete_sequence=['#8B0000', '#FF4500', '#FFA500', '#FFFF00'])

    # Save fig_barh in session state
    st.session_state.state['fig_barh'] = fig_barh
    st.session_state.state['group_pct_obj'] = group_pct_obj
    st.session_state.state['feature_obj'] = feature_obj
    st.session_state.state['total_proportion_df'] = total_proportion_df
    st.session_state.state['corp_code_ls'] = corp_code_ls
    st.session_state.state['loaded_feature_1'] = True
    st.session_state.state['previous_group_classification_option'] = group_classification_option
    st.session_state.state['previous_group_name_option'] = group_name_option

    st.session_state.state['fig_pie_all_gic'] = fig_pie_all_gic


def pulse_score_feature_4(end_date, corp_codes_str, selected_gic_num):
    risk_df = exposure_pct_class.get_data_from_redshift(end_date, end_date, corp_codes_str).get_companies_index_data('risk')
    risk_df = risk_df.filter(regex=fr'date|corp_code|nfr_{selected_gic_num}_pulse$|nfr_{selected_gic_num}_score$|nfr_{selected_gic_num}_pulse_rsi')
    risk_df['corp_nm'] = risk_df.corp_code.apply(lambda x : temp_company_data[temp_company_data.corp_code==x]['corp_name_en_clean'].iloc[0])

    risk_df.rename(columns={f'nfr_{selected_gic_num}_pulse' : 'pulse',
                            f'nfr_{selected_gic_num}_score' : 'score',
                            f'nfr_{selected_gic_num}_pulse_rsi' : 'pulse_rsi'},
                            inplace=True)
    
    risk_df = round(risk_df, 2)

    custom_color_scale = ['#FFFFE0', '#FFFF00', '#FFA500', '#FF4500', '#8B0000']



    fig_scatter = px.scatter(
        risk_df, x='pulse', y='score', 
        color='pulse_rsi', 
        hover_name='corp_nm', 
        color_continuous_scale=custom_color_scale,
        labels={'x': 'pulse', 'y': 'score'}
    )

    st.plotly_chart(fig_scatter)    


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


# # Basic Data Retrieval
temp_company_data, gic_info_dict, gic_name_to_code_dict, sector_list, industry_list, group_name_option = get_basic_info()


# Set session_state
if 'state' not in st.session_state:
    st.session_state.state = {
        'dashboard_start_date' : '2024-04-01',
        'dashboard_end_date' : '2024-06-30',

        'previous_group_classification_option' : None,
        'changed_group_classification_option' : None,

        'group_name_option' : None,
        'loaded_feature_1' : False,
        'loaded_feature_timeline' : False,

        'previous_group_name_option' : None,
        'changed_group_name_option' : None,

        'previous_group_name_option_for_gic' : None,

        'previous_gic_selection_option' : None,
        'changed_gic_selection_option' : None,

        'timeline_previous_gic_selection_option' : None,
        'timeline_changed_gic_selection_option' : None,
        'timeline_previous_company_option' : None,
        'timeline_changed_company_option' : None,        
    }


# initialize_state()

# Layout with two main columns
left_col, right_col = st.columns([2, 1])

with left_col:
    st.header("ESG Controversy Analytics")

    # Top container for date range selection
    with st.container():
        start_date = st.session_state.state['dashboard_start_date']
        end_date = st.session_state.state['dashboard_end_date']
        st.write(f"Date Range: {start_date} ~ {end_date}")
        

        # 3x3 Grid layout for options and features
        for i in range(3):
            with st.container():
                col1, col2 = st.columns(2)
                if i == 0:    # Row 1 : Settings for group_classification, group_name
                    with col1:
                        group_classification_option = st.selectbox("Classification", ['sector'])

                    with col2:
                        if group_classification_option == 'sector':
                            group_name_option = st.selectbox("Select Sector", sector_list)
                        # elif group_classification_option == 'industry':
                        #     group_name_option = st.selectbox("Select Industry", industry_list)

                        # Update session state with selections
                        st.session_state.state['changed_group_classification_option'] = group_classification_option
                        st.session_state.state['changed_group_name_option'] = group_name_option


                elif i == 1:    # Row 2 : Feature 1, 2
                    with col1:    # Feature 1, barh chart for the proportion of neg_exposure
                        group_classification_changed_option = st.session_state.state['previous_group_classification_option'] != st.session_state.state['changed_group_classification_option']
                        group_name_changed_condition = st.session_state.state['previous_group_name_option'] != st.session_state.state['changed_group_name_option']

                        if not st.session_state.state['loaded_feature_1'] or group_classification_changed_option or group_name_changed_condition:
                                compute_and_diaplay_feature_1(group_classification_option, group_name_option, start_date, end_date)

                                fig_barh = st.session_state.state['fig_barh']
                                st.plotly_chart(fig_barh)

                                with col2:  # Feature 2
                                    st.subheader("Company Breakdown")
                                    fig_pie_all_gic = st.session_state.state['fig_pie_all_gic']
                                    st.plotly_chart(fig_pie_all_gic)

                        else:   # Display cached fig_barh
                            st.subheader(f"Top 10 Controversies")
                            fig_barh = st.session_state.state['fig_barh']
                            st.plotly_chart(fig_barh)

                            with col2:
                                st.subheader("Company Breakdown")  
                                fig_pie_all_gic = st.session_state.state['fig_pie_all_gic']
                                st.plotly_chart(fig_pie_all_gic)


                elif i == 2:    # Row 3 : Featur 3, 4
                    
                        with col1:  # Feature 3: Pie Chart for selected GIC
                            # if date_range_changed_condiiton or group_classification_changed_option or group_name_changed_condition:
                            if st.session_state.state['previous_group_name_option_for_gic'] != st.session_state.state['changed_group_name_option']:     # 'If selected grouop has been changed'
                                st.subheader("Company Breakdown by ESG Topics")

                                gic_selection_list = st.session_state.state['total_proportion_df'].gic_nm.tolist()
                                gic_selection_option = st.selectbox("Select GIC", gic_selection_list)

                                proportion_companies_one_gic = st.session_state.state['feature_obj'].proportion_companies_one_gic(gic_selection_option)

                                fig_pie = px.pie(proportion_companies_one_gic, values='proportion', names='corp_nm', 
                                                    # title='Topics and Their Percentages', 
                                                    # color_discrete_sequence=px.colors.sequential.RdBu
                                                    color_discrete_sequence = ['#8B0000', '#FF4500', '#FFA500', '#FFFF00'])
                                
                                st.plotly_chart(fig_pie)

                                st.session_state.state['gic_selection_option'] = gic_selection_option

                        with col2:
                            # Feature 4: Scatter Chart with Pulse and Score (Row 3, Column 2)
                            # if st.session_state.state['previous_group_name_option_for_gic'] != st.session_state.state['changed_group_name_option']:
                                st.subheader(f"Dynamics in '{gic_selection_option}'")

                                corp_codes_str = "','".join(st.session_state.state['corp_code_ls'])

                                selected_gic_num = gic_name_to_code_dict[gic_selection_option]
                                st.write(f"Standard Date : {end_date}")

                                pulse_score_feature_4(end_date, corp_codes_str, selected_gic_num)

        # Display session state for debugging
        st.session_state.state['previous_dashboard_date_range'] = [start_date, end_date]
        st.write(st.session_state)



with right_col:
    st.header('TimeLine')

    # Settion Options
    with st.container():
        timeline_start_date, timeline_end_date = date_range('2023-01-01', 365)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:      # Setting options
            corp_name_list = temp_company_data[temp_company_data.corp_code.isin(st.session_state.state['corp_code_ls'])]['corp_name_en_clean'].tolist()
            company_name_option = st.selectbox("Select Company", corp_name_list)

            corp_code_selected = temp_company_data[temp_company_data.corp_name_en_clean==company_name_option]['corp_code'].iloc[0]
            st.session_state.state['timeline_changed_company_option'] = company_name_option
            

        with col2:
            gic_selection_option = st.selectbox("Select GIC_timeline", gic_selection_list)
            st.session_state.state['timeline_changed_gic_selection_option'] = gic_selection_option
            

    with st.container():
        timeline_session_cond_1 = st.session_state.state['timeline_previous_gic_selection_option'] != st.session_state.state['timeline_changed_gic_selection_option']
        timeline_session_cond_2 = st.session_state.state['timeline_previous_company_option'] != st.session_state.state['timeline_changed_company_option']

        if timeline_session_cond_1 or timeline_session_cond_2:
            gic_tax_code_selected = gic_info_dict["tax_code"].get(next(key for key, value in gic_info_dict["general_issue_name_en"].items() if value == gic_selection_option), None)
            temp_news_df, temp_company_event_df = timeline_extract.make_timeline_data(timeline_start_date, timeline_end_date, corp_code_selected, gic_tax_code_selected, intersect_relevance_threshold=0.1)

            temp_pulse_df = exposure_pct_class.get_data_from_redshift(timeline_start_date, timeline_end_date, corp_code_selected).get_companies_index_data('risk')
            temp_pulse_df.date = pd.to_datetime(temp_pulse_df.date, errors='coerce')
            temp_pulse_df.sort_values('date', inplace=True)

            pulse_column_name = f'nfr_{gic_tax_code_selected[-3:]}_pulse'

            if isinstance(temp_company_event_df, str):
                # fig_timeline_chart = px.line(temp_pulse_df, x="date", y=pulse_column_name, title='Life expectancy in Canada')
                # st.plotly_chart(fig_timeline_chart)
    
                st.write(temp_company_event_df)

            else:
                temp_company_event_df.reset_index(drop=True, inplace=True)
                temp_company_event_df.index = temp_company_event_df.index + 1

                temp_mark_df = temp_pulse_df[temp_pulse_df.date.isin(temp_company_event_df.date)][['date', pulse_column_name]]

                st.session_state.state['timeline_previous_gic_selection_option'] = gic_selection_option
                st.session_state.state['timeline_previous_company_option'] = company_name_option

                # Function to create Plotly figure
                
                def create_plot(temp_df, temp_mark_df, sasb_pulse):
                    # Scatter plot data
                    scatter_data = go.Scatter(
                        x=temp_mark_df['date'],
                        y=temp_mark_df[sasb_pulse],
                        mode='markers',
                        marker=dict(color='#32CD32', symbol='circle', size=10),
                        name='Issue Point'
                    )

                    # Line plot data
                    line_data = go.Scatter(
                        x=temp_df['date'],
                        y=temp_df[sasb_pulse],
                        mode='lines',
                        line=dict(color='orange'),
                        name=sasb_pulse
                    )

                    # Create layout
                    layout = go.Layout(
                        title="Pulse Over Time",
                        xaxis=dict(title='Date', tickformat='%Y-%m'),
                        yaxis=dict(title='Pulse'),
                        hovermode='closest',
                        # plot_bgcolor='black'
                    )

                    # Combine data and layout into a figure
                    fig = go.Figure(data=[line_data, scatter_data], layout=layout)

                    return fig
                
                timeline_fig = create_plot(temp_pulse_df, temp_mark_df, pulse_column_name)

                # Display the Plotly figure using Streamlit's plotly_chart function
                st.plotly_chart(timeline_fig)

                st.dataframe(temp_company_event_df, height=600)

                st.session_state.state['timeline_fig'] = timeline_fig
                st.session_state.state['timeline_df'] = temp_company_event_df

        else:
            timeline_fig = st.session_state.state['timeline_fig']
            temp_company_event_df = st.session_state.state['timeline_df']
            
            st.plotly_chart(timeline_fig)
            st.dataframe(temp_company_event_df, height=600)
        




    # elastic_search.make_timeline_data(date_range)
# create_dashboard_page()
# if __name__ == "__main__":
#     create_dashboard_page()