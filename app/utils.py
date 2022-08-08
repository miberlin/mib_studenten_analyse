import time
import numpy
import numpy as np
import requests
import pandas
from ruamel.yaml import YAML
import matplotlib.pyplot
import streamlit
from scipy.interpolate import interp1d
from datetime import datetime
import os


AIRTABLE_API_KEY = streamlit.secrets['AIRTABLE_API_KEY']


@streamlit.cache
def read_config(filename):
    with open(filename) as f:
        yaml = YAML(typ='safe')
        cfg = yaml.load(f)
    return cfg


def list_airtable_records(base, table, cfg):
    # Get info from config
    api_url = cfg['api_url']
    base_id = cfg[base]['id']
    table_dict = cfg[base][table]
    table_id = table_dict['id']
    table_name = table_dict['name']
    fields = table_dict['fields']
    #Composing URLs
    table_url = f'{api_url}{base_id}/{table_id}'
    endpoint = f'{table_url}?view=Python'  #maxRecords=100000&view=Python'
    # python requests headers
    headers_list = {
        "Authorization": f'Bearer {AIRTABLE_API_KEY}'
    }
    offset = '0'
    counter = 0
    result = []
    while True:
        querystring = {
            "offset": offset}
        counter += 1
        try:
            response = requests.get(endpoint, params=querystring, headers=headers_list)
            response_table = response.json()
            records = response_table['records']
            result = result + records
            if counter > 5:
                time.sleep(1)
                counter = 0

            try:
                offset = response_table['offset']
            except Exception:
                break
        except Exception as e:
            print(counter, e)

    df1 = pandas.DataFrame.from_dict(result)
    df1 = df1.set_index('id')
    df = pandas.concat([df1.drop(['fields'], axis=1), df1['fields'].apply(pandas.Series)], axis=1)
    if table_name == "Studenten Allgemein":
        df = df.set_index('MiB-ID')
        # Array to string
        # which_array_to_str = ["MiB-Kurse", "Uni-Module", "Studenten x Termine"]
        # for cat in which_array_to_str:
        #     df[cat] = df[cat].str[0]
    elif table_name == "Studenten":
        df = df.set_index('MiB-ID')
        for cat in table_dict['to_clean']:
            df[cat] = df[cat].str[0]
    elif table_name == "Studenten x Termine":
        for cat in table_dict['to_clean']:
            df[cat] = df[cat].str[0]
        # Fill nan with False
        for cat in table_dict['fill_nan']:
            df[cat] = df[cat].fillna(value=False)
    elif table_name == "Termine":
        # Array to string
        for cat in table_dict['to_clean']:
            df[cat] = df[cat].str[0]
        for cat in table_dict['clean_nan']:
            df[cat] = df[cat].astype('float64')
    elif table_name == "PK Ergebnisse":
        for cat in table_dict['to_clean']:
            df[cat] = df[cat].str[0]

    df = df[fields]  # Reduce db
    return df


@streamlit.cache
def generate_dataframes(cfg):
    df_studenten = list_airtable_records('test_ws2122_base', 'studenten_table', cfg)
    df_studentenxtermine = list_airtable_records('test_ws2122_base', 'studentenxtermine_table', cfg)
    # df_mibkurse = list_airtable_records('test_ws2122_base', 'mibkurse_table', cfg)
    df_termine = list_airtable_records('test_ws2122_base', 'termine_table', cfg)
    df_pk_stud = list_airtable_records('test_ws2122_base', 'pkergebnisse_table', cfg)
    return df_termine, df_studenten, df_studentenxtermine, df_pk_stud



# headers_create = {
#         "Authorization": "Bearer key2rtabfsnUeI5pi",
#         "Content-Type": "application/json"
#     }
# http methods
# GET: The GET method requests a representation of the specified resource. Requests using GET should only retrieve data.
# HEAD: The HEAD method asks for a response identical to that of a GET request, but without the response body.
# POST: The POST method is used to submit an entity to the specified resource, often causing a change in state or
#       side effects on the server.

# df = pandas.DataFrame(r.json())
# print(df)

# Data visualization

# Student plots
@streamlit.cache # Interpolations
def interpolation_values(number_of_values, series):
    x = np.linspace(0, number_of_values - 1, num=len(series.dropna()))
    y = series.dropna()
    f = interp1d(x, y, kind='linear')
    xnew = numpy.linspace(0, number_of_values - 1, 50)
    ynew = f(xnew)
    return xnew, ynew


@streamlit.cache
def min_max_dates(df):
    dates = df.unique()
    dates = pandas.to_datetime(dates, format='%d/%m/%y').sort_values()
    dates = dates.strftime('%d/%m/%y')
    min_date = dates[0]
    max_date = dates[-1]
    min_date = datetime.strptime(min_date, '%d/%m/%y')
    max_date = datetime.strptime(max_date, '%d/%m/%y')
    return min_date , max_date


@streamlit.cache
def missing_values_plot(df):
    fill1 = df.ffill()
    fill1.fillna(method='bfill')
    fill2 = df.bfill()
    fill2.fillna(method='ffill')
    df = (fill1 + fill2) / 2
    return df


# For dates: Dates always have to be imported sorted.
# Merge different courses
@streamlit.cache
def student_plot_data_options(df,df_pk,cfg,student_id,start_date,end_date):
    # Student data
    student_data = df[cfg['plots']['students']['fields']]

    # Student data for exam results
    pk_results_data = df_pk[cfg['plots']['students']['pk_results']]

    all_data = pandas.concat([student_data,pk_results_data])
    all_data['Art des Termin'] = all_data['Art des Termin'].fillna('PK')
    values_name = all_data[all_data['MiB-ID']==student_id]
    values_name['Datum-df'] = pandas.to_datetime(values_name['Datum-df'], format='%d/%m/%y')
    values_name.sort_values(by='Datum-df', inplace=True)
    values_name['Datum-df'] = values_name['Datum-df'].dt.date
    values_name['Datum-df'] = values_name['Datum-df'].loc[(start_date <= values_name['Datum-df']) &
                                                          (end_date >= values_name['Datum-df'])]
    values_name['Datum-df'] = pandas.to_datetime(values_name['Datum-df'])
    values_name['Datum-df'] = values_name['Datum-df'].dt.strftime('%d/%m/%y')
    dates = values_name['Datum-df']

    values_name = values_name.loc[values_name['Datum-df'].isin(dates)]
    number_of_values = values_name.shape[0]
    values_range = numpy.linspace(0, number_of_values - 1, num=number_of_values)
    anw = values_name['Anwesenheit']
    aufm = values_name['Aufmerksamkeit'] / (numpy.ones(number_of_values) * 5)
    vers = values_name['Verständnis'] / (numpy.ones(number_of_values) * 5)
    fun = values_name['Fun'] / (numpy.ones(number_of_values) * 10)
    date = values_name['Datum-df']
    late_arrival = numpy.array(values_name['Zu spät?'])
    late_arrival = numpy.argwhere(late_arrival == True).flatten()
    nicht_dabei_idx = numpy.copy(anw)
    nicht_dabei_idx = numpy.argwhere(nicht_dabei_idx == 0).flatten()
    height = 100 * numpy.ones(len(nicht_dabei_idx))

    pk_idx = numpy.array(values_name['Art des Termin'])
    pk_idx = numpy.argwhere(pk_idx == 'PK').flatten()
    #pk_results = values_name[pk_idx]
    pk_actual_points = values_name['Erreicht Prozentual'].dropna() * 100
    pk_guessed_points = values_name['Geschätzt Prozentual'].dropna() * 100

    return number_of_values, values_range, aufm, vers, fun, date, late_arrival, nicht_dabei_idx, height, pk_idx, pk_actual_points,pk_guessed_points


def plot_student_data(df,df_pk, cfg, student_id, start_date, end_date):
    # define columns in dashboard
    col1, col2 = streamlit.columns((1.5, 5))

    # Line selection using checkboxes
    with col1:
        aufm_checkbox = streamlit.checkbox('Aufmerksamkeit', value=True)
        vers_checkbox = streamlit.checkbox('Verständnis', value=True)
        fun_checkbox = streamlit.checkbox('Fun Faktor', value=True)
        absent_checkbox = streamlit.checkbox('Nicht anwesend', value=True)
        late_checkbox = streamlit.checkbox('Spät gekommen', value=True)
        exams_checkbox = streamlit.checkbox('PK Ergebnisse', value=True)
        # Choose between interpolated or normal plots
        plots_options_radio = streamlit.radio("Typ von Plots", ("Normal", "Interpolated"))
        #plots_options_calendar = streamlit.checkbox('Calendar view')

        # get plot with student info
        fig, ax = matplotlib.pyplot.subplots()
        fig.set_size_inches(12, 8)

        # Plot options
        plot_options = student_plot_data_options(df,df_pk, cfg, student_id,start_date,end_date)
        number_of_values, values_range, aufm, vers, fun, dates, late_arrival,nicht_dabei_idx, height, pk_idx, pk_actual_points,pk_guessed_points = plot_options
        # fill missing values (nan) with mean
        missing_values_aufm = missing_values_plot(aufm)
        missing_values_vers = missing_values_plot(vers)
        missing_values_fun = missing_values_plot(fun)
        # Selected lines to plot
        if exams_checkbox:
            #ax.bar(pk_idx, numpy.ones(len(pk_idx)) * 100, color='white', edgecolor='green', zorder=-1)
            ax.bar(pk_idx, pk_actual_points,width=0.6,alpha=.7,color='aqua',label='Erreichte Punkte')#, zorder=-1)
            ax.bar(pk_idx, pk_guessed_points,width=0.6,alpha=0.7 ,fill=False,hatch='\\\\',label='Geschätzte Punkte')#, zorder=-1)  # '..'
        if absent_checkbox:
            ax.bar(nicht_dabei_idx, height, alpha=1, color='white', width=1)
            ax.bar(nicht_dabei_idx, height, alpha=.2, color='red', width=1,label='Nicht anwesend')
        if aufm_checkbox:
            if plots_options_radio == "Interpolated":
                x_aufm_ip, y_aufm_ip = interpolation_values(number_of_values, aufm)
                ax.plot(x_aufm_ip, 100 * y_aufm_ip,color='orange', linewidth=4,
                                       label='Aufmerksamkeit interpolated', linestyle='-.', zorder=-1)
            else:
                ax.plot(values_range, 100 * aufm, linewidth=4,
                                       label='Aufmerksamkeit', linestyle='-', color='orange', zorder=-1)
                ax.plot(values_range, 100 * missing_values_aufm, linewidth=4,
                        linestyle=':', color='orange', zorder=1)
        if vers_checkbox:
            if plots_options_radio == "Interpolated":
                x_vers_ip, y_vers_ip = interpolation_values(number_of_values, vers)
                ax.plot(x_vers_ip, 100 * y_vers_ip, color='green', linewidth=4,
                                       label='Verständnis interpolated', linestyle='-.', zorder=-1)
            else:
                ax.plot(values_range, 100 * vers, linewidth=4,
                                       label='Verständnis', linestyle='-', color='green', zorder=-1)
                ax.plot(values_range, 100 * missing_values_vers, linewidth=4,
                        linestyle=':', color='green', zorder=1)
        if fun_checkbox:
            if plots_options_radio == "Interpolated":
                x_fun_ip, y_fun_ip = interpolation_values(number_of_values, fun)
                ax.plot(x_fun_ip, 100 * y_fun_ip, 'k-.', linewidth=4,
                                       label='Fun interpolated', linestyle='-.', zorder=-1)
            else:
                ax.plot(values_range, 100 * fun, linewidth=4,
                                       label='Fun', linestyle='-', color='blue', zorder=-1)
                ax.plot(values_range, 100 * missing_values_fun, linewidth=4,
                        linestyle=':', color='blue', zorder=1)
        if late_checkbox:
            ax.vlines(x=late_arrival,ymin=0,ymax=100, color='red', label='Zu spät',linestyles='-.')

        matplotlib.pyplot.legend(fontsize=14,loc='lower left')
        matplotlib.pyplot.title(f'Die Informationen über {student_id}', fontsize=14)
        matplotlib.pyplot.xlabel('Datum', fontsize=14)
        matplotlib.pyplot.ylabel('Prozent', fontsize=14)
        matplotlib.pyplot.xticks(values_range,labels=dates, fontsize=14, rotation=45)

        matplotlib.pyplot.yticks(100 * numpy.linspace(0, 1, 5), fontsize=14)
        matplotlib.pyplot.subplots_adjust(bottom=0.2)
        matplotlib.pyplot.grid(linewidth=.4)
        streamlit.write(values_range)
        # matplotlib.pyplot.show()

    with col2:
        streamlit.pyplot(fig)

# Tables
@streamlit.cache
def student_data_table(df, id, start_date, end_date):
    student_data = df[df['MiB-ID'] == id].drop(columns=['MiB-ID'])
    dates = student_data['Datum-df']
    dates = pandas.to_datetime(dates, format='%d/%m/%y')
    dates = dates.dt.date.sort_values()
    dates = dates.loc[(start_date <= dates) & (end_date >= dates)]
    dates = pandas.to_datetime(dates)
    dates = dates.dt.strftime('%d/%m/%y')
    student_data = student_data.loc[student_data['Datum-df'].isin(dates)]
    student_data = student_data.set_index('Datum-df')
    return student_data


@streamlit.cache
def kurs_plot_data_options(df,cfg,student_id,start_date,end_date):
    kurs_data = df[cfg['plots']['kurse']['fields']]
    values_name = kurs_data[kurs_data['MiB-Kurs-Name'] == student_id]
    dates = values_name['Datum-df']
    dates = pandas.to_datetime(dates, format='%d/%m/%y')
    dates = dates.dt.date.sort_values()
    dates = dates.loc[(start_date <= dates) & (end_date >= dates)]
    dates = pandas.to_datetime(dates)
    dates = dates.dt.strftime('%d/%m/%y')
    values_name = values_name.loc[values_name['Datum-df'].isin(dates)]
    number_of_values = values_name.shape[0]
    values_range = numpy.linspace(0, number_of_values - 1, num=number_of_values)
    anw = values_name['Anwesenheit Rollup (from Studenten x Termine)']
    total_students = values_name['Studentenanzahl (from Studenten x Termine)']
    anw_percent = anw / total_students
    aufm = values_name['Aufmerksamkeit Mittel'] / (numpy.ones(number_of_values) * 5)
    vers = values_name['Verständnis Mittel'] / (numpy.ones(number_of_values) * 5)
    fun = values_name['Fun Mittel'] / (numpy.ones(number_of_values) * 10)
    return values_range, anw_percent, aufm, vers, fun, dates


def plot_kurs_data(df, cfg, student_id,start_date,end_date):

    # define columns in dashboard
    col1, col2 = streamlit.columns((1.5, 5))

    # Line selection using checkboxes
    with col1:
        aufm_checkbox = streamlit.checkbox('Aufmerksamkeit', value=True)
        vers_checkbox = streamlit.checkbox('Verständnis', value=True)
        fun_checkbox = streamlit.checkbox('Fun Faktor', value=True)
        anw_checkbox = streamlit.checkbox('Anwesenheit', value=True)

        fig,ax = matplotlib.pyplot.subplots()
        fig.set_size_inches(16,8)
        # plot options
        values_range, anw_percent, aufm, vers, fun, dates = kurs_plot_data_options(df,cfg,student_id
                                                                                   ,start_date,end_date)
        if aufm_checkbox:
            ax.plot(values_range,100*aufm,label='Aufmerksamkeit',linewidth=4,
                    linestyle='-',color='orange',zorder=-1)
        if vers_checkbox:
            ax.plot(values_range,100*vers,label='Verständnis',linewidth=4,
                    linestyle='-',color='green',zorder=-1)
        if fun_checkbox:
            ax.plot(values_range,100*fun,label='Fun',linewidth=4,
                    linestyle='-',color='blue',zorder=-1)
        if anw_checkbox:
            ax.plot(values_range,100*anw_percent,label='Anwesenheit',linewidth=4,
                    linestyle='--',color='black')

        matplotlib.pyplot.legend(fontsize=14)
        matplotlib.pyplot.title(f'Die Informationen über {student_id}',fontsize=14)
        matplotlib.pyplot.xlabel('Datum',fontsize=14)
        matplotlib.pyplot.ylabel('Prozent',fontsize=14)
        matplotlib.pyplot.xticks(values_range,labels=dates,fontsize=14,rotation=45)
        matplotlib.pyplot.yticks(100*numpy.linspace(0,1,11),fontsize=14)
        matplotlib.pyplot.grid(linewidth=.4)
        # matplotlib.pyplot.show()

        # Show figure in Dashboard
        with col2:
            streamlit.pyplot(fig)


# Used for the variables wich are going to be prefill using JS
def get_param(param_name):
    query_params = streamlit.experimental_get_query_params()
    try:
        return query_params[param_name][0]
    except:
        streamlit.write('Parameters is missing')
        return False


def get_params(params_names_list):
    query_params = streamlit.experimental_get_query_params()
    responses = []
    for parameter in params_names_list:
        try:
            responses.append(query_params[parameter][0])
        except Exception as e:
            responses.append(None)
    return responses