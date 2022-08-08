import streamlit

from utils import *
def main():
    # read the configuration file and initialize random generators
    cfg = read_config('/app/mib_studenten_analyse/app/config/config.yaml')

    # read dataframes
    df_termine, df_studenten, df_studentenxtermine, df_pk_stud = generate_dataframes(cfg)

    # Kurs Info
    # Kurs dataframe
    all_dates_kurse = df_termine[cfg['plots']['kurse']['fields']]

    # All courses
    all_courses_names = df_termine['MiB-Kurs-Name'].unique()

    # Studenten ID
    mib_id = streamlit.text_input('Student', value=str(get_param('mib_id')))

    # Options to select o sidebar
    kurs_id = df_studentenxtermine[df_studentenxtermine['MiB-ID']==mib_id]['MiB-Kurs-Name']
    kurs_id = kurs_id.unique()
    streamlit.write(kurs_id)
    # Select value range for courses
    streamlit.dataframe(all_dates_kurse['MiB-Kurs-Name'])
    streamlit.write(type(kurs_id[0]))
    course_dates = all_dates_kurse[all_dates_kurse['MiB-Kurs-Name'] == kurs_id[0]]
    course_dates = course_dates['Datum-df']
    min_date_kurs, max_date_kurs = min_max_dates(course_dates)

    col_date_1, col_date_2 = streamlit.columns(2)
    with col_date_1:
        start_date = streamlit.date_input('Anfangsdatum', min_value=min_date_kurs,
                                            max_value=max_date_kurs, value=min_date_kurs)
    with col_date_2:
        end_date = streamlit.date_input('Enddatum', min_value=min_date_kurs,
                                            max_value=max_date_kurs, value=max_date_kurs)

    plot_student_data(df_studentenxtermine,df_pk_stud, cfg, mib_id, start_date, end_date)


if __name__ == "__main__":
    main()

