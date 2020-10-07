import pandas as pd
import numpy as np
import string 
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime as dt
from string import ascii_letters


#DATOS CALIDAD DE AIRE ANUALES: Calidad del aire. Datos diarios años 2001 a 2020, https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=aecb88a7e2b73410VgnVCM2000000c205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default
# url = 'data2020.csv'
# df = pd.read_csv(url, sep = ';')

# Contamination values are casted to float
def day_data_to_float(df):
    for column in df.columns:
        if ('D' in column and column != 'MAGNITUD'):
            # print(type(df[column].to_list()[0]))
            to_float = lambda x: float(x[:].replace(',','.'))
            df[column] = df[column].apply(to_float)
    return df

# The contamination values labeled with letter "N" are not valid so they were replaced
#by nan
def valid_data(df):    
    num_colum = 0
    for column in df.columns:  
        if column[0] == 'D':
            for row in range(len(df[column])):    
                if df.iloc[row,num_colum+1] == 'N':
                    df.iloc[row,num_colum] = np.nan
        num_colum += 1
    return df
#localitation of columns with contamination values and columns with validation labels   
def position_d_y_v(df):
    pos_colum = 0
    d_columns = []
    v_columns = []
    for column in df.columns:
        if column[0] == 'D':
            d_columns.append(pos_colum)
        elif column[0] == 'V':
            v_columns.append(pos_colum)
        pos_colum += 1
    
    return d_columns
# d_columns = position_d_y_v(df)

# Transformation of the df to time series df: date - contamination value 
def days_in_one_column(df,d_columns):
    # d_columns = position_d_y_v(df)
    #id_vars: los que se mantienen
    #value_vars: los que se modifican
    df = pd.melt(df, id_vars=df.columns[:7],value_vars =   df.columns[d_columns], var_name="DIA", value_name="Value").dropna()
    return df
        
# df = df.dropna()

# Creation of datetime column with date column
def creo_fecha(df):
    #me quedo con el numero D01: 01
    df['DIA_NUM'] = df['DIA'].apply(lambda x: x[1:])
    #columna mes con dos digitos
    df['MES_0'] = df['MES'].apply(lambda x: '%.2d'%x)
    #hago una columna FECHA con el siguiente formato año-mes-dia:
    df['FECHA'] = df['ANO'].map(str)+'-'+df['MES_0'].map(str)+'-'+df['DIA_NUM'].map(str)
    #lo paso a datetime
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    #para representarlo en una grafica:
    df["FECHA_str"] = df["FECHA"].dt.strftime('%Y-%m-%d')
    return df

# Label each measuring station with the district in which it's located
def column_distrito(df):

    distrito = pd.read_csv('estacion_distrito.csv', sep =';')
    dic_distrito = distrito.set_index('ESTACION')['DISTRITO'].to_dict()
        #creamos un diccionario estacion - distrito a partir de los datos del csv "distrito".
    #En el for nos recorremos df, la columna ESTACIÓN y accedemos al diccionario donde estacion es la clave y distrito es el calor
    distri = []
    distri_name = []
    for i in df['ESTACION']:
        #almacenamos los valores en la serie distrito que añadimos al df
        distri.append(dic_distrito[i])

    df['distrito'] = distri
    distrito_name = pd.read_csv('numDistrito_nombreDistrito.csv', sep =';')
    dic_distrito_name = distrito_name.set_index('num')['nombre'].to_dict()
    distri_name = []
   
    for d in df['distrito']:
        distri_name.append(dic_distrito_name[d])
       
    df['distrito_name'] = distri_name
    return df
# End os the df cleaning + aggreagtion of district column

# daily_airmad is a fuction that calls the cleaning fuctions of the df  
def daily_airmad(url):
    #es una función que llama a todas las anteriores
    df = pd.read_csv(url, sep = ';')
    if not (isinstance(df.D01[5], float)):   
        df = day_data_to_float(df)
    df = valid_data(df)
    d_columns = position_d_y_v(df)
    df =days_in_one_column(df,d_columns)
    df = df.dropna()
    df = creo_fecha(df)
    df = column_distrito(df)
    #este df tiene la fecha, el distrito, la magnitud a la que se refiere y los values de contaminacion
    return df
# generalization of daily_airmad
def daily_airmad_varios(list_url):
    dfs = []
    for url in list_url:
        df = daily_airmad(url)
        dfs.append(df)
    
    concat_dfs = pd.concat(dfs, axis = 0).reset_index()
    return concat_dfs
#mapa_por_distritos plots a map with contamination values in each district
def mapa_por_distritos(url):
    df2 = daily_airmad_mu_m3(url)
    df_magnitudes_selected = df2[(df2.MAGNITUD==1) | (df2.MAGNITUD==6)| (df2.MAGNITUD==7)| (df2.MAGNITUD==8)| (df2.MAGNITUD==9)| (df2.MAGNITUD==12)| (df2.MAGNITUD==14)]
    maping =df_magnitudes_selected.groupby(['FECHA','MAGNITUD', 'distrito']).mean().reset_index()
    maping.distrito = maping.distrito.apply(lambda x: '%02d' % (x,))
    print('maping:', maping)
    from urllib.request import urlopen
    import json
    import plotly.express as px
    #with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    #    counties = json.load(response)

    file = open('distrito_geojson.geojson')
    distritos = json.load(file)
    fig = px.choropleth_mapbox(maping, geojson=distritos, locations='distrito', color='MAGNITUD',
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

#we can select the data to plot a map with contamination in each district
def mapa_por_distritos_periodos(url):
    df2 = daily_airmad_mu_m3(url)
    df_magnitudes_selected = df2[(df2.MAGNITUD==1) | (df2.MAGNITUD==6)| (df2.MAGNITUD==7)| (df2.MAGNITUD==8)| (df2.MAGNITUD==9)| (df2.MAGNITUD==12)| (df2.MAGNITUD==14)]
    df_magnitudes_selected.distrito = df_magnitudes_selected.distrito.apply(lambda x: '%02d' % (x,))
    df_magnitudes_selected = df_magnitudes_selected.sort_values(by= 'FECHA')
    
    
    mask_antesq = (df_magnitudes_selected.FECHA >= '2020-01-01')&(df_magnitudes_selected.FECHA < '2020-03-15')
    mask_despues = (df_magnitudes_selected.FECHA >= '2020-03-15')
    
    
    maping_antes =df_magnitudes_selected[mask_antesq].groupby(['FECHA','MAGNITUD', 'distrito']).mean().reset_index()  
    maping_despues =df_magnitudes_selected[mask_despues].groupby(['FECHA','MAGNITUD', 'distrito']).mean().reset_index()  
#     print('maping:', maping)

    from urllib.request import urlopen
    import json
    import plotly.express as px
    #with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    #    counties = json.load(response)

    file = open('distrito_geojson.geojson')
    distritos = json.load(file)
    print('BEFORE QUARANTINE 2020')
#     print('ANTES',maping_antes.FECHA.tail())
    fig = px.choropleth_mapbox(maping_antes, geojson=distritos, locations='distrito', color='MAGNITUD',range_color=(0,140),
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
    print('AFTER QUARANTINE 2020')
#     print('DESPUÉS',maping_despues.FECHA)
    fig = px.choropleth_mapbox(maping_despues, geojson=distritos, locations='distrito', color='MAGNITUD',range_color=(0,140),
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()


# contaminantes: average contamination per measurement station
def contaminantes(df):
    df = df.groupby(['FECHA','MAGNITUD']).mean()[['FECHA', 'MAGNITUD','Value']]
    fig, ax = plt.subplots(figsize = (10,5))
    for magnitud in df.MAGNITUD.unique():
        df_mag = df[df.MAGNITUD == magnitud]
        resampled = df_mag.Value.resample('D').mean()
        x = resampled.index.strftime('%Y-%m-%d')
        ax.plot(resampled, label = str(magnitud), marker ='o')
        plt.xticks(rotation = 90)
        years_fmt = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(years_fmt)
        ax.legend(loc = 'upper right')
        every_nth = 1
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
    return df

# The different contaminations are separated into columns
def one_column_per_element(df):

    df=df.reset_index().pivot_table(values='Value', index='FECHA', columns='MAGNITUD')
    #LOS DIFERENTES VALORES DE MAGNITUD SE DESPIEGAN si hay alguna repeticion pivot table por defecto tiene aggfunc el mean de esos valores. en este caso hay valores de magnitud en la misma fecha para distintas estaciones, hace la media. No interesaría si queremos ver cosas por distritos/zonas.
    #cambio los nombres de las columnas.
    # 1. creo un df de traduccion, 
    df_traduccion = 'mag_translator.csv'
    mag_tr = pd.read_csv(df_traduccion, sep = ';')
    #2.a partir de ahi creo un diccionario numero: nombre_del_elemento 
    #2.1 se quitan tildes (en el propio df) y espacios(con la funcion lambda)
    mag_tr['mag_nombre_j'] = mag_tr['mag_nombre'].apply(lambda a: a.replace(" ","")) #quito espacios
    dictionary = mag_tr.set_index('MAGNITUD')['mag_nombre_j'].to_dict()
    #y 3. luego cambio el nombre de las columnas
    df = df.rename(columns = dictionary)
    return df

# list of cleaned df. This fuctions is created to contac all df's.
def list_of_dfs(list_url):
    list_df = []
    list_y = []
    for url in list_url:
        df = daily_airmad(url)
        df = one_column_per_element(df)
        list_df.append(df)
    return list_df

# plotting the different contamination levels
def ploteo_one_element_per_column_general(list_url):
    #plot todos los elementos de cada df por separado
    list_df= list_of_dfs(list_url)
    #df -> dado por  daily_airmad+one_column_per_element
    for df in list_df:
        df.plot(figsize = (10,5), marker ='o')
        plt.legend(loc='best',bbox_to_anchor=(1.0, 0.9))
        title = df.index.year[0]
        plt.title(title)

# Ploting contamination level in 5 different districts over the years
def heatplot_distritos_general(url):
    sns.set()
    df2 = daily_airmad_mu_m3(url)
#     df_magnitudes_selected = df[(df.MAGNITUD!=6) & (df.MAGNITUD!=42)& (df.MAGNITUD!=44)& (df.MAGNITUD!=44)]
     #dioxido de azufre, monoxido de nitrogeno, dioxido de nitrogeno, particulas <2.5micras, partículas < 10 micras, óxidos de nitrógeno, ozono, tolueno, benceno, etilbenceno, metalxileno, paraxileno, ortoxileno (microg/m3)

    #se excluyen = 6- monóxido de carbono, 42-hidrocarburos totales, 43-metano e 44-hidrocarburos no meránicos(hexano)
#     df_magnitudes_selected = df2[(df2.MAGNITUD==6) | (df2.MAGNITUD==42)| (df2.MAGNITUD==44)| (df2.MAGNITUD==44)]
    
    df_magnitudes_selected = df2[(df2.MAGNITUD==1) | (df2.MAGNITUD==6)| (df2.MAGNITUD==7)| (df2.MAGNITUD==8)| (df2.MAGNITUD==9)| (df2.MAGNITUD==12)| (df2.MAGNITUD==14)]
    df_distritos_selected = df_magnitudes_selected[(df_magnitudes_selected.distrito == 1) | (df_magnitudes_selected.distrito == 4) | (df_magnitudes_selected.distrito == 9) | (df_magnitudes_selected.distrito == 13) | (df_magnitudes_selected.distrito == 21)]
    mapeo = df_distritos_selected.groupby(['FECHA','distrito_name']).mean().reset_index()[['FECHA','Value','distrito_name']]

    heatplot = mapeo.pivot('distrito_name','FECHA','Value') 
    plt.subplots(figsize=(30,10))
    ax = sns.set(font_scale=2)
    # ax = sns.set()
    ax = sns.heatmap(heatplot, cmap="RdPu", annot_kws={'size':12})   
    ax.set_xlabel('FECHA', fontsize = 18)
    ax.set_ylabel('Distritos', fontsize = 18)
    ax.set_xticks(range(len(heatplot.columns)))
    ax.set_xticklabels(heatplot.columns.strftime('%Y - %m - %d'),fontsize =20)
    plt.yticks(rotation=0, fontsize =20) 

    
    every_nth = 7
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:

            label.set_visible(False)
    plt.show()
  
# contamination levels in the different months and different years  
def bars_for_years(list_urls):

    list_test2=[]

    for url in list_urls:
        df2 = daily_airmad(url)
        df_magnitudes_selected = df2[(df2.MAGNITUD!=6) & (df2.MAGNITUD!=42)& (df2.MAGNITUD!=44)& (df2.MAGNITUD!=44)]
           #dioxido de azufre, monoxido de nitrogeno, dioxido de nitrogeno, particulas <2.5micras, partículas < 10 micras, óxidos de nitrógeno, ozono, tolueno, benceno, etilbenceno, metalxileno, paraxileno, ortoxileno (microg/m3)

    #se excluyen = 6- monóxido de carbono, 42-hidrocarburos totales, 43-metano e 44-hidrocarburos no meránicos(hexano)
        # print(df_magnitudes_selected.head(16))
        test2 = df_magnitudes_selected.resample('SMS', on = 'FECHA').sum().reset_index()[['FECHA', 'Value']]
        #nuevas columnas sacadas de datatime
        test2['Year'] = test2.FECHA.dt.year
        test2['mes'] = test2.FECHA.dt.month
        test2['day'] = test2.FECHA.dt.day
        test2["Date"] = test2["FECHA"].dt.strftime('%m - %d')

        list_test2.append(test2) 

    concatenacion_dfs = pd.concat(list_test2, axis = 0).reset_index()

    hmap_m=concatenacion_dfs.pivot_table(values='Value', index='Date', columns='Year')
    hmap_m.plot(kind='bar', figsize = (14,5), align='edge')
    plt.yticks([])
    plt.xticks(rotation = 45)
    plt.ylabel('Contamination (AU)')
    plt.title('Sulphur dioxide, nitrogen monoxide, nitrogen dioxide, microparticles, nitrogen oxides, ozone,\n toluene, benzene, ethylbenzene, metalxylene, paraxylene, orthoxylene contamination levels in Madrid')
    
    
def bars_for_years_2(list_urls):
    # 6- monóxido de carbono, 42-hidrocarburos totales, 43-metano e 44-hidrocarburos no meránicos(hexano)
    #SE EXPLUYEN: dioxido de azufre, monoxido de nitrogeno, dioxido de nitrogeno, particulas <2.5micras, partículas < 10 micras, óxidos de nitrógeno, ozono, tolueno, benceno, etilbenceno, metalxileno, paraxileno, ortoxileno (microg/m3)
    list_test2=[]

    for url in list_urls:
        df2 = daily_airmad(url)
        df_magnitudes_selected = df2[(df2.MAGNITUD==6) | (df2.MAGNITUD==42)| (df2.MAGNITUD==44)| (df2.MAGNITUD==44)]
        # print(df_magnitudes_selected.head(16))
        test2 = df_magnitudes_selected.resample('SMS', on = 'FECHA').sum().reset_index()[['FECHA', 'Value']]
        #nuevas columnas sacadas de datatime
        test2['Year'] = test2.FECHA.dt.year
        test2['mes'] = test2.FECHA.dt.month
        test2['day'] = test2.FECHA.dt.day
        test2["Date"] = test2["FECHA"].dt.strftime('%m - %d')

        list_test2.append(test2) 

    concatenacion_dfs = pd.concat(list_test2, axis = 0).reset_index()

    hmap_m=concatenacion_dfs.pivot_table(values='Value', index='Date', columns='Year')
    hmap_m.plot(kind='bar', figsize = (14,5), align='edge')
    plt.yticks([])
    plt.ylabel('Contamination (AU)')
    plt.title('Carbon monoxide, Total hydrocarbons, Methane and hexane contamination levels in Madrid)')

def heat_map_years(list_urls):
        #se INcluyen = 6- monóxido de carbono, 42-hidrocarburos totales, 43-metano e 44-hidrocarburos no meránicos(hexano) (mg/m3)'''
    list_test2=[]
    
    for url in list_urls:
        df2 = daily_airmad(url)
#         df_magnitudes_selected = df2[(df2.MAGNITUD==6) | (df2.MAGNITUD==42)| (df2.MAGNITUD==44)| (df2.MAGNITUD==44)]
        df_magnitudes_selected = df2[(df2.MAGNITUD==1) | (df2.MAGNITUD==6)| (df2.MAGNITUD==7)| (df2.MAGNITUD==8)| (df2.MAGNITUD==9)| (df2.MAGNITUD==12)| (df2.MAGNITUD==14)]
        # print(df_magnitudes_selected.head(16))
        test2 = df_magnitudes_selected.resample('M', on = 'FECHA').mean().reset_index()[['FECHA', 'Value']]
        #nuevas columnas sacadas de datatime
        test2['Year'] = test2.FECHA.dt.year
        test2['mes'] = test2.FECHA.dt.strftime('%B')
        test2['day'] = test2.FECHA.dt.day
        test2["Date"] = test2["FECHA"].dt.strftime('%m - %d')

        list_test2.append(test2)
    
    concatenacion_dfs = pd.concat(list_test2, axis = 0).reset_index()
    hmap =  concatenacion_dfs.pivot('mes','Year','Value')
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    hmap = hmap.reindex(months)
    sns.set()
    ax = sns.heatmap(hmap,cmap="YlGnBu",cbar_kws={'label': 'Contamination (AU)'})
    ax.set_ylabel('')
    



'''FUNCIONES AEMET TIEMPO EN RETIRO Y BARAJAS'''
def merge_json(list_json):
    dfs = []
    selected = ['fecha','nombre','tmed','tmin','horatmin','tmax','horatmax','presMax','horaPresMax','presMin','horaPresMin']
    for json in list_json:
        clima = pd.read_json(json)
        dfs.append(clima)
    df = dfs[0]
    for pos in range(len(dfs)-1):  
        df = df.merge(dfs[pos+1],how = 'outer',on ='fecha' )
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df
def concat_aemet(list_json):
    list_dfs = []
    for json in list_json:
        clima = pd.read_json(json, dtype = True)
        list_dfs.append(clima)
    df = pd.concat(list_dfs, axis =0, sort = True)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.sort_values(by = 'fecha', inplace = True)
    selected = ['fecha','nombre','tmed','tmin','horatmin','tmax','horatmax','presMax','horaPresMax','presMin','horaPresMin']
    df = df[selected].set_index('fecha')
    dic_nombre_distrito = {'MADRID, RETIRO':'Retiro', 'MADRID AEROPUERTO':'Barajas'}
    distri_name = []
    for d in df['nombre']:
        distri_name.append(dic_nombre_distrito[d])
    df['distrito_name'] = distri_name
#     nuevo:
    df = df.reset_index().rename(columns = {'fecha': 'FECHA'})
    df = serie_to_float(df)
    return df

def concat_aemet_adapt(list_json,selected = ['fecha','nombre','tmed','tmin','horatmin','tmax','horatmax','presMax','horaPresMax','presMin','horaPresMin','racha','velmedia']):
    list_dfs = []
    for json in list_json:
        clima = pd.read_json(json, dtype = True)
        list_dfs.append(clima)
    df = pd.concat(list_dfs, axis =0, sort = True)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.sort_values(by = 'fecha', inplace = True)
#     selected = ['fecha','nombre','tmed','tmin','horatmin','tmax','horatmax','presMax','horaPresMax','presMin','horaPresMin','racha','velmedia']
    df = df[selected].set_index('fecha')
    dic_nombre_distrito = {'MADRID, RETIRO':'Retiro', 'MADRID AEROPUERTO':'Barajas'}
    distri_name = []
    for d in df['nombre']:
        distri_name.append(dic_nombre_distrito[d])
    df['distrito_name'] = distri_name
#     nuevo:
    df = df.reset_index().rename(columns = {'fecha': 'FECHA'})
    df = serie_to_float(df)
    return df
def one_column_per_element_and_distrito(df):

    df=df.reset_index().pivot_table(values=['Value'], index=['FECHA','distrito_name'], columns=['MAGNITUD'])
    #LOS DIFERENTES VALORES DE MAGNITUD SE DESPIEGAN si hay alguna repeticion pivot table por defecto tiene aggfunc el mean de esos valores. en este caso hay valores de magnitud en la misma fecha para distintas estaciones, hace la media. No interesaría si queremos ver cosas por distritos/zonas.
    #cambio los nombres de las columnas.
    # 1. creo un df de traduccion, 
    df_traduccion = 'mag_translator.csv'
    mag_tr = pd.read_csv(df_traduccion, sep = ';')
    #2.a partir de ahi creo un diccionario numero: nombre_del_elemento 
    #2.1 se quitan tildes (en el propio df) y espacios(con la funcion lambda)
    mag_tr['mag_nombre_j'] = mag_tr['mag_nombre'].apply(lambda a: a.replace(" ","")) #quito espacios
    dictionary = mag_tr.set_index('MAGNITUD')['mag_nombre_j'].to_dict()
    #y 3. luego cambio el nombre de las columnas
    df = df.rename(columns = dictionary)
    return df

def serie_to_float(df):
    quitar = ['fecha','indicativo','nombre','provincia','altitud','dir','distrito_name']
    for column in df.columns:
        if ((type(df[column][0]) == str) and (column not in quitar) and (not column.startswith('h'))):
#             print(column)
            df[column] = df[column].str.replace(',','.').astype(float) 
#         if column.startswith('t') or column.startswith('p'):           
#             df[column] = df[column].str.replace(',','.').astype(float)      
    return df

def newcolumn_from_dic(df,csv_todic, name_column_clave, name_column_valor, new_name_column):
    #     df -> df al que quiero añadirle una nueva columna con los valores de un dic
    # csv_todic -> pares clave, valor: csv con el que haré un df que contiene los pares clave, valor
    # name_column_clave, name_column_valor = df de traducción
    # new_name_column -> en el df matriz

    df_todic = pd.read_csv(csv_todic, sep =';')
    df_todic[name_column_valor] = df_todic[name_column_valor].apply(lambda a: a.replace(" ","")) #quito espacios
    
    dic = df_todic.set_index(name_column_clave)[name_column_valor].to_dict()
        #creamos un diccionario estacion - distrito a partir de los datos del csv "distrito".
    #En el for nos recorremos df, la columna ESTACIÓN y accedemos al diccionario donde estacion es la clave y distrito es el calor
    lista_valores = []
    #     name_column_clave es igual en el df que en el df de traduccion que convierto a dic
    for i in df[name_column_clave]:
        #almacenamos los valores en la serie distrito que añadimos al df
        lista_valores.append(dic[i])

    df[new_name_column] = lista_valores
    
    return df

def contamina_and_clima(list_csv,list_json): 
    #     list_csv = ['data2018.csv','data2019.csv','datos202003.csv']
    '''list_json = ['climamad.json',
                  'barajas_2010_2015.json',
    'barajas_2015_2020.json',
    'primeros_2010_barajas.json',
    'primeros_2010_retiro.json',
    'retiro_2010_2015.json']'''

    df1 = daily_airmad_varios(list_csv)
    df1 = newcolumn_from_dic(df1,'mag_translator.csv', 'MAGNITUD', 'mag_nombre', 'nombre_magnitud')

#     df2 = concat_aemet(list_json)
    df2 = concat_aemet_adapt(list_json)
    union = df1.merge(df2, on = ['FECHA','distrito_name'])
    return union

def heat_map_years_aemet(list_json):
        #se INcluyen = 6- monóxido de carbono, 42-hidrocarburos totales, 43-metano e 44-hidrocarburos no meránicos(hexano) (mg/m3)'''
    df = concat_aemet(list_json).resample('M', on = 'FECHA').mean().reset_index()
    df['Year'] = df.FECHA.dt.year
    df['mes'] = df.FECHA.dt.strftime('%B')
    hmap =  df.pivot('mes','Year','tmax')
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    hmap = hmap.reindex(months)
    sns.set()
    ax = sns.heatmap(hmap,cmap="YlGnBu",cbar_kws={'label': 'Tª máxima (\xb0 C)'})
    ax.set_ylabel('')

def corrheat(list_csv, list_json):
    union = contamina_and_clima(list_csv,list_json)[['Value','tmed','tmax','tmin','presMin','presMax']]
    
    corr = union.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(240, 10, n=9),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
def contaminantes_barplot(url, co = True):
#     df = daily_airmad_varios([url])
    df =daily_airmad_mu_m3(url)
    df_traduccion = 'mag_translator.csv'
    mag_tr = pd.read_csv(df_traduccion, sep = ';')
    #2.a partir de ahi creo un diccionario numero: nombre_del_elemento 
    #2.1 se quitan tildes (en el propio df) y espacios(con la funcion lambda)
    mag_tr['mag_nombre_j'] = mag_tr['mag_nombre'].apply(lambda a: a.replace(" ","")) #quito espacios
    dictionary = mag_tr.set_index('MAGNITUD')['mag_nombre_j'].to_dict()
    if co:
        df = df[(df.MAGNITUD == 1)|(df.MAGNITUD == 6)|(df.MAGNITUD == 9 )|(df.MAGNITUD == 12)|(df.MAGNITUD == 14)]
        labels = [dictionary[1],dictionary[6],dictionary[9],dictionary[12],dictionary[14]]
    else:
        df = df[(df.MAGNITUD == 1)|(df.MAGNITUD == 9 )|(df.MAGNITUD == 12)|(df.MAGNITUD == 14)]
        labels = [dictionary[1],dictionary[9],dictionary[12],dictionary[14],]
        
    lista_dfs = []
    for name, group in df.groupby('MAGNITUD'):
        name = str(name)
        df = group.resample('M', on = 'FECHA').mean()
        lista_dfs.append(df)
    concatenacion_dfs = pd.concat(lista_dfs, axis = 0).reset_index()
    concatenacion_dfs["Date"] = concatenacion_dfs["FECHA"].dt.strftime('%m')
    
    sns.set(style="ticks")
    
    sns.set(rc={'figure.figsize':(12,6)})
          
#     df_traduccion = 'mag_translator.csv'
#     mag_tr = pd.read_csv(df_traduccion, sep = ';')
#     #2.a partir de ahi creo un diccionario numero: nombre_del_elemento 
#     #2.1 se quitan tildes (en el propio df) y espacios(con la funcion lambda)
#     mag_tr['mag_nombre_j'] = mag_tr['mag_nombre'].apply(lambda a: a.replace(" ","")) #quito espacios
#     dictionary = mag_tr.set_index('MAGNITUD')['mag_nombre_j'].to_dict()
#     labels = [dictionary[6],dictionary[9],dictionary[12],dictionary[14],]
    ax = sns.barplot(x='Date', y="Value", hue='MAGNITUD', data=concatenacion_dfs)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, labels, title="Contaminantes")
    meses = ['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ax.set(xticklabels=meses)
    plt.ylabel('Contaminacion media μg/m3')
    plt.xticks(rotation=45)
    
#     plt.axhline(y=500, color='b', linestyle='-')
    plt.show()
# def contaminantes_barplot(url):
# #     df = daily_airmad_varios([url])
#     df =daily_airmad_mu_m3(url)
#     df = df[(df.MAGNITUD == 6)|(df.MAGNITUD == 9 )|(df.MAGNITUD == 12)|(df.MAGNITUD == 14)]
#     lista_dfs = []
#     for name, group in df.groupby('MAGNITUD'):
#         name = str(name)
#         df = group.resample('M', on = 'FECHA').mean()
#         lista_dfs.append(df)
#     concatenacion_dfs = pd.concat(lista_dfs, axis = 0).reset_index()
#     concatenacion_dfs["Date"] = concatenacion_dfs["FECHA"].dt.strftime('%b')
#     sns.set(style="whitegrid")
          
#     df_traduccion = 'mag_translator.csv'
#     mag_tr = pd.read_csv(df_traduccion, sep = ';')
#     #2.a partir de ahi creo un diccionario numero: nombre_del_elemento 
#     #2.1 se quitan tildes (en el propio df) y espacios(con la funcion lambda)
#     mag_tr['mag_nombre_j'] = mag_tr['mag_nombre'].apply(lambda a: a.replace(" ","")) #quito espacios
#     dictionary = mag_tr.set_index('MAGNITUD')['mag_nombre_j'].to_dict()
#     labels = [dictionary[6],dictionary[9],dictionary[12],dictionary[14],]
#     ax = sns.barplot(x='Date', y="Value", hue='MAGNITUD', data=concatenacion_dfs)
#     h, l = ax.get_legend_handles_labels()
#     ax.legend(h, labels, title="Contaminantes")
#     plt.xticks(rotation=45)

def daily_airmad_mu_m3(url):
    #es una función que llama a todas las anteriores
    df = pd.read_csv(url, sep = ';')
    if not (isinstance(df.D01[5], float)):   
        df = day_data_to_float(df)
    df = valid_data(df)
    d_columns = position_d_y_v(df)
    df =days_in_one_column(df,d_columns)
    df = df.dropna()
    df = creo_fecha(df)
    df = column_distrito(df)
    df.reset_index(inplace = True)
    for pos,valor in enumerate(df.MAGNITUD):
        if valor ==6:
            df.Value[pos] = df.Value[pos]*1000
    
    df_traduccion = 'mag_translator.csv'
    mag_tr = pd.read_csv(df_traduccion, sep = ';')
    #2.a partir de ahi creo un diccionario numero: nombre_del_elemento 
    #2.1 se quitan tildes (en el propio df) y espacios(con la funcion lambda)
    mag_tr['mag_nombre_j'] = mag_tr['mag_nombre'].apply(lambda a: a.replace(" ","")) #quito espacios
    dictionary = mag_tr.set_index('MAGNITUD')['mag_nombre_j'].to_dict()
    mag_name = []
    for mag in df.MAGNITUD:
        mag_name.append(dictionary[mag])      
    df['mag_name'] = mag_name
    
    #este df tiene la fecha, el distrito, la magnitud a la que se refiere y los values de contaminacion
    return df
def daily_airmad_varios_mu_m3(list_url):
    dfs = []
    for url in list_url:
        df = daily_airmad_mu_m3(url)
        dfs.append(df)
    
    concat_dfs = pd.concat(dfs, axis = 0).reset_index()
    return concat_dfs 
def bars_for_years_mu_m3(list_urls,titulo):

    list_test2=[]

    for url in list_urls:
        df2 = daily_airmad_mu_m3(url)
        
        df_magnitudes_selected = df2[(df2.MAGNITUD==1) | (df2.MAGNITUD==6)| (df2.MAGNITUD==7)| (df2.MAGNITUD==8)| (df2.MAGNITUD==9)| (df2.MAGNITUD==12)| (df2.MAGNITUD==14)]
           #dioxido de azufre, monoxido de nitrogeno, dioxido de nitrogeno, particulas <2.5micras, partículas < 10 micras, óxidos de nitrógeno, ozono, tolueno, benceno, etilbenceno, metalxileno, paraxileno, ortoxileno (microg/m3)

    #se excluyen = 6- monóxido de carbono, 42-hidrocarburos totales, 43-metano e 44-hidrocarburos 
#     no meránicos(hexano)
        # print(df_magnitudes_selected.head(16))
#         test2 = df_magnitudes_selected.resample('SMS', on = 'FECHA').sum().reset_index()[['FECHA', 'Value']]
        
        test2 = df_magnitudes_selected.resample('M', on = 'FECHA').sum().reset_index()[['FECHA', 'Value']]
        print('------------------------------------------------------------------')
        print('Mayor contaminación en el año {} es en el mes:'.format(test2.FECHA.dt.year[0]),test2.FECHA.loc[test2.Value.idxmax()].month)
        print('------------------------------------------------------------------')
        print('Menor contaminación:en el año {} es en el mes:'.format(test2.FECHA.dt.year[0]),test2.FECHA.loc[test2.Value.idxmin()].month)
        print('------------------------------------------------------------------')
        #nuevas columnas sacadas de datatime
        test2['Year'] = test2.FECHA.dt.year
        test2['mes'] = test2.FECHA.dt.month
        test2['day'] = test2.FECHA.dt.day
        test2["Date"] = test2["FECHA"].dt.strftime('%m')

        list_test2.append(test2) 
#         print(test2)

    concatenacion_dfs = pd.concat(list_test2, axis = 0).reset_index()
#     print(concatenacion_dfs)

    hmap_m=concatenacion_dfs.pivot_table(values='Value', index='Date', columns='Year')
   
    
    hmap_m.plot(kind='bar', figsize = (15,7), align='edge')
    plt.yticks([])
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    plt.xticks(rotation = 45)
    plt.ylabel('Contamination (AU)')
    plt.title(titulo)
    plt.xticks(np.arange(12), months, rotation=20)
    
def contamina_mu_m3_and_clima(list_csv,list_json): 
    #     list_csv = ['data2018.csv','data2019.csv','datos202003.csv']
    '''list_json = ['climamad.json',
                  'barajas_2010_2015.json',
    'barajas_2015_2020.json',
    'primeros_2010_barajas.json',
    'primeros_2010_retiro.json',
    'retiro_2010_2015.json']'''

    print('list_csv:', list_csv)
    df1 = daily_airmad_varios_mu_m3(list_csv)
    print('primero',df1)
    df1 = newcolumn_from_dic(df1,'mag_translator.csv', 'MAGNITUD', 'mag_nombre', 'nombre_magnitud')
    print('segundo',df1)

    df2 = concat_aemet(list_json)
    union = df1.merge(df2, on = ['FECHA','distrito_name'])
    print('Unión')
    return union
def barplot_temp(list_json, date1 = '2019-01-01', date2 = '2019-12-31', ano='2019'):
    selected = ['fecha','nombre','tmed','tmax','racha','velmedia']
    temp = concat_aemet_adapt(list_json,selected).resample('M', on = 'FECHA').mean().reset_index()
    temp.shape
    mask = (temp.FECHA >= date1) & (temp.FECHA <= date2)
    temp = temp[mask]
    
  
    sns.set(style="whitegrid")

    ax = sns.barplot(x='FECHA', y="tmax", data=temp)
    sns.set(rc={'figure.figsize':(6,9)})
    h, l = ax.get_legend_handles_labels()
    ax.legend(h,title="Tª max")
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ax.set(xticklabels=months)
    
#     plt.yticks([])
    
    plt.xticks(rotation = 90)
    plt.ylabel('Tmax (\N{DEGREE SIGN} C) ')
    plt.title('tmax')
    plt.xlabel(ano)
def mapa_por_distritos_periodos(url):
    df2 = daily_airmad_mu_m3(url)
    df_magnitudes_selected = df2[(df2.MAGNITUD==1) | (df2.MAGNITUD==6)| (df2.MAGNITUD==7)| (df2.MAGNITUD==8)| (df2.MAGNITUD==9)| (df2.MAGNITUD==12)| (df2.MAGNITUD==14)]
    df_magnitudes_selected.distrito = df_magnitudes_selected.distrito.apply(lambda x: '%02d' % (x,))
    df_magnitudes_selected = df_magnitudes_selected.sort_values(by= 'FECHA')
    
    
    mask_antesq = (df_magnitudes_selected.FECHA >= '2020-01-01')&(df_magnitudes_selected.FECHA < '2020-03-15')
    mask_despues = (df_magnitudes_selected.FECHA >= '2020-03-15')
    
    
    maping_antes =df_magnitudes_selected[mask_antesq].groupby(['distrito']).mean().reset_index()  
    maping_despues =df_magnitudes_selected[mask_despues].groupby(['distrito']).mean().reset_index()  
#     print('maping:', maping)

    from urllib.request import urlopen
    import json
    import plotly.express as px
    #with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    #    counties = json.load(response)

    file = open('distrito_geojson.geojson')
    distritos = json.load(file)
    print('BEFORE QUARANTINE 2020')
#     print('ANTES',maping_antes.FECHA.tail())
    fig = px.choropleth_mapbox(maping_antes, geojson=distritos, locations='distrito', color='Value',
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
    print('AFTER QUARANTINE 2020')
#     print('DESPUÉS',maping_despues.FECHA)
    fig = px.choropleth_mapbox(maping_despues, geojson=distritos, locations='distrito', color='Value',
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

#ahora hay diferentes opciones de presentación del df
#OPCIÓN1:
#se hace la media de las diferentes estaciones en las que se ha tomado la medida. pierdo info de las ESTACIONES.
# df1 = df.groupby(['FECHA','MAGNITUD']).mean()[['FECHA', 'MAGNITUD','Value']]

def mapa_por_distritos_periodos_varios(list_url):
    df2 = daily_airmad_varios_mu_m3(list_url)
    df_magnitudes_selected = df2[(df2.MAGNITUD==1) | (df2.MAGNITUD==6)| (df2.MAGNITUD==7)| (df2.MAGNITUD==8)| (df2.MAGNITUD==9)| (df2.MAGNITUD==12)| (df2.MAGNITUD==14)]
    df_magnitudes_selected.distrito = df_magnitudes_selected.distrito.apply(lambda x: '%02d' % (x,))
    df_magnitudes_selected = df_magnitudes_selected.sort_values(by= 'FECHA')
    
    
    mask_antesq = (df_magnitudes_selected.FECHA >= '2019-03-15')&(df_magnitudes_selected.FECHA < '2019-04-30')
    mask_despues = (df_magnitudes_selected.FECHA >= '2020-03-15')
    
    
    maping_antes =df_magnitudes_selected[mask_antesq].groupby(['distrito']).mean().reset_index()  
    maping_despues =df_magnitudes_selected[mask_despues].groupby(['distrito']).mean().reset_index()  
#     print('maping:', maping)
    contamina = pd.merge(left = maping_antes[['distrito','Value']],right =maping_despues[['distrito','Value']],left_index = True,right_index = True,how = 'outer',suffixes = ('antes','despues'))
    contamina['porcentaje_antes_despues'] = contamina['Valuedespues']*100/contamina['Valueantes']
#     print(contamina)
    print('--------------------------------------')
    print("\x1b[1;33m"+'% Contaminación respecto al año anterior',contamina['porcentaje_antes_despues'].describe())
    print('--------------------------------------')
    print('Distrito con minima caida es RETIRO', contamina.set_index('distritoantes').porcentaje_antes_despues.idxmax())
    print('--------------------------------------')
    print('Distrito con maxima caida es CHAMARTÍN', contamina.set_index('distritoantes').porcentaje_antes_despues.idxmin())
    print('--------------------------------------')

    from urllib.request import urlopen
    import json
    import plotly.express as px
    #with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    #    counties = json.load(response)

    file = open('distrito_geojson.geojson')
    distritos = json.load(file)
    
    print("\033[;36m"+'From 15th March to 30th April'+ '\033[4;35m'+ ' 2019'+'\033[0;m') 
#     print('ANTES',maping_antes.FECHA.tail())
    fig = px.choropleth_mapbox(maping_antes, geojson=distritos, locations='distrito', color='Value',range_color=(0, 130),
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    
    
#     print('From 15th March to 30th April'+ "\033[4;35;47m"+ '2020'+'\033[0;m') 
    print("\033[;36m"+'From 15th March to 30th April'+ '\033[4;35m' + ' 2020'+'\033[0;m') 
#     print('DESPUÉS',maping_despues.FECHA)
    fig = px.choropleth_mapbox(maping_despues, geojson=distritos, locations='distrito', color='Value',range_color=(0, 130),
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

#ahora hay diferentes opciones de presentación del df
#OPCIÓN1:
#se hace la media de las diferentes estaciones en las que se ha tomado la medida. pierdo info de las ESTACIONES.
# df1 = df.groupby(['FECHA','MAGNITUD']).mean()[['FECHA', 'MAGNITUD','Value']]

def media_period_quarantine(list_url, list_mag= [1,6,7,8,9,12,14]):
#     list_mag = [1,6,7,8,9,12,14]
    df2 = daily_airmad_varios_mu_m3(list_url)
    df_magnitudes_selected = df2[(df2.MAGNITUD==list_mag[0]) | (df2.MAGNITUD==list_mag[1])| (df2.MAGNITUD==list_mag[2])| (df2.MAGNITUD==list_mag[3])| (df2.MAGNITUD==list_mag[4])| (df2.MAGNITUD==list_mag[5])| (df2.MAGNITUD==list_mag[6])]
    df_magnitudes_selected.distrito = df_magnitudes_selected.distrito.apply(lambda x: '%02d' % (x,))
    df_magnitudes_selected = df_magnitudes_selected.sort_values(by= 'FECHA')
#     print(df_magnitudes_selected)
    df_magnitudes_selected.DIA_NUM = df_magnitudes_selected.DIA_NUM.astype('int64')
    
    mask_meses = (df_magnitudes_selected.MES>=3) & (df_magnitudes_selected.MES<=4)
    mask_dias = (df_magnitudes_selected.DIA_NUM>=15) & (df_magnitudes_selected.DIA_NUM<=30)
    mask_anos = None
    value_year = []
    year = []
    m = pd.DataFrame()
  
    for i in df_magnitudes_selected.ANO.unique():
        mask_anos = (df_magnitudes_selected.ANO == i)
        value_year.append(df_magnitudes_selected[mask_meses & mask_dias & mask_anos].Value.mean())
        year.append(i)
        
    m['year'] = year
#     m['year'] = pd.to_datetime(m['year'])
    m['value_year'] = value_year
#     sns.set(style="whitegrid")

#     ax = sns.barplot(x='year', y="value_year", data=m)
#     h, l = ax.get_legend_handles_labels()
#     ax.legend(h,year, title="Quarantine")
#     plt.xticks(rotation=90)
#     plt.ylabel('Contamination during quarantine period')
# -------------------------------------------------------------
    
#     import plotly.graph_objs as go
#     # create trace1 
#     trace1 = go.Bar(
#                     x = m.year,
#                     y = m.value_year,
#                     name = "Contamination during quarantine period 2",
#                     marker = dict(color = 'rgba(255, 174, 255, 0.5)',
#                                  line=dict(color='rgb(0,0,0)',width=1.5)))
#     data = [trace1]
# #     layout = go.Layout(barmode = "group")
# #     layout = {
# #   'xaxis': {'title': 'Years'},
# #   'barmode': 'relative',
# #   'title': 'Contamination during quarantine period 2'
# # };
#     layout = dict(title = 'Contamination during quarantine period',
#               xaxis= dict(title= 'Year',ticklen= 5,zeroline= False),
#               yaxis= dict(title= 'Contamination level',ticklen= 5,zeroline= False)
#              )
    

#     fig = go.Figure(data = data, layout = layout)
#     iplot(fig)
# -------------------------------------------------------------
    import plotly.express as px
    data = px.data.gapminder()

    df = m
    fig= px.bar(df, x='year', y='value_year',
                 hover_data=['value_year'], color='value_year',
                 labels={'value_year':'Contamination level'})
    fig.update_layout(xaxis_type='category',title={
        'text':'Contamination during Quarantine','y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_xaxes(tickangle=-90)
    
#     plt.xticks(rotation=-90)
#     plt.xticks(range(len(year)), year)
    fig.show()
#     return [data.year,m.year]

def corrheat_triangular(list_csv, list_json):
    sns.set(style="white")
#     union = contamina_and_clima(list_csv,list_json)[['Value','tmed','tmax','tmin','presMin','presMax']]
    union = contamina_and_clima(list_csv,list_json)
#     print(union.info())
    corr = union.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(9, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
def mapa_por_distritos_madridcentral(list_url):
#     magnitudes CO y NOx más relacionadas con las emisiones del transporte
    
    df2 = daily_airmad_varios_mu_m3(list_url)
    df_magnitudes_selected = df2[(df2.MAGNITUD==6)|(df2.MAGNITUD==12)]
    df_magnitudes_selected.distrito = df_magnitudes_selected.distrito.apply(lambda x: '%02d' % (x,))
    df_magnitudes_selected = df_magnitudes_selected.sort_values(by= 'FECHA')
    print(df_magnitudes_selected.columns)
    
    
    mask_antes = (df_magnitudes_selected.FECHA >= '2017-11-30')&(df_magnitudes_selected.FECHA < '2018-11-30')
    mask_despue = (df_magnitudes_selected.FECHA >= '2018-11-30')&(df_magnitudes_selected.FECHA < '2019-11-30')
    
    
#     maping_antes =df_magnitudes_selected[mask_antes].groupby(['FECHA','MAGNITUD', 'distrito']).mean().reset_index()  
#     maping_despues =df_magnitudes_selected[mask_despue].groupby(['FECHA','MAGNITUD', 'distrito']).mean().reset_index()
    maping_antes =df_magnitudes_selected[mask_antes].groupby(['distrito']).mean().reset_index()  
    maping_despues =df_magnitudes_selected[mask_despue].groupby(['distrito']).mean().reset_index()  
#     print('maping:', maping)
    
    contamina = pd.merge(left = maping_antes[['distrito','Value']],right =maping_despues[['distrito','Value']],left_index = True,right_index = True,how = 'outer',suffixes = ('antes','despues'))
    contamina['porcentaje_antes_despues'] = contamina['Valuedespues']*100/contamina['Valueantes']
#     print(contamina)
    print('DATOS MADRID CENTRAL PARA MONÓXIDO DE CARBONO Y ÓXIDOS DE NITRÓGENO')
    
    
#     print('mean, porcentaje',contamina['porcentaje_antes_despues'].describe())
#     print('idxmax distrito con minima caida', contamina.set_index('distritoantes').porcentaje_antes_despues.idxmax())
#     print('idxmax distrito con maxima caida', contamina.set_index('distritoantes').porcentaje_antes_despues.idxmin())
    
    print('--------------------------------------')
    print("\x1b[1;33m"+'% Contaminación respecto al año anterior',contamina['porcentaje_antes_despues'].describe())
    print('--------------------------------------')
    print('Distrito con minima caida es RETIRO', contamina.set_index('distritoantes').porcentaje_antes_despues.idxmax())
    print('--------------------------------------')
    print('Distrito con maxima caida es VILLAVERDE', contamina.set_index('distritoantes').porcentaje_antes_despues.idxmin())
    print('--------------------------------------')
    
    
    

    from urllib.request import urlopen
    import json
    import plotly.express as px
    #with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    #    counties = json.load(response)

    file = open('distrito_geojson.geojson')
    distritos = json.load(file)
    print('--------------------------------------')
    print('BEFORE MADRID CENTRAL')
    print('--------------------------------------')
#     print('ANTES',maping_antes.FECHA)
#     print('ANTES mag',maping_antes.head())
    fig = px.choropleth_mapbox(maping_antes, geojson=distritos, locations='distrito', color='Value',range_color = (0,190),
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    print('--------------------------------------')
    print('AFTER MADRID CENTRAL')
    print('--------------------------------------')
#     print('DESPUÉS',maping_despues.FECHA)
#     print('despues mag',maping_despues.head())
    fig = px.choropleth_mapbox(maping_despues, geojson=distritos, locations='distrito', color='Value',range_color = (0,190),
                                featureidkey='properties.codigoalternativo',
                            color_continuous_scale="reds",
                            center={"lat": 40.4165000, "lon": -3.7025600},
                                mapbox_style="stamen-toner", zoom=10, opacity=0.9
                            )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
    print("\033[;36m"+ 'La caída en distrito centro es inferior al 5%')

#ahora hay diferentes opciones de presentación del df
#OPCIÓN1:
#se hace la media de las diferentes estaciones en las que se ha tomado la medida. pierdo info de las ESTACIONES.
# df1 = df.groupby(['FECHA','MAGNITUD']).mean()[['FECHA', 'MAGNITUD','Value']]