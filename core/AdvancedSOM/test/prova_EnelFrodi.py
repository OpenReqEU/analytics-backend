#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-


from ASOM import ASOM
from ASOM import GapStatisticsKopt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pylab
import  pandas as pd
from   sklearn.cluster import MiniBatchKMeans
import matplotlib
# matplotlib.use('Qt4Agg')

from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd
import numpy as np

#pulizia:
    #selezione colonne
def cleaner(dataset,threshold=0):


    clean=dataset[[#'CDC_FISCALE',
                     'ETA',
#                     'SESSO',#da togliere asterisco - gestire i null
#                     'LUOGO_RESIDENZA',#da mappare (N\S\C???) - gestire i null - presenti valori numerici
#                     'PROVINCIA',#asterischi - Da mappare - gestire i null
#                     'COMMODITY',#asterischi - gestire i null
                     'FLN_CLIENTE_ELE',
                     'FLN_CLIENTE_GAS',
                     'FLN_CLIENTE_ELE_GAS',
                     'SEGMENTO_COMMERCIALE',
#                     'TIPO_CLIENTE_RES',#presenti asterischi + null
#                     'SEGMENTO_CLIENTE',#asterischi presenti - società uguali con nomi diversi - gestire i null
#                     'ANZIANITA_CLIENTE',#presenti valori strani - chiedere a Nello
                     'OFFERTE_ATT_ELE',
                     'OFFERTE_ATT_GAS',
#                     'FLN_BL_PRESENTE',#presente solo NO - chiedere a NELLO
                     'FLC_BL_PASSATO',
                     'COUNT_BP_EMAIL',
                     'COUNT_BP_TEL',
#                     'CODICE_ATECO',#presenti (Pochi) codici composti da sole due cifre - presenti null oltre che per persone fisiche, anche per alcune aziende
#                     'FLN_INDIRIZZO_CF_VS_POD', # presenti null - chiedere spiegazioni a nello perchè dovrebe essere un flag calcolato sulla coerenza tra indirizzi
#                     'STATO_CESSAZIONE',#presenti null - chiedere spiegazioni a nello perchè dovrebe essere un flag calcolato
                     'COUNT_PA_MESE',
                     'COUNT_PA_TRIMESTRE',
                     'COUNT_PA_SEMESTRE',
                     'COUNT_PA_ANNO',
                     'COUNT_SWA_MESE',
                     'COUNT_SWA_TRIMESTRE',
                     'COUNT_SWA_SEMESTRE',
                     'COUNT_SWA_ANNO',
                     'COUNT_SUBENTRO_MESE',
                     'COUNT_SUBENTRO_TRIMESTRE',
                     'COUNT_SUBENTRO_SEMESTRE',
                     'COUNT_SUBENTRO_ANNO',
                     'COUNT_VOLTURA_MESE',
                     'COUNT_VOLTURA_TRIMESTRE',
                     'COUNT_VOLTURA_SEMESTRE',
                     'COUNT_VOLTURA_ANNO',
#                     'DTA_ULTIMA_PA',#VALORI NULL presenti, 01-01-1900, 01-01-1905, 31-12-2999
#                     'DTA_ULTIMA_SWA', #VALORI NULL presenti, 01-01-1000, 01-01-1900, 31-12-2999
#                     'DTA_ULTIMA_SUBENTRO',#VALORI NULL ",31-12-2999
#                     'DTA_ULTIMA_VOLTURA',#VALORI NULL ",01-01-1000, 01-01-1900,31-12-2999
                     'MEDIA_PA_MESE',
                     'MEDIA_PA_TRIMESTRE',
                     'MEDIA_PA_SEMESTRE',
                     'MEDIA_PA_ANNO',
                     'MEDIA_SWA_MESE',
                     'MEDIA_SWA_TRIMESTRE',
                     'MEDIA_SWA_SEMESTRE',
                     'MEDIA_SWA_ANNO',
                     'MEDIA_SUBENTRO_MESE',
                     'MEDIA_SUBENTRO_TRIMESTRE',
                     'MEDIA_SUBENTRO_SEMESTRE',
                     'MEDIA_SUBENTRO_ANNO',
                     'MEDIA_VOLTURA_MESE',
                     'MEDIA_VOLTURA_TRIMESTRE',
                     'MEDIA_VOLTURA_SEMESTRE',
                     'MEDIA_VOLTURA_ANNO',
                     'COUNT_FATTURE',
                     'COUNT_FATTURE_ELE',
                     'COUNT_FATTURE_GAS',
                     'TOTALE_FATTURATO', #valori negativi - chiedere a NELLO
                     'COUNT_FATTURE_MESE',
                     'COUNT_FATTURE_TRIMESTRE',
                     'COUNT_FATTURE_SEMESTRE',
                     'COUNT_FATTURE_ANNO',
                     'COUNT_FORN_TOT',
                     'COUNT_FORN_ELE',
                     'COUNT_FORN_GAS',
                     'COUNT_FORN_ATTIVE_TOT',
                     'COUNT_ATTIVAZIONI_ELE',
                     'COUNT_ATTIVAZIONI_GAS',
                     'COUNT_CESSAZIONI_ELE',
                     'COUNT_CESSAZIONI_GAS',
#                     'COMPANY_LEGAL_FORM', #presenti null - presenti valore 'X' - come interpretare i numeri presenti
#                     'CUSTOMER_PROFILE', # PRESENTI NULL - 0 COME INTERPRETARE I NUMERI
#                     'SETTORE_MERCEOLOGICO', #PRESENTI ASTERISCHI E NULL
#                     'STATO_DOMICILIAZIONE', #PRESENTI NULL
#                     'SEGMENTO_ANZIANITA', #PRESENTI NULL E ASTERISCHI
#                     'SEGMENTO_COMMERCIO', # PRESENTI NULL E ASTERISCHI
#                     'VALORE', - #PRESENTI NULL E ASTERISCHI - CHIEDERE INTERPRETAZIONE
                     'FATTURATO_MENSILE',
                     'FATTURATO_FRODE_FA',
                     'INCASSI_MENSILI',
                     'MEDIA_GG_RITARDO_FATT_ANNO',
                     'MEDIA_GG_RITARDO_FATT_SEM',
                     'MEDIA_GG_RITARDO_FATT_TRIM',
                     'PD_MONITORAGGIO',
                     'CRED_BUCKET_0_30',
                     'CRED_BUCKET_30_60',
                     'CRED_BUCKET_60_90',
                     'CRED_BUCKET_180_360',
                     'CRED_BUCKET_360_720',
                     'CRED_BUCKET_OLTR_720',
                     'CRED_IMP_RATEIZZATO',
                     'DELINQUENCY',
#                     'DT_AFFIDO_ADR', # presenti null - capire come gestirli - presenti anni strani (2999) - capire come gestirli e chiedere a nello siegazioni
                     'GARANZIE',
                     'IMP_CEDUTO',
                     'QTA_FALSI_DIMOSTRATI_PAGAM',
                     'QTA_FRODI_ANNO_RIF',
                     'QTA_PHONE_COLL_ANNO_RIF',
                     'T_VAL_DEPOSITO_CAUZ_FATTURATO',
                     'VAL_ASSICURAZIONE',
                     'VAL_DEPOSITO_CAUZ_INCASSATO',
                     'VAL_FIDEJUSSIONE',]]

        #gestione null
    clean=clean.fillna(0)

        #gestine vatriabili nominali - GETDUMMIES
    clean=pd.get_dummies(clean)

    #salva lista colonne:
    columns=clean.columns

#        #analisi varianza:
#    variances=[]
#    for a in columns:
#        variances.append([np.var(clean[a]),a])
#    variances.sort()
#    highVarianceColumns=[a[1] for a in variances if a[0]>=threshold]
#    clean=clean[[i for i in highVarianceColumns]]


        #STANDARDIZZAZIONE
    nrm=MinMaxScaler()
    cleanSTD=nrm.fit_transform(clean)

    return clean,columns,cleanSTD






def start():

    #import dataframe
    df = pd.read_csv('./data/20170919_2220_dataframe.csv',sep=',',na_values=['***','*'])

    # clean dataset
    X,columns,X_STD = cleaner(df,1)

    X=np.array(X_STD[:100000],dtype=float)

    som = ASOM(alpha_max=0.2, height=12, width=12, outlier_unit_threshold=0.01, outlier_percentile=99., Koutlier_percentile=1.5)

    som.train_batch(X, num_epoch=1000, verbose=2)

    data2unit, data2cell, data2dist, data2saliency, data2saliency_prob, data2maps = som.predict(X)
    # print "som.unit2saliency=",som.unit2saliency #lista di codebook normali (salient unit)
    # print "som.data_saliency=",som.data_saliency #lista di data normali (salient instances)
    o = np.sum(data2saliency == False)
    print "--------------Anomaly     =",o
    print "--------------Anomaly_perc=",o/float(X.shape[0])

    som.plot_codebook_weights()

    # som.plot_mapping( )
    som.plot_units_clusters_outlier(model_features_reduction=None)
    # for k in range(som.W.shape[1]):
    #     som.plot_cellDistribution(component=k)
    cluster_model=MiniBatchKMeans(n_clusters=19)
    som.plot_mapping()
    # som.plot_mapping_cluster()
    som.plot_mapping_cluster(cluster_model=cluster_model)

    # som.plot_clusters()
    # som.plot_units( )
    # som.plot_units_outlier( )
    # som.plot_units_clusters_outlier()
    # som.plot_activations()
    som.plot_mapping(X)
    print

    # pylab.figure()
    # for k in range(som.num_units):
    #     pylab.plot(som.W[k], label=str(k))




if __name__ == "__main__":
    start()
    pylab.show()
