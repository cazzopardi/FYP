import pickle
from typing import Callable, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from preprocessing.supervised import preprocess

# Methodology proposed by Cao et al. in doi.org/10.1109/TCYB.2018.2838668
# Implementation retrieved from https://github.com/vanloicao/SAEDVAE/tree/master

"Convert string into number"
def string_to_number(uni_values, data):           
    #create a dictionary
    dic = dict()
    idx = 1
    for key in uni_values:
        dic[key]= idx
        idx=idx+1
        
    #replace category values by numbers
    k=0
    for element in data:
        data[k] = dic[element[0]]
        k=k+1
             
    return data
    
 
"Real-value encoder for NSL-KDD" 
def real_value_encode_KDD(train, test, list_features): 
    
    raw_train = train
    raw_test  = test
    for i in list_features:
        #extract category features
        f_train = []
        f_test  = []
        f_train = raw_train[:,i:i+1]
        f_test  = raw_test[:,i:i+1]
        feature = np.concatenate((f_train, f_test))
        uni_values, indices = np.unique(feature, return_index=True)  #find the set of different values
       
        """
        if (i==1):
            uni_values = ['tcp','udp','icmp']     #re-order protocol
        elif (i==2):
            uni_values = ['http', 'domain_u','smtp','ftp_data'	,'other','private','ftp','telnet',\
                        'urp_i','finger','eco_i','auth','ecr_i','IRC','pop_3','ntp_u','time','X11',\
                        'domain','urh_i','red_i','ssh', 'tim_i','shell','imap4','tftp_u','Z39_50','aol',\
                        'bgp','courier','csnet_ns','ctf','daytime','discard','echo','efs','exec','gopher',\
                        'harvest','hostnames','http_2784','http_443','http_8001','iso_tsap','klogin','kshell',\
                        'ldap','link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat',\
                        'nnsp','nntp','pm_dump','pop_2','printer','remote_job','rje','sql_net','sunrpc','supdup',\
                        'systat','uucp','uucp_path','vmnet','whois']
        else:
            uni_values =['SF','REJ','S1','S0','RSTO','RSTR','S2','S3','OTH','SH','RSTOS0']
        """        
        
        print ("\nFeature_" + str(i) + ":")
        print (uni_values)
        np.savetxt("data/NSL-KDD/feature_" + str(i)+ "_value.csv",  uni_values  ,delimiter=",", fmt="%s")  
        
        f_train = string_to_number(uni_values, f_train)
        f_test  = string_to_number(uni_values, f_test)
        
        #delete the feature from raw_data
        raw_train = np.delete(raw_train, i, axis = 1)
        raw_test  = np.delete(raw_test,  i, axis = 1)
        #insert new features into array
        raw_train = np.insert(raw_train, [i], f_train, axis=1)
        raw_test  = np.insert(raw_test , [i], f_test, axis=1)
    
    return raw_train, raw_test


"One-Hot-Encoder for NSL-KDD"
def one_hot_encode_KDD(raw_train, raw_test, list_features): 

    for i in list_features:
        #extract category features
        f_train = []
        f_test  = []
        f_train = raw_train[:,i:i+1]
        f_test  = raw_test[:,i:i+1]
        feature = np.concatenate((f_train, f_test))
      
        uni_values, indices = np.unique(feature, return_index=True)  
 
        f_train = string_to_number(uni_values, f_train)
        f_test  = string_to_number(uni_values, f_test)
        
        value = np.copy(uni_values)
        value  = np.reshape(value, (len(value),1))
        f_dic  = string_to_number(uni_values, value)  
        
        print (uni_values)
        print (f_dic)
        
        "One Hot Encoder"
        enc = OneHotEncoder()
        enc.fit(f_dic)  
        new_f_train = enc.transform(f_train).toarray()
        new_f_test  = enc.transform(f_test).toarray()
        
        #delete the feature from raw_data
        raw_train = np.delete(raw_train, i, axis = 1)
        raw_test  = np.delete(raw_test, i, axis = 1)
        #insert new features into array
        raw_train = np.insert(raw_train, [i], new_f_train, axis=1)
        raw_test  = np.insert(raw_test , [i], new_f_test, axis=1)
        
    return raw_train, raw_test
        

"Split data into four groups of attacks and normal"
def split_NSLKDD_group(raw_data): 
    
    probe_label  = ['ipsweep', 'nmap', 'portsweep', 'satan',\
                    'mscan', 'saint'] # add 2 more attacks from test set 
    
    dos_label    = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',\
                    'apache2', 'mailbomb', 'processtable', 'udpstorm'] #add 4 more attacks from test set
                    
                    
    r2l_label    = ['ftp_write','guess_passwd','imap','multihop','phf','spy',\
                    'warezclient','warezmaster',\
                     'httptunnel','named', 'sendmail',\
                    'snmpgetattack','snmpguess', 'worm', 'xlock', 'xsnoop']#add 8 more attacks from test set
                    
    u2r_label    = ['buffer_overflow','loadmodule','perl','rootkit',\
                    'xterm', 'ps', 'sqlattack']  #add 3 more from test set
    
    dim = raw_data.shape[1]
    n   = raw_data.shape[0]
    
    #extract label column
    label =  raw_data[:,(dim-1):dim]
    
    for i in range(n):
        if  (label[i] in probe_label):
            label[i] = 1                #Probe
        elif (label[i] in dos_label):
            label[i] = 2                #DoS
        elif (label[i] in r2l_label):  
            label[i] = 3                #R2L   
        elif (label[i] in u2r_label):  
            label[i] = 4                #U2R
        elif (label[i] == "normal"):    #Normal 
            label[i] = 0
        else:
            print("No group of attacks chosen")
            
     #delete the feature from raw_data
    raw_data = np.delete(raw_data, [dim-1], axis = 1)
    #insert new features into array
    raw_data = np.insert(raw_data, [dim-1], label, axis=1) 
    
    return raw_data
    
    
def encode_NSLKDD(raw_train: np.ndarray, raw_test: np.ndarray, encoder_type: Literal['real_value', 'one_hot']) -> tuple[np.ndarray, np.ndarray]:
    #    a1 =   raw_train[:,-1]
    #    a2 =   raw_test[:,-1]
    #    raw_train = raw_train[np.float64(a1) > 15]
    #    raw_test  = raw_test[np.float64(a2) > 15]

    #remove the difficulty level in the last column
    raw_train = raw_train[:,0:-1]
    raw_test  = raw_test[:,0:-1]
    #list of the categorical features
    # "1 - protocol_type","2 - service","3 - flag"
    list_features = [3,2,1] 
    
    "1 - duration: continuous"
    "5 - src_bytes: continuous"
    "6 - dst_bytes: continuous"   
    "13- num_compromised: continuous (may not need to log2)"
    "16- num_root: continuous (may not need to log2)"
    "https://github.com/thinline72/nsl-kdd"
    "https://github.com/jmnwong/NSL-KDD-Dataset"
    
    "convert extremely large values features into small values by log2 function"
    feature_log = [1, 5, 6]     #index: 0, 4, 5
    for i in feature_log:
        raw_train[:,i-1:i] = np.log2((raw_train[:,i-1:i]).astype(np.float64)+1)
        raw_test[:, i-1:i] = np.log2((raw_test[:, i-1:i]).astype(np.float64)+1)    
    
    if (encoder_type == "real_value"):
        pro_train, pro_test = real_value_encode_KDD(raw_train, raw_test, list_features)
    elif (encoder_type == "one_hot"):
        pro_train, pro_test = one_hot_encode_KDD(raw_train, raw_test, list_features)
    else:
        print ("no encoder data is choosen")

    pro_train = split_NSLKDD_group(pro_train)
    pro_test  = split_NSLKDD_group(pro_test)
    
    print ("\n")
    print ("train set: ",len(pro_train) )
    print ("test set: ", len(pro_test) )

    # np.savetxt("Data/NSLKDD/NSLKDD_Train.csv", pro_train  ,delimiter=",", fmt="%s")
    # np.savetxt("Data/NSLKDD/NSLKDD_Test.csv",  pro_test   ,delimiter=",", fmt="%s")
    return pro_train.astype(float), pro_test.astype(float)


def preprocess_nsl_kdd(train_path: str, test_path: str, encoding:Literal['real_value', 'one_hot']='one_hot'):
    raw_train = np.genfromtxt(train_path, delimiter=",", dtype='str')
    raw_test = np.genfromtxt(test_path, delimiter=",", dtype='str')   
    train_data, test_data = encode_NSLKDD(raw_train, raw_test, encoder_type=encoding)

    y_train = train_data[:,-1]                #Select label column
    X_train = train_data[y_train == 0]        #Select only normal data for training  
    X_train = X_train[:,0:-1]                 #Remove label column
    print("Normal training data: ", X_train.shape[0]) 
    np.random.shuffle(X_train)
    X_train = X_train[:6734]                  #Sample 5000 connections for training 


    y_test = test_data[:,-1]                  #Select label column  
    X_test = test_data[:,0:-1]                #Select data except label column

    test_X0 = X_test[y_test == 0]             #Normal test
    test_X1 = X_test[y_test > 0]              #Anomaly test 
    print("Normal testing data: ", test_X0.shape[0])
    print("Anomaly testing data: ", test_X1.shape[0])

    X_test = np.concatenate((test_X0, test_X1))

    test_y0 = np.full((len(test_X0)), True, dtype=bool)
    test_y1 = np.full((len(test_X1)), False,  dtype=bool)
    y_test =  np.concatenate((test_y0, test_y1))

    #create binary label (1-normal, 0-anomaly) for compute AUC later
    y_test = (y_test).astype(np.int8)

    #scaler = MinMaxScaler()
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def preprocess_cic_ids(data: pd.DataFrame, target:str='attack category', encoding:Literal['real_value', 'one_hot']='one_hot') -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # TODO: consider one-hot encoding
    # if (encoding == "real_value"):
    #     pro_train, pro_test = real_value_encode_KDD(raw_train, raw_test, list_features)
    # elif (encoding == "one_hot"):
    #     pro_train, pro_test = one_hot_encode_KDD(raw_train, raw_test, list_features)
    # else:
    #     raise ValueError('Encoding must be specified')
    
    X_train, X_test, y_train, y_test = preprocess(data)

    # pickle.dump((X_train, X_test, y_train, y_test), open('uns_dl_sampled_data.pkl','wb'))
    # X_train, X_test, y_train, y_test = pickle.load(open('uns_dl_sampled_data.pkl','rb'))

    # np.savetxt("Data/NSLKDD/NSLKDD_Train.csv", pro_train  ,delimiter=",", fmt="%s")
    # np.savetxt("Data/NSLKDD/NSLKDD_Test.csv",  pro_test   ,delimiter=",", fmt="%s")

    # y_train = train[target]
    # y_test = test[target]
    # adjust -1s to remain -1s following log operation
    X_train['Init Fwd Win Byts'] = X_train['Init Fwd Win Byts'].astype('float64').replace(-1, -0.5)
    X_train['Init Bwd Win Byts'] = X_train['Init Bwd Win Byts'].astype('float64').replace(-1, -0.5)
    X_test['Init Fwd Win Byts'] = X_test['Init Fwd Win Byts'].astype('float64').replace(-1, -0.5)
    X_test['Init Bwd Win Byts'] = X_test['Init Bwd Win Byts'].astype('float64').replace(-1, -0.5)

    X_train: pd.DataFrame = X_train[y_train == 0]        #Select only normal data for training  
    X_train = X_train.iloc[:6734]                  #Sample 5000 connections for training 
    y_train = y_train.iloc[:6734]

    "convert extremely large values features into small values by log2 function"
    get_large_col_mask: Callable[[pd.DataFrame],pd.Series] = lambda df: (df > 10000).any(axis=0)
    feature_log: pd.Series = get_large_col_mask(X_train)|get_large_col_mask(X_test)  # mask indicating which columns contain large values
    X_train.loc[:,feature_log] = np.log2(X_train.loc[:,feature_log].astype('float64') + 1).astype('float64')
    X_test.loc[:,feature_log] = np.log2(X_test.loc[:,feature_log].astype('float64') + 1).astype('float64')

    #scaler = MinMaxScaler()
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    scaled_data: np.ndarray = scaler.transform(X_train)
    X_train = pd.DataFrame(scaled_data, index=X_train.index, columns=X_train.columns)
    scaled_data = scaler.transform(X_test)
    X_test = pd.DataFrame(scaled_data, index=X_test.index, columns=X_test.columns)
    return X_train, X_test, y_train, y_test
