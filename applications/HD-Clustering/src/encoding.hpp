

def train(X_train, y_train, X_test, y_test, D=500, alg='rp', epoch=20, lr=1.0, L=64):
    np.random.seed(0)
    #randomly select 20% of train data as validation
    breakpoint()

    if (hasattr(X_train, 'to_numpy')):
        X_train = X_train.to_numpy()
    if hasattr(y_train, 'to_numpy'):
        y_train = y_train.to_numpy().astype(int)
    if hasattr(X_test, 'to_numpy'):
        X_test = X_test.to_numpy()
    if hasattr(y_test, 'to_numpy'):
        y_test = y_test.to_numpy().astype(int)

    X_train = X_train
    permvar = np.arange(0, len(X_train))
    np.random.shuffle(permvar)
    X_train = [X_train[i] for i in permvar]
    y_train = [y_train[i] for i in permvar]
    cnt_vld = int(0.2 * len(X_train))
    X_validation = X_train[0:cnt_vld]
    y_validation = y_train[0:cnt_vld]
    X_train = X_train[cnt_vld:]
    y_train = y_train[cnt_vld:]

    #encodings
    if alg in ['rp', 'rp-sign']:
        #create base matrix
        base_matrix = np.random.rand(D, len(X_train[0]))
        base_matrix = np.where(base_matrix > 0.5, 1, -1)
        base_matrix = np.array(base_matrix, np.int8)
        breakpoint()
        print('\nEncoding ' + str(len(X_train)) + ' train data')
        train_enc_hvs = encoding_rp(X_train, base_matrix, signed=(alg == 'rp-sign'))
        print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
        validation_enc_hvs = encoding_rp(X_validation, base_matrix, signed=(alg == 'rp-sign'))
    
    elif alg in ['idlv', 'perm']:
        #create level matrix
        lvl_hvs = []
        temp = [-1]*int(D/2) + [1]*int(D/2)
        np.random.shuffle(temp)
        lvl_hvs.append(temp)
        change_list = np.arange(0, D)
        np.random.shuffle(change_list)
        cnt_toChange = int(D/2 / (L-1))
        for i in range(1, L):
            temp = np.array(lvl_hvs[i-1])
            temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]] = -temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]]
            lvl_hvs.append(list(temp))
        lvl_hvs = np.array(lvl_hvs, dtype=np.int8)
        x_min = min( np.min(X_train), np.min(X_validation) )
        x_max = max( np.max(X_train), np.max(X_validation) )
        bin_len = (x_max - x_min)/float(L)
        
        #need to create id hypervectors if encoding is level-id
        if alg == 'idlv':
            cnt_id = len(X_train[0])
            id_hvs = []
            for i in range(cnt_id):
                temp = [-1]*int(D/2) + [1]*int(D/2)
                np.random.shuffle(temp)
                id_hvs.append(temp)
            id_hvs = np.array(id_hvs, dtype=np.int8)
            print('\nEncoding ' + str(len(X_train)) + ' train data')
            train_enc_hvs = encoding_idlv(X_train, lvl_hvs, id_hvs, D, bin_len, x_min, L)
            print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
            validation_enc_hvs = encoding_idlv(X_validation, lvl_hvs, id_hvs, D, bin_len, x_min, L)