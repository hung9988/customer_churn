from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
tree_in_forest = 100
def random_dataset(X_train,y_train):
    train_set_size=X_train.shape[0]
    indices=np.random.choice(train_set_size,size=train_set_size,replace=True)
    
    num_ran_features = int(np.sqrt(X_train.shape[1]))
    features_indices=np.random.choice(X_train.shape[1],size=num_ran_features,replace=False)
    
    return X_train[indices,:][:,features_indices],y_train[indices],features_indices

def random_forest(X_train,y_train,tree_in_forest,tree_depth=10,min_sample_split=2):
    forest = []
    feature_indices_for_tree=[]
    for i in range(tree_in_forest):
        X_train_,y_train_,features = random_dataset(X_train,y_train)
        feature_indices_for_tree.append(features)
        tree = DecisionTreeRegressor(max_depth=tree_depth,min_samples_split=min_sample_split)
        tree.fit(X_train_,y_train_)
        forest.append(tree)
    return forest,feature_indices_for_tree


def predict_forest(X,forest,feature_indices):
        y_pred = np.zeros((X.shape[0],1))
        for i in range(len(forest)):
            tree_i_prediction=forest[i].predict(X[:,feature_indices[i]])
            tree_i_prediction=np.round(tree_i_prediction)
            y_pred+=tree_i_prediction.reshape(-1,1)
        return y_pred/len(forest)
    

def grid_search_random_forest(X_train,y_train,X_val,y_val,tree_in_forest_list,tree_depth_list,min_sample_split_list):
    best_acc_val=0
    for tree_in_forest in tree_in_forest_list:
        for tree_depth in tree_depth_list:
            for min_sample_split in min_sample_split_list:
                forest,feature_indices=random_forest(X_train,y_train,tree_in_forest,tree_depth=tree_depth,min_sample_split=min_sample_split)
                y_pred=predict_forest(X_val,forest,feature_indices)
                y_pred=np.round(y_pred)
                acc_val=accuracy_score(y_val,y_pred)
                print('tree_in_forest:',tree_in_forest,'tree_depth:',tree_depth,'min_sample_split:',min_sample_split,'accuracy:',acc_val)
                if acc_val > best_acc_val:
                    best_acc_val=acc_val
                    best_forest=forest
                    best_feature_indices=feature_indices
                    best_tree_in_forest=tree_in_forest
                    best_tree_depth=tree_depth
                    best_min_sample_split=min_sample_split
    return {'best_acc_val':best_acc_val,'best_forest':best_forest,'best_feature_indices':best_feature_indices,'best_tree_in_forest':best_tree_in_forest,'best_tree_depth':best_tree_depth,'best_min_sample_split':best_min_sample_split}

