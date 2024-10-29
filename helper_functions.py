def helper_boxplotEDA(dataframe, xlabel=False):
    '''
    Plotting all df feature boxplots and visualising them side by side. 
    '''
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.figure(figsize=(15,6))
    dataframe.boxplot()
    plt.title("Box plot to identify any significant outliers.")
    plt.ylabel("Values")
    if xlabel == False:
        plt.xticks([])
        plt.show()
    elif xlabel == True:
        plt.show()
    

def helper_findOutliers(featureDF,targetDF):
    '''
    Requires a featureDF and a targetDF to return a master df with all outliers removed.
    Outlier calculation is done by checking the 1.5*IQR range.
    '''
    import pandas as pd

    Q1 = featureDF.quantile(0.25)
    Q3 = featureDF.quantile(0.75)
    IQR = Q3 - Q1
    lowerBound = Q1 - 1.5 * IQR
    upperBound = Q3 + 1.5 * IQR

    outlierIndices = []

    for col in featureDF.columns:
        outliers = featureDF[(featureDF[col] < lowerBound[col]) | (featureDF[col]>upperBound[col])]
        outlierIndices.extend(outliers.index.tolist())

    # Making the outlierIndices unique
    uniqueOutlierIndices = set(outlierIndices)

    #Removing the index rows no longer needed
    cleanedFeatures = featureDF.drop(index=uniqueOutlierIndices)

    # Left joining to add the classes back to the normalised and cleaned dataset. 
    master = cleanedFeatures.join(target)

    return master


def vifChecker(df, target):
    '''
    Input: df and target (as a string).
    Output: For each column (with the target removed from the df), a VIF score is printed
    '''
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Feature Extraction from data
    cols = list(df.columns.values)
    cols.remove(target)

    # the independent variables set
    X = df[cols]

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                            for i in range(len(X.columns))]

    print(vif_data)

def helper_classificationEval(y_test, y_pred, average=None)
    '''
    This function takes in y_test, y_pred and returns:
    accuracy
    precision
    recall
    f1
    classification report
    visual confusion matrix
    '''
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score,recall_score,f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall=recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)

    # Confusion Matrix
    confusionM = confusion_matrix(y_test, y_pred)

    # Using Seaborn to Visualise the Confusion Matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusionM,annot=True)
    plt.show()