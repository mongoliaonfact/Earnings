
import pandas as pd 

def lm_sentiment_score(text: str=None)-> dict:

    # intialize dictionary
    sentiment_dict = {'LM_Positive':[], 'LM_Negative':[], 'LM_Uncertainty':[]}
    
    # Initialize sentiment scores
    pos_score = 0
    neg_score = 0
    uncertain_score = 0
    
    # upload Loughran-McDonald sentiment dataset
    lm_dataframe = pd.read_csv('../data/14401178/LM-SA-2020.csv') 

    for token in text:
        if token in lm_dataframe['word'].to_list():
            token_row = lm_dataframe[lm_dataframe['word'] == token]
            lm_sent = token_row.iloc[0]['sentiment']
            if lm_sent == 'Positive':
                pos_score+=1
            elif lm_sent == 'Negative':
                neg_score+=1
            elif lm_sent == 'Uncertainty':
                uncertain_score+=1
        else:
            pass
    sentiment_dict['LM_Positive'].append(pos_score)
    sentiment_dict['LM_Negative'].append(neg_score)
    sentiment_dict['LM_Uncertainty'].append(uncertain_score)

    return sentiment_dict


def compute_lm_sentiment(dataframe: pd.DataFrame=None)-> pd.DataFrame:
    results = []
    for ind, row in enumerate(dataframe.to_dict('records')):
        if ind%10==0:
            print(f'printing: {ind}')
        text = row['clean']
        my_sent = lm_sentiment_score(text=text)
        df = pd.DataFrame(my_sent)
        
        results.append(df)
    
    results_concat = pd.concat(results, axis=0)
    results_concat = results_concat.reset_index(drop=True)

    return results_concat