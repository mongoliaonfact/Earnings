# author: bbaasan
# date: 2023-08-02
#


from bs4 import BeautifulSoup
import pandas as pd
import requests
from datetime import datetime, timedelta, date
from time import strptime
import os
import re
import numpy as np
import yfinance as yfin
import nltk
from nltk.corpus import stopwords

class SeekingAlpha:
    # non script characters
    java = 'javascript frequently feedback blocked proceeding disable'
    

    def __init__(self):
        pass

    def get_transcript_info(self, page_num: int) -> list:
        '''
        gets list of links

        '''
        script = list()

        url = f'https://seekingalpha.com/earnings/earnings-call-transcripts?page={page_num}'
        page = requests.get(url)

        soup = BeautifulSoup(page.content.decode('utf-8'), 'html.parser')

        for tags in soup.find_all('h3'):
            each_h3_tag = tags
            href_anchor = each_h3_tag.a['href']
            identifier = ['earnings', 'call', 'transcript']
            for word in href_anchor.split('-'):
                if word in identifier:
                    script.append(href_anchor)
                else:
                    pass

        return script

    def get_single_transcript(self, url: str):
        '''
        scrapes the url page and returns text
        '''
        rows = list()

        page = requests.get(url)
        soup = BeautifulSoup(page.content.decode('utf-8'), features = 'html.parser')

        # gets paragraph tags
        texts = soup.find_all('p')

        for row in texts:
            rows.append(row.get_text()+'\n')

        return rows


    def get_pages(self, page_num: int) -> pd.DataFrame:

        extracted = list()

        url = f'https://seekingalpha.com/earnings/earnings-call-transcripts?page={page_num}'
        page = requests.get(url)
        soup = BeautifulSoup(page.content.decode('utf-8'), 'html.parser')

        h3_tags = soup.find_all('h3')    # list of tags
        footer = soup.find_all('footer')

        for i in range(len(h3_tags)):
            h3_val = h3_tags[i]
            f_val = footer[i]

            url_link = h3_val.a['href']
            recorded_date = f_val.find_all('span', recursive=False)[-1].text
            extracted.append({
                'url':url_link,
                'recorded_date':recorded_date})

        pd.DataFrame(extracted).to_pickle(f'page_{page_num}.pkl')


    def convert_weekday(self, x: str) -> datetime.date:
        odor = ''

        date_splitted = x.split(',')
        if len(date_splitted) == 2:
            odor, e2 = date_splitted

            if odor == 'Today':
                odor = datetime.today()
            else: # for yesterday
                odor = date.today() - timedelta(days=1)
        return odor


    def convert_odor_sar_jil(self, x: str):
        ''''''
        e1 = x.split(',')[1:]
        jil = e1[-1].strip()

        e2 = e1[0].split('.')[0].strip()
        sar = strptime(sar, '%b').tm_mon
        odor = e1[0].split(',')[1]

        return datetime(int(jil), int(sar), int(odor))

    #TODO: this is a very questionable method. CHECK AGAIN FOR CONFIRMATION
    def convert_seekingalpha_datetime(self, x: str):
        today_yester = ['today', 'yesterday']

        sar, odor, jil = '', '', ''

        splitted = [element for element in x.split(',')]
        if len(splitted) == 3:
            sar_odor = splitted[1]
            if ',' in sar_odor:
                sar_odor = sar_odor.replace(',',' ')
                sar, odor = sar_odor
                jil = splitted[2]

                try:
                    sar = self.month_to_int(a_month=sar)
                except KeyError:
                    print('sar is not here. Check the format, which is "jan".')
                return datetime(int(jil), int(sar), int(odor))
            else:
                sar, odor = sar_odor
                jil = splitted[2]
                try:
                    sar = self.month_to_int(a_month=sar)
                except KeyError:
                    print('sar is not here. Check the format, which is "jan".')
                return datetime(int(jil), int(sar), int(odor))

        elif len(splitted) == 2:
            if splitted[0] in today_yester:
                if splitted[0] == 'today':
                    odor = datetime.today().date() - timedelta(days=1)
                    return odor # returns datetime object
                else: # which is yesterday
                    odor = datetime.today().date() - timedelta(days=2)
                    return odor # also datetime object
            else:
                sar_odor = splitted[1]
                if '.' in sar_odor:
                    sar_odor = sar_odor.replace('.', ' ')
                    sar, odor = sar_odor
                else:
                    pass

                sar, odor = sar_odor
                sar = self.month_to_int(a_month=sar)
                jil = 2023

                return datetime(jil, sar, int(odor)).date()

        else:
            pass


    def month_to_int(self, a_month):

        if a_month.istitle():
            a_month = a_month[:3].lower()

        list_of_months = {'jan': 1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                          'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

        if a_month in list_of_months:
            return list_of_months[a_month]
        else:
            return False


    def concat_pkl_files(self, folder_path, char: str=None)-> pd.DataFrame:
        '''reads and returns char files in a given folder'''
        pkls = list()
        for file in os.listdir(folder_path):
            if file.startswith(char):
                file_path = os.getcwd()
                pkl_file = file
                filename = os.path.join(file_path, filename)
                pkl_table = pd.read_pickle(filename)
                pkls.append(pkl_table)

        return pd.concat(pkls, axis=0)


    def create_dataframe(self, df: pd.DataFrame=None, new_filename: str=False):

        output = {'id':[], 'url':[], 'transcript':[], 'head':[]}

        seeking_alpha = 'https://seekingalpha.com'

        for i in range(df.shape[0]):
            url = ''.join((seeking_alpha, df.loc[i, 'url']))
            text_content = self.get_single_transcript(url=url)
            text_head = text_content[0]

            output['id'].append(i)
            output['url'].append(url)
            output['transcript'].append(text_content)
            output['head'].append(text_head)

            if i%20 == 0:
                print(f'{i} is Done!!')

        print('All done.')

        if new_filename:
            pd.DataFrame(output).to_pickle(f'{new_filename}.pkl')
        else:
            return pd.DataFrame(output)


    def tag_missing_info(self, text_string) -> str:

        retu_strin = ''

        text_string = text_string.replace('\n', '')
        if text_string.startswith('The following audio'):
            retu_strin = 'NoScript'
        elif text_string.startswith('The audio will'):
            retu_strin = 'NoScript'
        elif text_string.startswith('Company Participants'):
            retu_strin = 'NoScript'
        elif text_string.startswith('Earnings Conference Call Start'):
            retu_strin = 'NoScript'
        elif text_string.startswith('Executives'): # two_145002_155002.pkl
            retu_strin = 'NoScript'

        elif text_string.startswith('Call Start'):
            retu_strin = 'Missing'
        elif text_string.startswith('Start Time'):
            retu_strin = 'Missing'
        elif text_string.startswith('Call End:'):
            retu_strin = 'Missing'
        elif text_string.startswith('Call End'):
            retu_strin = 'Missing'

        else:
            retu_strin = text_string


        return retu_strin

    def retrieve_missing_info(self, df: pd.DataFrame=None) -> pd.DataFrame:

        output = []

        df_copy = df[['transcript', 'tagged']].copy()
        # iterrate through rows
        for index, row in df_copy.iterrows():
            trans = row['transcript']
            tag = row['tagged']

            if tag == 'NoScript':
                output.append('NoScript')

            elif tag == '':
                output.append('NoScript')

            elif tag == 'Missing':
                output.append(' '.join([item.replace('\n', '') for item in trans[1:5] ]))

            else:
                output.append(tag)

        df['tagged'] = output

        return df



    def get_ticker_info(self, tagged_text: str) -> list:

        need_toremove_chars = ['publ', 'cayman', 'holdings', 'holding', 'de', 'USA', 'US', 'JOHN']

        state_codes = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT","DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY","LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

        cate = self.stock_cate_list

        stock, ticker = 'noInfo','noInfo'
        output = []

        # make sure type is str
        if isinstance(tagged_text, str):

            if tagged_text == 'NoScript': # no data
                output = [stock, ticker]

            else:
                # primary text cleaning happens here
                matches = re.findall(r'\((.*?)\)', tagged_text)
                splitted = re.split(r'[:,]', ','.join(matches))

                # no info
                if splitted == ['']:
                    output = [stock, ticker]

                elif len(splitted) == 1:     # only ticker info
                    output = [stock, ''.join(splitted)]

                elif len(splitted) == 3:    #
                    splitted = [s for s in splitted if not s.istitle()]  #remove chars with first letter is title. ex: Brazil
                    splitted = [w for w in splitted if w not in need_toremove_chars]
                    #print('or', splitted)
                    if 'ADR' in splitted:
                        adr_index = splitted.index('ADR')
                        stock_index_adr = adr_index+1
                        stock_adr = '-'.join((splitted[adr_index],splitted[stock_index_adr]))
                        output = [stock_adr, splitted[-1]]
                        #print('ADR', output)

                    elif 'REIT' in splitted:
                        reit_index = splitted.index('REIT')
                        stock_index_reit = reit_index+1
                        stock_reit = '-'.join((splitted[reit_index], splitted[stock_index_reit]))
                        output = [stock_reit, splitted[-1]]
                        #print('REIT', output)

                    else:
                        #splitted = [u for u in splitted if u != 'JOHN']
                        output = splitted

                elif len(splitted) > 3:
                    splitted = [s for s in splitted if not s.istitle()]  # remove chars with first letter is in title such as Brazil
                    splitted = [s for s in splitted if not s.strip() in state_codes]
                    output = splitted

                elif len(splitted) == 2:
                    #print(splitted, '*************************')
                    #splitted = [s for s in splitted if not s.istitle()]
                    #if len(splitted) == 1:
                    #    print(splitted, '&&&&&&&&&&&&&&&&&&&')
                    #    splitted = [stock, ''.join(splitted) ]
                    #    print(splitted, '#####################')
                    #print([s for s in splitted if any(k in s for k in cate)], splitted, '%%%%%%')

                    output = splitted

                else:
                    #print(splitted)
                    output = splitted
                    #print(splitted)
                    #splitted = [s for s in splitted if not s.istitle()]
                    #print(splitted)
        else:
            #
            print(tagged_text) #seems everything is string type

        return output


    def get_stock(self, stock_ticker: list)-> str:

        #return stock_ticker

        cate = self.stock_cate_list

        return stock_ticker[0] if stock_ticker[0] in cate else 'noInfo'

    @property
    def stock_cate_list(self) -> list:
        cate = ['NYSE','NASDAQ',
                'OTCPK', 'OTC','OTCQX','OTCQB',
                'NYSEMKT','NYSEARCA',
                'BATS',
                'ADR-NYSE','ADR-NASDAQ', 'ADR-OTC','ADR-OTCPK','ADR-OTCQX', 'ADR',
                'REIT-NYSE','REIT-NASDAQ']
        return cate


    def simple_moving_average(self, df: pd.DataFrame = None, create=False, filename=None) -> pd.DataFrame:

        output = []

        # make copy
        df_copied = df.copy()

        # add columns for future data
        #df_copied['close'] = np.repeat(np.nan, df_copied.shape[0])
        #df_copied['volume'] = np.repeat(np.nan, df_copied.shape[0])
        #df_copied['sma30_close'] = np.repeat(np.nan, df_copied.shape[0])
        #df_copied['sma100_close'] = np.repeat(np.nan, df_copied.shape[0])
        #df_copied['sma30_vol'] = np.repeat(np.nan, df_copied.shape[0])
        #df_copied['sma100_vol'] = np.repeat(np.nan, df_copied.shape[0])

        # date time objects
        days_go_back = timedelta(days=300)
        two_days_later = timedelta(days=2)
        one_day: str = '1d'

        # extract object from yfinance

        close_volume = ['Close', 'Volume']

        for index, row in df_copied.iterrows():

            print(f'Now printing {index}.###########')

            # NO TRANSCRIPT
            if row['tick'] == 'noInfo':
                output.append(np.array([np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]))

            # YES TRANSCRIPT
            else:

                # when instance of datetime
                if isinstance(row['meeting_date'], date):

                    # data
                    ticker = yfin.Ticker(row['tick'])
                    start_date = row['meeting_date'] - days_go_back
                    end_date = row['meeting_date'] + two_days_later


                    # data
                    try:
                        data = ticker.history(start=start_date, end=end_date, period=one_day)[close_volume]
                        if data.shape[0] != 0:
                            data.index = data.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d"))

                            # close
                            data['sma50_close'] = data['Close'].rolling(50).mean()
                            data['sma200_close'] = data['Close'].rolling(200).mean()

                            # volume
                            data['sma50_vol'] = data['Volume'].rolling(50).mean()
                            data['sma200_vol'] = data['Volume'].rolling(200).mean()

                            data = data.dropna()

                            date_str = row['date'].strftime('%Y-%m-%d')

                            output.append(data[data.index == date_str].values)

                        else:

                            print('meeting no data', data.shape[0])

                    except:
                        raise
                        print(row['tick'])#, data.shape)



                    # following meeting date was not datetime object.
                    # the meeting date is extracted from the transcript.
                    # it not presentm, it will get datetime from date column instead
                else:
                    # initiate ticker and its datetime object
                    ticker = yfin.Ticker(row['tick'])
                    start_date = row['date'] - days_go_back
                    end_date = row['date'] + two_days_later

                    # data
                    try:
                        data = ticker.history(start=start_date, end=end_date, period=one_day)[close_volume]

                        if data.shape[0] != 0:
                            data.index = data.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d"))

                            # close
                            data['sma50_close'] = data['Close'].rolling(50).mean()
                            data['sma200_close'] = data['Close'].rolling(200).mean()

                            # volume
                            data['sma50_vol'] = data['Volume'].rolling(50).mean()
                            data['sma200_vol'] = data['Volume'].rolling(200).mean()

                            data = data.dropna()

                            date_str = row['date'].strftime('%Y-%m-%d')

                            output.append(data[data.index == date_str].values)

                        else:

                            print(data.shape[0])

                    except:
                        raise
                        print(row['tick'])#, data.shape)

        return output


    def text_to_datetime(self, dataframe: pd.DataFrame=None, num_of_rows: int=None) -> pd.DataFrame:

        """
        extracts datetime object of the meeting date from the transcript.

        """

        # create column that the extracted meeting date will be placed
        meeting_date = [d for d in range(dataframe.shape[0]) ]
        dataframe['meeting_date'] = meeting_date


        # Use regular expressions to extract the date information
        pattern = r'([A-Za-z]+) (\d{1,2}), (\d{4})'


        df_copy = dataframe[['date', 'tagged']].copy()

        for index, row in df_copy.loc[:num_of_rows, :].iterrows():
            tag = row['tagged']
            recorded_date = row['date']

            # when tagged element
            if tag != 'NoScript':

                matches = re.findall(pattern, tag)

                # check whether there are more than or equal to 2 dates
                matches_len = len(matches)

                if matches_len > 1:

                    # if matches are two dates, get last date
                    matches = matches[1]

                    month, day, year = matches
                    month_int = self.month_to_int(month)
                    day_int = int(day)
                    year_int = int(year)
                    new_d = datetime(year_int, month_int, day_int).date()

                    dataframe.loc[index, 'meeting_date'] = new_d

                elif matches_len == 1:

                    # datetime must be month, day, and year format
                    if len(matches[0]) == 3:

                        month, day, year = matches[0]
                        if year == '0000':
                            # a case when year is '0000', use recorded date
                            dataframe.loc[index, 'meeting_date'] = recorded_date

                        else:
                            month_int = self.month_to_int(month)
                            day_int = int(day)
                            year_int = int(year)
                            new_d = datetime(year_int, month_int, day_int).date()

                            dataframe.loc[index,'meeting_date'] = new_d

                    else:
                        print(index, 'something went wrong. Check the format ')

            # when no tagged
            else:
                dataframe.loc[index, 'meeting_date'] = np.nan

        return dataframe


    def get_daily_moving_average(self, ticker: str='AAPL', meeting_date: date = datetime(2020,2,2)) -> pd.DataFrame:
        '''
        Example:
        ticker: 'AAPL'
        meeting_date: datetime.date, '2020-01-01'
        id_vars = datetime.date
        var names: close, DMA50, and DMA200 from wide to long format
        value: simple moving average and close price of ticker
        returns: pd.melted dataframe
        '''
        days_go_back = timedelta(700)
        twenty_days_later = timedelta(20)

        start = meeting_date - days_go_back
        end = meeting_date + twenty_days_later
        one_day = '1d'

        ticker = yfin.Ticker(ticker)

        data = ticker.history(start=start,  end=end, period=one_day)[['Close', 'Volume']]
        data = data.reset_index(inplace=False)
        data['Date'] = data['Date'].apply(lambda x: x.date())

        # Add features: simple moving average 50 vs. 200 days -> for smooting
        data['DMA50'] = data['Close'].rolling(50).mean()
        data['DMA200'] = data['Close'].rolling(200).mean()

        print(f'start: {start}')
        print(f'meeting_date: {meeting_date}')
        print(f'end: {end}')
        print(f'Ticker: {ticker}')

        data_table = data[['Date', 'Close', 'DMA50', 'DMA200']]

        data_table = data_table.dropna()
        final = data_table[data_table['Date'] == meeting_date].iloc[0]

        return final


    def clean_text_data(self, review_text):

        review_letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
        review_lower_case = review_letters_only.lower()
        review_words = review_lower_case.split()
        stop_words = stopwords.words('english')
        meaningful_words = [w for w in review_words if w not in stop_words]
        drop_2_or_less = [m for m in meaningful_words if len(m) >= 3]

        return (' '.join(drop_2_or_less))

    #test0005['clean'] = test0005['transcript'].apply(lambda x: ' '.join([sent.strip() for sent in x]))
    #test0005['clean'] = test0005['clean'].apply(clean_text_data)
    
    def split_earnings(self, dataframe: pd.DataFrame=None):

        output = []

        for r in dataframe.to_dict('records'):
            # id -er
            tagged = r['tagged']
            if tagged == 'NoScript':
                output.append({
                    'presentation':'NoScript',
                    'qa_session':'NoScript'})

            else:
                #print(row['transcript'])
                script = [line.lower().replace('\n', '') for line in r['transcript'] ]
                
                # This pattern will match all the mentioned variations
                patterns = ['question-and-answer session', \
                    'questions-and-answers session', \
                    'question and answer session', \
                    'questions and answers session', \
                    'question-and-answer session ']
                
                qa_index = None
                for inde, line in enumerate(script):
                    if line in patterns:
                        qa_index = inde
                        break

                if qa_index is not None:
                
                    presentation = script[:qa_index]
                    qa_session = script[qa_index:]

                    output.append({
                        'presentation':presentation,
                        'qa_session':qa_session})
                
                else:
                    presentation = script
                    qa_session = 'no_qa_session'
                    output.append({
                        'presentation':presentation,
                        'qa_session': qa_session})
        
        dataframe = pd.concat([dataframe, pd.DataFrame(output)], axis=1)
        return dataframe
    
    def split_text_using_newline(self, text):
        #text = "MTY Food Group Inc. (OTCPK:MTYFF) Q1 2022 Earnings Conference Call April 8, 2022 8:30 AM ET\nCompany Participants\nEric Lefebvre - Chief Executive Officer\nRenee St-Onge - Chief Financial Officer\nConference Call Participants\n"

        # Split the text into lines using '\n'
        lines = text.split('\n')

        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]

        # Join the lines with newline characters to create the new format
        new_text = '\n'.join(lines)
    
        return new_text
    
    
    def get_closing_data(self, ticker: str='AAPL', start_date: date = datetime(2020,2,2), meeting_date: date = datetime(2023,2,2) ) -> pd.DataFrame:
        '''
        Example: 
        ticker: 'AAPL'
        meeting_date: datetime.date, '2020-01-01' 
        id_vars = datetime.date 
        var names: close, DMA50, and DMA200 from wide to long format
        value: simple moving average and close price of ticker
        returns: pd.melted dataframe 

        
        '''
        days_go_back = timedelta(700)
        two_days_later = timedelta(20)

        #m_day = start_date.date()

        start = start_date - days_go_back
        end = meeting_date + two_days_later
        one_day = '1d'

        ticker = yfin.Ticker(ticker)

        data = ticker.history(start=start,  end=end, period=one_day)[['Close', 'Volume']]
        data = data.reset_index(inplace=False)
        data['Date'] = data['Date'].apply(lambda x: x.date())
        
        # Add features: simple moving average 50 vs. 200 days -> for smooting
        data['50DMA'] = data['Close'].rolling(50).mean()
        data['200DMA'] = data['Close'].rolling(200).mean()

        print(f'start date: {start}')
        print(f'meeting date: {meeting_date}')
        print(f'ending date: {end}')
        print(f'Ticker: {ticker}')
        
        data_table = data[['Date', 'Close', '50DMA', '200DMA']] 

        data_table = data_table.dropna()

        
        data_melted = pd.melt(data_table, id_vars='Date', var_name='Type', value_name='Price')

        return data_melted


    def get_single_ticker(self, dataframe: pd.DataFrame=None, tick: str=None) -> pd.DataFrame:
        data = dataframe[dataframe['tick'] == tick].copy()
        data = data.sort_values(by='meeting_date')
        data.reset_index(drop=True, inplace=True)
    
        min_date = data.meeting_date.min()
        max_date = data.meeting_date.max()
    
        closing = self.get_closing_data(ticker=tick, start_date=min_date.date(), meeting_date = max_date.date() )
    
        return closing