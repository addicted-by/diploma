import os
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.parse as urlparse
from bs4 import BeautifulSoup
import re
import pandas as pd
from googleapiclient.discovery import build
import json
from csv import writer
from urllib.request import urlopen
from urllib.parse import urlencode
import os
import time


class YouTubeScrapper():
    # Constructor
    def __init__(self,
                    webdriver_path:str='./',
                    browser={'edge', 'chrome'},
                    output_path:str='', headless:bool=True):
        assert browser in {'edge', 'chrome'}, 'browser not comparable'
        self.scrapper_class = 'YouTube scrapper'
        options = webdriver.edge.options.Options() if browser == 'edge' \
                                                else webdriver.chrome.options.Options()
        if headless:
            options.add_argument("--headless")
        try:
            if webdriver_path == './':
                self.driver = webdriver.Edge(options=options) if browser == 'edge' \
                                        else webdriver.Chrome(options=options)
            else:
                self.driver = webdriver.Edge(webdriver_path, 
                                    options=options) if browser=='edge' \
                            else webdriver.Chrome(webdriver_path, options=options) 
        except:
            print("webdriver can't be found with that input path")
        self.output_path = output_path

    def get_video_id(self, link):
        query = urlparse.urlparse(link)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = urlparse.parse_qs(query.query)
                return p['v'][0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
        # fail?
        return None

    def scroll_to_bottom(self, limit=None, time_sleep:int=1.5):
        last_height = self.driver.execute_script('\
        return window.scrollY')
        current = 1
        limit_check = lambda x: True if x > limit else False
        while True:
            from selenium.webdriver.common.keys import Keys
            html = self.driver.find_element_by_tag_name('html')
            html.send_keys(Keys.END)
            time.sleep(time_sleep)
            new_height = self.driver.execute_script('\
                return window.scrollY')
            if last_height == new_height:
                break
            if limit:
                if limit_check(current):
                    break
                current += 1
            last_height = new_height
        
    def from_string_to_int(self, fol_string: str):
        tens = {'K': 10e2, 'M': 10e5, 'B': 10e8, 'k': 10e2, 'm': 10e5, 'b': 10e8}
        if (fol_string[-1] != 'K' and fol_string[-1] != 'M' 
                        and fol_string[-1] != 'k' and fol_string[-1] != 'm'
                        and fol_string[-1] != 'b' and fol_string[-1] != 'B'):
            return int(fol_string)
        f = lambda x: int(float(x[:-1])*tens[x[-1]])
        return f(fol_string)
    # get likes method
    def get_likes(self) -> int:
        fol_string = self.soup.find(id='menu-container').find(id='text').text
        return self.from_string_to_int(fol_string=fol_string)
    #get views
    def get_views(self) -> int:
        fol_string = self.soup.find_all(
            'div', id='info')[-1].find(
                'div', id='count').find(
                    class_=re.compile(r'short*')).text.split()[0]
        return self.from_string_to_int(fol_string=fol_string)
    #scrape method
    def scrape_comments(self, link: str, save=True,
                            scroll_limit=None, time_sleep:int=3):
        self.driver.get(link)
        time.sleep(time_sleep//2)
        self.soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        del self.driver
        video_info = {
            'video_id': self.get_video_id(link=link),
            'link': link,
            'likes': self.get_likes(),
            'views': self.get_views(),
        }
        df_video = pd.DataFrame(video_info, index=['video'])
        if save: 
            df_video.to_csv(os.path.join(self.output_path, 'video_info.csv'), index=False)
        return df_video



class YouTubeAPIscrapper():
    def __init__(self,
     key="AIzaSyAND6aT1SjcmWu6abbWVsP6ztfRShkERyM", api_version:str = 'v3'):
        YOUTUBE_API_SERVICE_NAME = "youtube"
        YOUTUBE_API_VERSION = api_version
        self.service = build(YOUTUBE_API_SERVICE_NAME,YOUTUBE_API_VERSION,developerKey=key)

    def get_comments(self, part='snippet', 
                    maxResults = 100,
                    textFormat='plainText',
                    order='time',
                    videoId='UrqqpUQtFwc',
                    output_path = "./",
                    csv_filename="data.csv"):

        #3 create empty lists to store desired information
        comments, authors,sources, dates, likes, replies = [], [], [], [], [], []
        # build our service from path/to/apikey
        # service = build_service()
        
        #4 make an API call using our service
        response = self.service.commentThreads().list(part=part, maxResults=maxResults,
                                textFormat=textFormat,
                                order=order, videoId=videoId).execute()
        pd.DataFrame(columns=['Sources', 'Date', 'Author name', 'Comments',
                                'Likes', 'Replies']).to_csv(os.path.join(output_path, csv_filename), index=False)
        while response: # this loop will continue to run until you max out your quota
            for item in response['items']:
                #4 index item for desired data features
                comment1 = item['snippet']['topLevelComment']['snippet']
                comment = comment1['textDisplay'].replace('\n', '')
                author = comment1['authorDisplayName']
                date = comment1['publishedAt']
                source = comment1['videoId']
                likes_ = comment1['likeCount']
                replies_ = item['snippet']['totalReplyCount']

                
                #4 append to lists
                comments.append(comment)
                authors.append(author)
                sources.append(source)
                dates.append(date)
                likes.append(likes_)
                replies.append(replies_)
            

                #7 write line by line
                with open(os.path.join(output_path, csv_filename),'a+',encoding='utf-8-sig') as f:
                    # write the data in csv file with colums(source, date, author, text of comment)
                    csv_writer = writer(f)
                    csv_writer.writerow([source,date,author,comment, likes_, replies_])
                    
                #8 check for nextPageToken, and if it exists, set response equal to the JSON response
            if 'nextPageToken' in response:
                response = self.service.commentThreads().list(
                    part=part,
                    maxResults=maxResults,
                    textFormat=textFormat,
                    order=order,
                    videoId=videoId,
                    pageToken=response['nextPageToken']
                ).execute()
            else:
                break
            

        #9 return our data of interest
        return {
            'Sources' : sources,
            'Date': dates,
            'Author name': authors,
            'Comments': comments,
            'Likes': likes,
            'Replies': replies
        }   